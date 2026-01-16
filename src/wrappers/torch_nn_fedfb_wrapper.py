#Define the torch_nn_wrapper.py file, which contains the TorchNNWrapper class.
# This class is a wrapper for a PyTorch neural network model. It inherits from the BaseWrapper class, which is defined in the base_wrapper.py file.
# The TorchNNWrapper class implements the fit, predict, predict_proba, score, save, load, get_params, set_params, get_feature_names, get_feature_types, get_feature_count, get_target_names, get_target_count, get_classes, and get_class_count methods.
from wrappers.base_wrapper import BaseWrapper
import torch 
import tqdm
import numpy as np
from loggers import BaseLogger
from callbacks import EarlyStopping, ModelCheckpoint
from dataloaders import DataModule
from loggers import WandbLogger,BaseLogger,FileLogger
from dataloaders import DataModule
from metrics import BaseMetric,Performance,GroupFairnessMetric
import tqdm
from .torch_nn_wrapper import TorchNNWrapper
import copy 

class EarlyStoppingException(Exception):
    pass

class TorchFedFBWrapper(TorchNNWrapper):
    
    def __init__(self, *args, **kwargs):
        super(TorchFedFBWrapper,self).__init__(*args, **kwargs)
        self.training_group_name = kwargs.get('training_group_name')
        self.local_model_checkpoint:ModelCheckpoint = kwargs.get('local_model_checkpoint')

    def _training_step(self,batch,batch_idx):
        self.model.train()
        inputs = batch['data'] 
        targets = batch['labels']
        weight_vector = torch.ones_like(targets).double()
        group_idx = {}
        for y,z in self.lambdas.keys():
            group_idx[(y,z)] = torch.where((targets == y) & (batch['groups'][self.training_group_name] == z))[0]
            weight_vector[group_idx[(y,z)]] = self.lambdas[(y,z)] / self.group_cardinality_vector[z]
        inputs = inputs.float().to(self.device)
        targets = targets.long().to(self.device)
        weight_vector = weight_vector.float().to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        loss = torch.mean(loss*weight_vector)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _validation_step(self,batch,batch_idx):
        self.model.eval()
        with torch.no_grad():
            inputs = batch['data'] 
            targets = batch['labels']
            weight_vector = torch.ones_like(targets).double()
            group_idx = {}
            for y,z in self.lambdas.keys():
                group_idx[(y,z)] = torch.where((targets == y) & (batch['groups'][self.training_group_name] == z))[0]
                weight_vector[group_idx[(y,z)]] = self.lambdas[(y,z)] / self.group_cardinality_vector[z]
            inputs = inputs.float().to(self.device)
            targets = targets.long().to(self.device)
            weight_vector = weight_vector.float().to(self.device)
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss = torch.mean(loss*weight_vector)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
           
        return loss.item(), outputs, targets, predictions
    
    
    # Fit the model
    def fit(self,**kwargs):
        self.lambdas = kwargs.get('lambdas')
        assert self.lambdas is not None, "lambdas should be provided"
        self.group_cardinality_vector = kwargs.get('group_cardinality_vector')
        assert self.group_cardinality_vector is not None, "group_cardinality_vector should be provided"
        self.loss = self.loss_fn(reduction='none')
        num_epochs = kwargs.get('num_epochs',-1)
        disable_log = kwargs.get('disable_log',True)
        n_rounds = self.num_epochs if num_epochs == -1 else num_epochs
        self.model.to(self.device)
        
        
        # self.model_checkpoint([self.model])

        try:
            for epoch in tqdm.tqdm(range(n_rounds)):
                train_loss = 0
                val_loss = 0
                train_loader = self.data_module.train_loader()
                val_loader = self.data_module.val_loader()
                train_loader_eval = self.data_module.train_loader_eval()
                for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
                    train_loss += self._training_step(batch, batch_idx)
                
                train_loss /= len(train_loader)
                
                train_outputs = []
                train_targets = []
                train_predictions=[]
                train_groups = []
                
                val_outputs = []
                val_targets = []
                val_predictions = []
                val_groups = []
                for batch_idx, batch in enumerate(val_loader):
                    loss, outputs, targets,predictions = self._validation_step(batch, batch_idx)
                    val_loss += loss
                    val_outputs.append(outputs)
                    val_targets.append(targets)
                    val_predictions.append(predictions)
                    val_groups.append(batch['groups'])
                val_loss /= len(val_loader)
                
                for batch_idx, batch in enumerate(train_loader_eval):
                    _, outputs, targets,predictions = self._validation_step(batch, batch_idx)
                    train_outputs.append(outputs)
                    train_targets.append(targets)
                    train_predictions.append(predictions)
                    train_groups.append(batch['groups'])
                
                
                val_outputs = torch.cat(val_outputs, dim=0)
                val_targets = torch.cat(val_targets, dim=0).detach().cpu()
                val_predictions = torch.cat(val_predictions, dim=0).detach().cpu()
                val_groups_dict = {group_name:torch.cat([batch[group_name] for batch in val_groups],dim=0).detach().cpu() for group_name in val_groups[0].keys()}
                train_outputs = torch.cat(train_outputs, dim=0)
                train_targets = torch.cat(train_targets, dim=0).detach().cpu()
                train_predictions = torch.cat(train_predictions, dim=0).detach().cpu()
                train_groups_dict = {group_name:torch.cat([batch[group_name] for batch in train_groups],dim=0).detach().cpu() for group_name in train_groups[0].keys()}

                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }

                metrics.update(self._compute_metrics(self.metrics,
                                                    val_predictions,
                                                    val_targets,
                                                    val_groups_dict,
                                                    prefix='val',
                                                    logits=val_outputs))
                metrics.update(self._compute_metrics(self.metrics,
                                                    train_predictions,
                                                    train_targets,
                                                    train_groups_dict,
                                                    prefix='train',
                                                    logits=train_outputs))
                
                
                self.local_model_checkpoint(
                                save_fn=self.save, 
                                metrics=metrics)
                
                for checkpoint in self.checkpoints:
                    if isinstance(checkpoint, EarlyStopping):
                        stop,counter = checkpoint(metrics=metrics)
                        metrics['early_stopping'] = counter
                        if stop:
                            if not disable_log:
                                self.logger.log(metrics)  
                            raise EarlyStoppingException 
                       
                    elif isinstance(checkpoint, ModelCheckpoint):
                        model_checkpoint= checkpoint(
                                save_fn=self.save, 
                                metrics=metrics)
                        metrics['model_checkpoint'] = 1 if model_checkpoint else 0
                if not disable_log:
                    self.logger.log(metrics)  
        except EarlyStoppingException:
                    pass
        
        for checkpoint in self.checkpoints:
                if isinstance(checkpoint, ModelCheckpoint):
                   self.load(checkpoint.get_model_path())

        return self.model
    
    def inference(self,**kwargs):
        group_cardinality_vector = kwargs.get('group_cardinality_vector')
        group_cardinality = kwargs.get('group_cardinality')
        self.loss = self.loss_fn(reduction='sum')
        self.model.to(self.device)
        with torch.no_grad():
            self.model.eval()
            loss_yz = {}
            for y,z in self.lambdas.keys():
                loss_yz[(y,z)] = 0
            group_boolean_idx = {}    
            train_loader_eval = self.data_module.train_loader_eval()
            batch = next(iter(train_loader_eval))
            inputs = batch['data']
            targets = batch['labels']
            groups = batch['groups'][self.training_group_name]
            inputs = inputs.float().to(self.device)
            targets = targets.long().to(self.device)
            groups = groups.long().to(self.device)
            outputs = self.model(inputs)
            for y,z in self.lambdas.keys():
                group_boolean_idx[(y,z)] = (targets == y) & (groups == z) 
                loss = self.loss(outputs[group_boolean_idx[(y,z)]], targets[group_boolean_idx[(y,z)]])
                loss_yz[(y,z)] = loss
            #for y,z in loss_yz.keys():
            #    loss_yz[(y,z)] = loss_yz[(y,z)] / self.group_cardinality_vector[z]

            fair_z = {}
            for z in group_cardinality_vector.keys():
                if z==0:
                    fair_z[0] = 0
                else:
                    fair_z[z] = -loss_yz[(0,0)] + loss_yz[(1,0)] + loss_yz[(0,z)] - loss_yz[(1,z)]
                    #fair_z[z] += group_cardinality[(0,0)]/group_cardinality_vector[0]
                    #fair_z[z] -= group_cardinality[(0,z)]/group_cardinality_vector[z]
            return {'fairness_yz': fair_z,
                    'loss_yz': loss_yz}
     