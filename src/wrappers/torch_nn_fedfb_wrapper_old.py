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
        weight_vector = torch.ones_like(targets, dtype=torch.float)
        group_idx = {}
        for y,z in self.lambdas.keys():
            group_idx[(y,z)] = torch.where((targets == y) & (batch['groups'][self.training_group_name] == z))[0]
            weight_vector[group_idx[(y,z)]] = float(self.lambdas[(y,z)]) / max(1.0, float(self.group_cardinality_vector[z]))
        inputs = inputs.float().to(self.device)
        targets = targets.long().to(self.device)
        weight_vector = weight_vector / (weight_vector.mean() + 1e-12)
        weight_vector = weight_vector.to(self.device)

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
            
            inputs = inputs.float().to(self.device)
            targets = targets.long().to(self.device)
            #weight_vector = weight_vector.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss = torch.mean(loss)
            predictions = torch.argmax(outputs, dim=1)
           
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
    
    def inference(self, **kwargs):
       """
        Restituisce contatori per la metrica di fairness richiesta.
        - fair_metric='demographic_parity': per ogni gruppo z => #predetti positivi (pos) e #totali (tot)
        - fair_metric='equalized_odds': per ogni gruppo z => TP, Pos (y=1), FP, Neg (y=0)
       """
       fair_metric = kwargs.get('fair_metric', 'demographic_parity')  # 'demographic_parity' oppure 'equalized_odds'
       group_cardinality_vector = kwargs.get('group_cardinality_vector')
       assert group_cardinality_vector is not None, "group_cardinality_vector richiesto"

       self.model.to(self.device)
       self.model.eval()

       with torch.no_grad():
           if fair_metric == 'demographic_parity':
               dp_pos = {int(z): 0 for z in group_cardinality_vector.keys()}
               dp_tot = {int(z): 0 for z in group_cardinality_vector.keys()}

               for batch in self.data_module.train_loader_eval():
                   inputs = batch['data'].float().to(self.device)
                   groups = batch['groups'][self.training_group_name].long().to(self.device)
                   logits = self.model(inputs)
                   yhat = torch.argmax(logits, dim=1)  # binario: classe positiva = 1
                   for z in group_cardinality_vector.keys():
                       z_int = int(z)
                       idx = (groups == z_int)
                       if idx.any():
                           dp_pos[z_int] += int((yhat[idx] == 1).sum().item())
                           dp_tot[z_int] += int(idx.sum().item())

               return {'metric': 'demographic_parity', 'dp_counts': {'pos': dp_pos, 'tot': dp_tot}}

           elif fair_metric == 'equalized_odds':
               # Equalized Odds: servono TPR e FPR per gruppo
               eod_tp  = {int(z): 0 for z in group_cardinality_vector.keys()}
               eod_pos = {int(z): 0 for z in group_cardinality_vector.keys()}  # y=1
               eod_fp  = {int(z): 0 for z in group_cardinality_vector.keys()}
               eod_neg = {int(z): 0 for z in group_cardinality_vector.keys()}  # y=0

               for batch in self.data_module.train_loader_eval():
                   inputs  = batch['data'].float().to(self.device)
                   targets = batch['labels'].long().to(self.device)   # {0,1}
                   groups  = batch['groups'][self.training_group_name].long().to(self.device)
                   logits  = self.model(inputs)
                   yhat    = torch.argmax(logits, dim=1)

                   for z in group_cardinality_vector.keys():
                       z_int = int(z)
                       idx_z = (groups == z_int)
                       if not idx_z.any():
                           continue
                       # Positivi reali (y=1) e Negativi reali (y=0)
                       idx_pos = idx_z & (targets == 1)
                       idx_neg = idx_z & (targets == 0)

                       eod_pos[z_int] += int(idx_pos.sum().item())
                       eod_neg[z_int] += int(idx_neg.sum().item())

                       # TP: y=1 & yhat=1; FP: y=0 & yhat=1
                       if idx_pos.any():
                           eod_tp[z_int] += int((yhat[idx_pos] == 1).sum().item())
                       if idx_neg.any():
                           eod_fp[z_int] += int((yhat[idx_neg] == 1).sum().item())

               return {'metric': 'equalized_odds',
                       'eod_counts': {'tp': eod_tp, 'pos': eod_pos, 'fp': eod_fp, 'neg': eod_neg}}

           else:
               raise ValueError(f"Unknown fairness metric : {fair_metric}")