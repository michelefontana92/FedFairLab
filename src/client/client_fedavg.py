from .client_base import BaseClient
import ray
from loggers.wandb_logger import WandbLogger
from callbacks import EarlyStopping, ModelCheckpoint
from wrappers.torch_nn_wrapper import TorchNNWrapper
from functools import partial
from .client_factory import register_client
import torch
import os

@register_client("client_fedavg")
@ray.remote
class ClientFedAvg(BaseClient):
    
    def __init__(self,**kwargs):
        self.config = kwargs.get('config')
        self.model = kwargs.get('model') 
        self.data = kwargs.get('data')
        assert self.model is not None, "Model is required"
        assert self.data is not None, "Data is required"
        self.log_model = kwargs.get('log_model', False)
        self.id = kwargs.get('client_name', 'client_fedavg')
       
        
        self.checkpoint_config = kwargs.get('checkpoint_config')
        self.checkpoint_dir = self.checkpoint_config['checkpoint_dir']
        self.checkpoint_name = self.checkpoint_config['checkpoint_name']
        self.checkpoint_path = os.path.join(self.checkpoint_dir,self.checkpoint_name)
        self.callbacks_fn = [
            partial(EarlyStopping,
                    patience=self.checkpoint_config['patience'],
                    monitor=self.checkpoint_config['monitor'],
                    mode=self.checkpoint_config['mode']
                    ),
            partial(ModelCheckpoint,
                    save_dir=self.checkpoint_dir,
                    save_name = self.checkpoint_name,
                    monitor=self.checkpoint_config['monitor'],
                    mode=self.checkpoint_config['mode'])
                          ]
        
        
        self.optimizer_name = kwargs.get('optimizer_name','Adam')
        self.optimizer_fn = partial(getattr(torch.optim,
                                            self.optimizer_name),
                                            lr=self.config['lr'])
        
        self.verbose = kwargs.get('verbose', False)
        self.fine_tune_epochs = kwargs.get('fine_tune_epochs', 20)
        self.loss = kwargs.get('loss')
        self.metrics = kwargs.get('metrics')
        self.num_epochs = kwargs.get('num_epochs')
       
        
        self.logger_fn = kwargs.get('logger')
        self.logger = self.logger_fn()

    def setup(self,**kwargs):
        self.global_model_ckpt_path= kwargs.get('global_model_ckpt_path')

        self.wrapper = TorchNNWrapper(
            model=self.model,
            optimizer=self.optimizer_fn(self.model.parameters()),
            loss=self.loss,
            data_module=self.data,
            logger=self.logger,
            checkpoints=[checkpoint_fn() for checkpoint_fn in self.callbacks_fn],
            metrics=self.metrics,
            num_epochs=self.num_epochs,
            verbose=self.verbose
           )

    def update(self,**kwargs):
        global_model = kwargs.get('global_model')
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        self.wrapper.reset(self.optimizer_fn,self.callbacks_fn)
        self.wrapper.set_params(global_model)
        self.wrapper.fit()
        
        result = {
            'weight': len(self.wrapper.data_module.datasets['train']),
            'params': self.wrapper.get_params()
        }
       

        return result
        
    def evaluate(self,**kwargs):
        eval_local_model = kwargs.get('eval_local_model', True)
        if eval_local_model:
            local_scores = self._evaluate_local_model()
        dict2send, global_scores = self._evaluate_global_model(**kwargs)
        if eval_local_model:
            global_scores.update(local_scores)
        self.wrapper.logger.log(global_scores)
        return dict2send
    
    def evaluate_best_model(self,**kwargs):
        #if os.path.exists(self.checkpoint_path):
        local_scores = self._evaluate_local_model()
        dict2send, global_scores = self._evaluate_global_model(**kwargs)
        global_scores.update(local_scores)
        final_scores = {}
        for key,v in global_scores.items():
            final_scores[f'final_{key}'] = v
        self.wrapper.logger.log(final_scores)
        return dict2send
    
    def _evaluate_global_model(self,**kwargs):
        global_model = kwargs.get('global_model')
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        self.wrapper.set_params(global_model)
        train_scores=self.wrapper.score(
            self.wrapper.data_module.train_loader_eval(),
            self.metrics)
        val_scores=self.wrapper.score(
            self.wrapper.data_module.val_loader(),
            self.metrics)
        local_metrics = {}
        for metric in train_scores.keys():
            local_metrics[f'global_train_{metric}'] = train_scores[metric]
            local_metrics[f'global_val_{metric}'] = val_scores[metric]
        return {'train':train_scores,'val':val_scores},local_metrics
    
    def _evaluate_local_model(self,**kwargs):
        local_model = torch.load(self.checkpoint_path)
        self.wrapper.set_params_from_dict(local_model)
        train_scores=self.wrapper.score(
            self.wrapper.data_module.train_loader_eval(),
            self.metrics)
        val_scores=self.wrapper.score(
            self.wrapper.data_module.val_loader(),
            self.metrics)
        local_metrics = {}
        for metric in train_scores.keys():
            local_metrics[f'local_train_{metric}'] = train_scores[metric]
            local_metrics[f'local_val_{metric}'] = val_scores[metric]
        return local_metrics

    def fine_tune(self,**kwargs):
        global_model = kwargs.get('global_model')        
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        local_model = torch.load(self.checkpoint_path)
        models = [global_model,local_model]
        self.wrapper.reset(self.optimizer_fn,self.callbacks_fn)
        best_idx = self.wrapper.model_checkpoint(models)
        best_model = models[best_idx]
        if isinstance(best_model,dict):
            self.wrapper.set_params_from_dict(best_model)
        else:
            self.wrapper.set_params(best_model)
        self.wrapper.fit(num_epochs=self.fine_tune_epochs)
        local_scores = self._evaluate_local_model()
        self.wrapper.logger.log(local_scores)

    def shutdown(self,**kwargs):
        self.wrapper.logger.close()

