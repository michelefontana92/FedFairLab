from .client_base import BaseClient
import ray
from loggers.wandb_logger import WandbLogger
from callbacks import EarlyStopping, ModelCheckpoint
from wrappers import TorchFedFBWrapper
from functools import partial
from .client_factory import register_client
import torch
import os

@register_client("client_fedfb")
@ray.remote(num_cpus=1)
class ClientFedFB(BaseClient):
    
    def __init__(self, **kwargs):
        
        self.config = kwargs.get('config')
        self.model = kwargs.get('model') 
        self.data = kwargs.get('data')
        assert self.model is not None, "Model is required"
        assert self.data is not None, "Data is required"
        self.log_model = kwargs.get('log_model', False)
        self.client_name = kwargs.get('client_name', 'client_fedavg')
       
        
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
        
        self.client_checkpoints = kwargs.get('client_callbacks')
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
        print(f"[INFO] Initializing ClientFedFB with id: {self.client_name}, metrics: {self.metrics}, monitor: {self.checkpoint_config['monitor']}, num_epochs: {self.num_epochs}")
        print(self.checkpoint_config['monitor'])
        self.local_model_checkpoint = partial(ModelCheckpoint,
                    save_dir=self.checkpoint_dir,
                    save_name = self.checkpoint_name,
                    monitor=self.checkpoint_config['monitor'],
                    mode=self.checkpoint_config['mode'])()
        self.training_group_name = kwargs.get('training_group_name')

    def setup(self,**kwargs):
        self.global_model_ckpt_path= kwargs.get('global_model_ckpt_path')
        self.wrapper = TorchFedFBWrapper(
            model=self.model,
            optimizer=self.optimizer_fn(self.model.parameters()),
            loss=self.loss,
            data_module=self.data,
            logger=self.logger,
            checkpoints=[checkpoint_fn() for checkpoint_fn in self.callbacks_fn],
            metrics=self.metrics,
            num_epochs=self.num_epochs,
            verbose=self.verbose,
            training_group_name=self.training_group_name,
            local_model_checkpoint=self.local_model_checkpoint
           )

    def get_groups_ids(self):
       gids = self.data.get_group_ids()[self.training_group_name]
       try:
        if isinstance(gids, (list, tuple)):
               group_ids = torch.unique(torch.cat([g if isinstance(g, torch.Tensor) else torch.tensor(g)
                                                  for g in gids], dim=0)).tolist()
        else:
               group_ids = gids.tolist()
       except Exception:
           group_ids = list(set(sum([g.tolist() for g in gids], []))) if isinstance(gids, (list, tuple)) else list(set(gids))
       return {'group_ids': group_ids[0]}
    
    def get_group_cardinality(self,**kwargs):
        y = kwargs.get('y')
        group_id = kwargs.get('group_id')
        #print('CARDINALITIES: ',self.data.get_group_cardinality(y,group_id,self.training_group_name))
        return {'group_cardinality': self.data.get_group_cardinality(y,group_id,self.training_group_name)}
    
    def update(self,**kwargs):
        global_model = kwargs.get('global_model')
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        lambdas = kwargs.get('lambdas')
        group_cardinality_vector = kwargs.get('group_cardinality_vector')
        
        
        self.wrapper.reset(self.optimizer_fn,self.callbacks_fn)
        self.wrapper.set_params(global_model)
        new_model =self.wrapper.fit(lambdas=lambdas,
                         group_cardinality_vector=group_cardinality_vector)
        

        result = {
            'weight': len(self.wrapper.data_module.datasets['train']),
            'params': self.wrapper.get_params()
        }

        #kwargs['model'] = new_model
        #self._eval_and_log(**kwargs)
        return result
    
    def inference(self,**kwargs):
        global_model = kwargs.get('global_model')
        fair_metric = kwargs.get('fair_metric', 'demographic_parity')
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        group_cardinality_vector = kwargs.get('group_cardinality_vector')
        group_cardinality = kwargs.get('group_cardinality')
        
        #self.wrapper.reset(self.optimizer_fn,self.callbacks_fn)
        self.wrapper.set_params(global_model)
        fair_z=self.wrapper.inference(group_cardinality_vector=group_cardinality_vector,
                                      group_cardinality=group_cardinality,
                                      fair_metric=fair_metric)
        return fair_z
    
    
    def fine_tune(self,**kwargs):
        global_model = kwargs.get('global_model')        
        assert isinstance(global_model,torch.nn.Module), "global_model must be a torch.nn.Module"
        lambdas = kwargs.get('lambdas')
        assert lambdas is not None, "lambdas must be provided"
        group_cardinality_vector = kwargs.get('group_cardinality_vector')
        assert group_cardinality_vector is not None, "group_cardinality_vector must be provided"
        local_model = torch.load(self.checkpoint_path)
        models = [global_model,local_model]
        
        self.wrapper.reset(self.optimizer_fn,self.callbacks_fn)
        best_idx = self.wrapper.model_checkpoint(models)
        best_model = models[best_idx]
        if isinstance(best_model,dict):
            self.wrapper.set_params_from_dict(best_model)
        else:
            self.wrapper.set_params(best_model)
        self.wrapper.fit(num_epochs=self.fine_tune_epochs,
                         lambdas=lambdas,
                         group_cardinality_vector=group_cardinality_vector)
        local_scores = self._evaluate_local_model()
        self.wrapper.logger.log(local_scores)

    def evaluate(self,**kwargs):
        return self._eval_and_log(**kwargs)
    
    def _get_final_results(self,**kwargs):
        checkpoint = self.client_checkpoints[0]
        assert isinstance(checkpoint,ModelCheckpoint), "Checkpoint must be an instance of ModelCheckpoint"
        best_results = self.load(checkpoint.get_model_path())
        file_path = checkpoint.get_model_path()
        best_metrics = best_results['metrics']
        best_model_params = best_results['model_state_dict']
        return best_model_params,best_metrics,file_path
    
    def _log_final_results(self,**kwargs):
        _,metrics,path = self._get_final_results(**kwargs)
        final_results = {}
        for key,v in metrics.items():
            final_results[f'final_{key}'] = v
        self.logger.log(final_results)
        self.logger.log_artifact(f'{self.client_name}_local_model',path)
    
    
    def shutdown(self,**kwargs):
        log_results = kwargs.get('log_results',True)
        if log_results:
            self._log_final_results(**kwargs)
        self.logger.close()

    def save(self, metrics,path):
        save_dict = {
            'model_state_dict': self.model,
            'metrics': metrics}
        torch.save(save_dict, path)
    
    def load(self, path):
      return torch.load(path)
    
    
    def _eval_and_log(self,**kwargs):
        model = kwargs.get('model')
        assert isinstance(model,torch.nn.Module), "model must be a torch.nn.Module"
        self.wrapper.set_params(model)
        train_scores=self.wrapper.score(
            self.wrapper.data_module.train_loader_eval(),
            self.metrics)
        val_scores=self.wrapper.score(
            self.wrapper.data_module.val_loader(),
            self.metrics)
        metrics = {}
        for metric in train_scores.keys():
            metrics[f'train_{metric}'] = train_scores[metric]
            metrics[f'val_{metric}'] = val_scores[metric]
        
        for checkpoint in self.client_checkpoints:
            if isinstance(checkpoint,ModelCheckpoint):
                model_checkpoint = checkpoint(save_fn=partial(self.save,metrics), metrics=metrics)
                metrics['model_checkpoint'] = 1 if model_checkpoint else 0
        self.logger.log(metrics)
        return {'train':train_scores,'val':val_scores}