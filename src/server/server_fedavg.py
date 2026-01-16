from .server_base import BaseServer
import ray

from .server_factory import register_server
from callbacks.early_stopping import EarlyStopping
from callbacks.model_checkpoint import ModelCheckpoint
from loggers.wandb_logger import WandbLogger
from functools import partial
from .aggregators import AggregatorFactory
import os
import numpy as np
import torch
import copy

class EarlyStoppingException(Exception):
    pass

@register_server("server_fedavg")
class ServerFedAvg(BaseServer):
    def __init__(self,**kwargs):
        
        self.config = kwargs.get('server_config')
        self.clients_init_fn_list = kwargs.get('clients_init_fn_list')
        self.model = kwargs.get('model')
        self.loss = kwargs.get('loss')
        self.metrics = kwargs.get('metrics')

        self.log_model = kwargs.get('log_model', False)
        self.id = kwargs.get('server_name', 'server_fedavg')
        self.project = kwargs.get('project_name', 'fedavg')
        self.checkpoint_dir = kwargs.get('checkpoint_dir','checkpoints')
        self.checkpoint_name = kwargs.get('checkpoint_name','global_model.h5')
        self.monitor = kwargs.get('monitor','val_global_f1')
        self.mode = kwargs.get('mode','max')
        self.verbose = kwargs.get('verbose', False)
        self.federated_rounds = kwargs.get('num_federated_iterations', 1)
        
        
        self.callbacks = [
            EarlyStopping(patience=self.config['early_stopping_patience'],
                          monitor=self.monitor,
                          mode=self.mode
                          ),
            ModelCheckpoint(save_dir=self.checkpoint_dir,
                            save_name = self.checkpoint_name,
                            monitor=self.monitor,
                            mode=self.mode)
                          ]
        
        self.logger = WandbLogger(
            project=self.project,
            config= self.config,
            id=self.id,
            checkpoint_dir= self.checkpoint_dir,
            checkpoint_path = self.checkpoint_name,
            data_module=self.data if self.log_model else None
        )

        self.aggregator = AggregatorFactory().create('FedAvgAggregator')
 
    def _create_clients(self,clients_init_fn_list):
        client_list = [client_init_fn() 
                for client_init_fn in clients_init_fn_list]
        return client_list
    
    
    def _broadcast_fn(self,fn_name,**kwargs):
        assert isinstance(fn_name,str), "fn_name must be a string"
        handlers = []
        results = []
        for client in self.clients:
            assert hasattr(client,fn_name), f"Client does not have {fn_name} method"
            handlers.append(getattr(client,fn_name).remote(**kwargs))
        for handler in handlers:
            results.append(ray.get(handler))
        return results
    
    def _evaluate_best_model(self):
        global_model = torch.load(self.checkpoint_path)
        self.model.load_state_dict(global_model)
        global_scores = self._evaluate_global_model(best_model=True)
        final_scores ={}
        for key,v in global_scores.items():
            final_scores[f'final_{key}'] = v
        self.logger.log(final_scores)

    def _evaluate_global_model(self):
        
        scores = self._broadcast_fn('evaluate',
                            model=copy.deepcopy(self.model))
        global_scores = {}
        for score in scores:
            for kind in score.keys():
                for metric in score[kind].keys():
                    name = f'global_{kind}_{metric}'
                    if name not in global_scores:
                        global_scores[name] = []
                    global_scores[name].append(score[kind][metric])
        for metric in global_scores:
            global_scores[metric] = np.mean(global_scores[metric])

        return global_scores

    
    def setup(self,**kwargs):
        self.checkpoint_path = os.path.join(self.checkpoint_dir,self.checkpoint_name)
        self.clients = self._create_clients(
            self.clients_init_fn_list)
        self._broadcast_fn('setup',
                           global_model_ckpt_path=self.checkpoint_path)
    
    
    def step(self,**kwargs):
        results = self._broadcast_fn('update',
                           global_model=copy.deepcopy(self.model))
        global_model = copy.deepcopy(self.model)
        new_params = self.aggregator(model=global_model,
                        params=results)
        self.model.load_state_dict(new_params)
        global_scores = self._evaluate_global_model()
        try:
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    stop,counter = callback(metrics=global_scores)
                    global_scores['global_early_stopping'] = counter
                    if stop:
                        self.logger.log(global_scores)  
                        raise EarlyStoppingException  
                elif isinstance(callback,ModelCheckpoint):
                    callback(save_fn=partial(self.save,
                                              global_scores),
                              metrics = global_scores
                              )
            self.logger.log(global_scores)
        
        except EarlyStoppingException:
            raise EarlyStoppingException 
                
    def execute(self,**kwargs):
        #global_scores = self._evaluate_global_model()
        #self.logger.log(global_scores)
        try:
            for i in range(self.federated_rounds):
                self.step(round=i)
        except EarlyStoppingException:
            pass
        #self.fine_tune()
        
    def evaluate(self,**kwargs):
        global_scores = self._evaluate_global_model()
        return global_scores
    
    def fine_tune(self,**kwargs):
        global_model = torch.load(self.checkpoint_path)
        self.model.load_state_dict(global_model)
        self._broadcast_fn('fine_tune',global_model=self.model)
        return
    
    def save(self,metrics,path):
        result_to_save = {
            'model_params': self.model.state_dict(),
            'metrics': metrics
        }
        torch.save(result_to_save, path)
        
    def log_final_results(self,**kwargs):
        for callback in self.callbacks:
          if isinstance(callback, ModelCheckpoint):
            best_results = callback.get_best_model() 
            artifact_name = 'global_model'
            artifact_path = callback.get_model_path()
            self.logger.log_artifact(artifact_name,
                                     artifact_path)
             
        metrics = best_results['metrics']
        final_scores = {}
        for key,v in metrics.items():
            final_scores[f'final_{key}'] = v
        self.logger.log(final_scores)

    def shutdown(self,**kwargs):
        log_results = kwargs.get('log_results',True)
        if log_results:
            self.log_final_results()
        
        self.logger.close()
        self._broadcast_fn('shutdown',
                           log_results=log_results)