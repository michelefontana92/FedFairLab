from .client_base import BaseClient
import ray
from .client_factory import register_client
from wrappers import OrchestratorWrapper
import copy
import numpy as np
import time
@register_client("client_fedfairlab")
@ray.remote
class ClientFedFairLab(BaseClient):
    


    def profile(func):
        def wrapper(*args, **kwargs):
            self = args[0] if args else None  # estrai self se presente
            name = getattr(self, 'client_name', 'Unknown')  # fallback se non esiste

            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"[PROFILE {name}] {func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper

    def __init__(self,**kwargs):
        #self.orchestrator = kwargs.get('orchestrator')
        self.client_name = kwargs.get('client_name')
        self.logger_fn = kwargs.get('logger')
        self.logger = self.logger_fn()
        self.orchestrator_fn = kwargs.get('orchestrator')
        self.num_global_epochs = kwargs.get('num_global_iterations',1)
        self.num_local_epochs = kwargs.get('num_local_iterations',1)
        self.local_model = kwargs.get('model')
        
        assert self.local_model is not None, "Model is required"
        self.state = None
        print(f"Client {self.client_name} initialized")
       
    
    def setup(self,**kwargs):
        print("Setting up client")

    ##@profile     
    def _init_orchestrator(self,**kwargs):
        problem = kwargs.get('problem')
        #print(f'Number of teachers: {len(problem["aggregation_teachers_list"])}')      
        assert problem is not None, "Problem is required"
        self.orchestrator = self.orchestrator_fn(logger=None,**problem)
        assert self.orchestrator is not None, "Orchestrator is required"
        assert isinstance(self.orchestrator, OrchestratorWrapper), "Orchestrator must be an instance of OrchestratorWrapper"
        #print(f"Client {self.client_name} problem teachers: {len(self.orchestrator.aggregation_teachers_list)}")
        current_model_params = kwargs.get('model_params')
        if current_model_params is not None:
            self.orchestrator.set_model_params(current_model_params)
       
    #@profile 
    def fit(self,**kwargs):
        
        self._init_orchestrator(**kwargs)
        #if self.state is not None:
        #    self.orchestrator.set_state(self.state)
        model_params = kwargs.get('model_params')
        self.state = {'model_state_dict': model_params}
        
        updated_model,state=self.orchestrator.fit(num_global_iterations=self.num_global_epochs,
                                            num_local_epochs=self.num_local_epochs,
                                            state=self.state)
        self.state = state
        return {'params':copy.deepcopy(updated_model.state_dict()),
                'weight':1.0}
    
    def update(self,**kwargs):
        pass
    #@profile  
    def evaluate(self,**kwargs):
        self._init_orchestrator(**kwargs)
        model_params = kwargs.get('model_params')
        assert model_params is not None, "Model parameters are required"
        results = self.orchestrator.evaluate(model_params)
        #print(f"Evaluation results: {results}")
        return results
    #@profile 
    def evaluate_constraints(self,**kwargs):
        self._init_orchestrator(**kwargs)
        model_params = kwargs.get('model_params')
        first_performance_constraint = kwargs.get('first_performance_constraint',False)
        results_dict = self.orchestrator.evaluate_constraints2(model_params)
        final_results = {'train_constraints':[],'val_constraints':[]}
        
        #for v in ['train_constraints','val_constraints']:
        for v in ['val_constraints']:
            for key,value in results_dict[v]['macro_constraints_violations'].items():
                if first_performance_constraint and key==0:
                     final_results[v].append(1-value[0])   
                else:
                    final_results[v].append(value[0])
        
        for v in results_dict.keys():
            #if v not in ['train_constraints','val_constraints']:
            if v not in ['val_constraints']:
                final_results[v] = results_dict[v]
        
        return final_results
    #@profile 
    def evaluate_constraints_list(self,**kwargs):
        self._init_orchestrator(**kwargs)
        model_params_list = kwargs.get('model_params_list')
        first_performance_constraint = kwargs.get('first_performance_constraint',False)
        results = []
        for model_params in model_params_list:
            results_dict = self.orchestrator.evaluate_constraints2(model_params)
            final_results = {'train_constraints':[],'val_constraints':[]}
            
            #for v in ['train_constraints','val_constraints']:
            for v in ['val_constraints']:
                for key,value in results_dict[v]['macro_constraints_violations'].items():
                    if first_performance_constraint and key==0:
                        final_results[v].append(1-value[0])   
                    else:
                        final_results[v].append(value[0])
            
            for v in results_dict.keys():
                #if v not in ['train_constraints','val_constraints']:
                if v not in ['val_constraints']:
                    final_results[v] = results_dict[v]
            results.append(final_results)
        #r = [r['metrics']['val_f1'] for r in results]
        #print(f'[CLIENT {self.client_name}] Evaluation results: {r}')
        #r = [r['metrics']['val_constraints_score'] for r in results]
        #print(f'[CLIENT {self.client_name}] Evaluation Score results: {r}')
        #print(f'[CLIENT {self.client_name}] Evaluation results: {results}')
        return results
    
    
    def evaluate_best_model(self,**kwargs):
        pass
    
    def _evaluate_global_model(self,**kwargs):
        pass
    
    def _evaluate_local_model(self,**kwargs):
        pass

    def fine_tune(self,**kwargs):
        pass
    
    def shutdown(self,**kwargs):
        self.logger.close()
