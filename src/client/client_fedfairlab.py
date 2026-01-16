from .client_base import BaseClient
import ray
from .client_factory import register_client
from wrappers import OrchestratorWrapper
import copy
import numpy as np
import time
from .utils import compute_global_score
from callbacks import ModelCheckpoint
import torch 
from functools import partial
@register_client("client_fedfairlab")
@ray.remote(num_cpus=2)
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
        self.num_personalization_epochs = kwargs.get('num_personalization_iterations', 1)

        self.local_model = kwargs.get('model')
        self.client_checkpoints = kwargs.get('client_callbacks')
        assert self.local_model is not None, "Model is required"
        self.state = None
        self.orchestrator_map = {}
        
        
        print(f"Client {self.client_name} initialized")
       
    def _get_orchestrator(self, problem_name, problem_kwargs, model_params=None):
    
        if problem_name not in self.orchestrator_map:
            print(f"Initializing orchestrator for {problem_name}")
            orchestrator = self.orchestrator_fn(logger=None, **problem_kwargs)
            assert isinstance(orchestrator, OrchestratorWrapper), f"Invalid orchestrator for {problem_name}"
            self.orchestrator_map[problem_name] = orchestrator
        else: 
            print(f"Using cached orchestrator for {problem_name}")

        orchestrator = self.orchestrator_map[problem_name]
        if model_params is not None:
            orchestrator.set_model_params(model_params)
    
        return orchestrator

    def setup(self,**kwargs):
        print("Setting up client")

    ##@profile     
    def _init_orchestrator(self,**kwargs):
        problem = kwargs.get('problem')
        #print(f'Number of teachers: {len(problem["aggregation_teachers_list"])}')      
        assert problem is not None, "Problem is required"
        self.orchestrator = self.orchestrator_fn(logger=None, **problem)
        assert self.orchestrator is not None, "Orchestrator is required"
        assert isinstance(self.orchestrator, OrchestratorWrapper), "Orchestrator must be an instance of OrchestratorWrapper"
        #print(f"Client {self.client_name} problem teachers: {len(self.orchestrator.aggregation_teachers_list)}")
        current_model_params = kwargs.get('model_params')
        if current_model_params is not None:
            self.orchestrator.set_model_params(current_model_params)
       
    @profile 
    def fit(self,**kwargs):
        num_local_epochs = kwargs.get('num_local_epochs',self.num_local_epochs)
        num_global_epochs = kwargs.get('num_global_epochs',self.num_global_epochs)
        print(f"Client {self.client_name} fitting for {num_global_epochs} global epochs and {num_local_epochs} local epochs")
        #if num_local_epochs != self.num_local_epochs:
        #    print(f"Client {self.client_name} num_local_epochs: {num_local_epochs}")
        problem = kwargs.get('problem')
        #constraint = problem['inequality_constraints'][1]
        #print(f"Client {self.client_name} constraint lower bound: {constraint.lower_bound}")
        problem_name = problem['name']
        aggregation_teachers_list = problem.get('aggregation_teachers_list',[])
        model_params = kwargs.get('model_params')
       
            
        aggregation_weights =problem.get('aggregation_weights',None)
        self.orchestrator = self._get_orchestrator(problem_name, problem,model_params)

        current_state = self.state if problem_name == 'local_problem' else None
        updated_model,state=self.orchestrator.fit(
                                            model_params=model_params,
                                            num_global_iterations=num_global_epochs,
                                            num_local_epochs=num_local_epochs,
                                            state=current_state,
                                            aggregation_teachers_list=aggregation_teachers_list,
                                            aggregation_weights=aggregation_weights,)
        
        if problem_name == 'local_problem':
            self.state = copy.deepcopy(state)
            #print(f"Client {self.client_name} state: {self.state['inequality_lambdas']}")
        #print(f'Client {self.client_name} new state: {self.state}')
        return {'params':updated_model.state_dict(),
                'weight':1.0
                }
    
    def update(self,**kwargs):
        pass
    
    
    def save(self, metrics,path):
        save_dict = {
            'model_state_dict': self.model,
            'metrics': metrics}
        torch.save(save_dict, path)

    # Load the model from the specified path. The path should include the file name and extension.
    def load(self, path):
      return torch.load(path)
    
    
    def _eval_and_log(self,**kwargs):
        performance_constraint = kwargs.get('performance_constraint')
        original_threshold_list = kwargs.get('original_threshold_list')
        model_params_list = kwargs.get('model_params_list')
        eval_results = kwargs.get('eval_results')
        #print(f'[CLIENT {self.client_name}] performance_constraint: {performance_constraint}')
        #print(f'[CLIENT {self.client_name}] original_threshold_list: {original_threshold_list}')
        #print(f'[CLIENT {self.client_name}] eval_results: {eval_results}')
        best_results = {}
        best_score = -np.inf
        best_model_params = None
        for model,result in zip(model_params_list,eval_results):
            local_result = compute_global_score(
                performance_constraint=performance_constraint,
                original_threshold_list=original_threshold_list,
                eval_results=[result],
            )
            #print(f'[CLIENT {self.client_name}] Input LOCAL Evaluation results: {result}') 
            local_result['metrics']['val_constraints_score'] = local_result['metrics']['val_global_score']
            del local_result['metrics']['val_global_score']
            if local_result['metrics']['val_constraints_score'] > best_score:
                best_score = local_result['metrics']['val_constraints_score']
                best_results = copy.deepcopy(local_result)
                best_model_params = model
            #print(f'[CLIENT {self.client_name}] LOCAL Evaluation results: {local_result}')
            
            #print(f'[CLIENT {self.client_name}] LOCAL Evaluation results: {local_result}')
            
            #print(f'[CLIENT {self.client_name}] LOCAL Evaluation results: {result["metrics"]}\n{result["metrics"]["val_constraints_score"]}')
        
        #print(f'[CLIENT {self.client_name}] Best LOCAL Evaluation results: {best_results}')
        self.model = copy.deepcopy(best_model_params)
        #print(f'[CLIENT {self.client_name}] Best LOCAL Evaluation results: {best_results["metrics"]}')
        metrics = best_results['metrics']
        for checkpoint in self.client_checkpoints:
            if isinstance(checkpoint,ModelCheckpoint):
                model_checkpoint = checkpoint(save_fn=partial(self.save,metrics), metrics=metrics)
                metrics['model_checkpoint'] = 1 if model_checkpoint else 0
        self.logger.log(metrics)
        return 
    
    
    #@profile  
    def evaluate(self,**kwargs):
        problem = kwargs.get('problem')
        problem_name = problem['name']
        
        problem = kwargs['problem']
        problem_name = problem['name']
        self.orchestrator = self._get_orchestrator(problem_name, problem)

        model_params = kwargs.get('model_params')
        assert model_params is not None, "Model parameters are required"
        results = self.orchestrator.evaluate(model_params)
        #print(f"Evaluation results: {results}")
        
        return results
    
    
    #@profile 
    def evaluate_constraints(self,**kwargs):
        #print(f"\n[CLIENT {self.client_name}] Evaluating constraints list")
        problem = kwargs['problem']
        problem_name = problem['name']
        self.orchestrator = self._get_orchestrator(problem_name, problem)

        model_params = kwargs.get('model_params')
        log_results = kwargs.get('log_results',True)
        performance_constraint = kwargs.get('performance_constraint')
        original_threshold_list = kwargs.get('original_threshold_list')
        problem = kwargs.get('problem')
        problem_name = problem['name']

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
        if problem_name == 'global_problem':
            #print(f'[CLIENT {self.client_name}] Evaluation results: {final_results}')
            if log_results:
                self._eval_and_log(
                    performance_constraint=performance_constraint,
                    original_threshold_list=original_threshold_list,
                    model_params_list=[model_params],
                    eval_results=[final_results],
                )
        return final_results
    
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
    
    @profile 
    def evaluate_constraints_list(self,**kwargs):
        #print(f"\n[CLIENT {self.client_name}] Evaluating constraints list")
        problem = kwargs['problem']
        problem_name = problem['name']
        self.orchestrator = self._get_orchestrator(problem_name, problem)

        log_results = kwargs.get('log_results',True)
        problem = kwargs.get('problem')
        problem_name = problem['name']
        model_params_list = kwargs.get('model_params_list')
        first_performance_constraint = kwargs.get('first_performance_constraint',False)
        
        performance_constraint = kwargs.get('performance_constraint')
        original_threshold_list = kwargs.get('original_threshold_list')
        
        
        results = []
        
        for model_params in model_params_list:
            results_dict = self.orchestrator.evaluate_constraints2(model_params)
            #print(f'[CLIENT {self.client_name}] Evaluation results: {results_dict}')
            
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
        if problem_name == 'global_problem':
            if log_results:
                self._eval_and_log(
                        performance_constraint=performance_constraint,
                        original_threshold_list=original_threshold_list,
                        model_params_list=model_params_list,
                        eval_results=results,
                    )           



        #r = [r['metrics']['val_f1'] for r in results]
        #print(f'[CLIENT {self.client_name}] Evaluation results: {[r for r in results]}')
        #r = [r['metrics']['val_constraints_score'] for r in results]
        #print(f'[CLIENT {self.client_name}] Evaluation Score results: {r}')
        #print(f'[CLIENT {self.client_name}] Evaluation results: {results}')
        return results
    
    def personalize(self,**kwargs):
        #problem = copy.deepcopy(kwargs.get('problem'))
        #best_local_model_params,_,_= self._get_final_results(**kwargs)
        #problem['aggregation_teachers_list'] += [best_local_model_params]
        #kwargs['problem'] = copy.deepcopy(problem)
        log_results = kwargs.get('log_results',True)
        best_local_model_params,_,_= self._get_final_results(**kwargs)
        kwargs['num_local_epochs'] = 10
        kwargs['num_global_epochs'] = 1
        kwargs['model_params'] = copy.deepcopy(best_local_model_params)
        self.state = None
        personalized_results_dict = self.fit(**kwargs)
        personalized_model_params = personalized_results_dict['params']
        
        results_dict = self.orchestrator.evaluate_constraints2(personalized_model_params)
        #print(f'[CLIENT {self.client_name}] Evaluation personalized results: {results_dict}')
        
        metrics = results_dict['metrics']
        for checkpoint in self.client_checkpoints:
            if isinstance(checkpoint,ModelCheckpoint):
                model_checkpoint = checkpoint(save_fn=partial(self.save,metrics), metrics=metrics)
                metrics['model_checkpoint'] = 1 if model_checkpoint else 0
        if log_results:
            self.logger.log(metrics)
       
    
    def shutdown(self,**kwargs):
        log_results = kwargs.get('log_results',True)
        if log_results:
            self._log_final_results(**kwargs)
        self.logger.close()

    def fine_tune(self, **kwargs):
        return super().fine_tune(**kwargs)