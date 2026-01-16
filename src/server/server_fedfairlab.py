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
from surrogates import SurrogateFactory
from .utils import compute_group_cardinality,compute_global_score,collect_local_results,select_from_scores
from tqdm import tqdm
import random
import math
class EarlyStoppingException(Exception):
    pass

def compute_group_cardinality(group_name,sensitive_attributes):
        for name,group_dict in sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 

def average_dictionary_list(dictionary_list):
    result = {k:0 for k in dictionary_list[0].keys()}
    for d in dictionary_list:
        for k,v in d.items():
            result[k] += v
    for k in result.keys():
        result[k] /= len(dictionary_list)
    return result


@register_server("server_fedfairlab")
class ServerFedFairLab(BaseServer):
    
    def __init__(self,**kwargs): 

        self.clients_init_fn_list = kwargs.get('clients_init_fn_list')
        self.model = kwargs.get('model')
        self.loss = kwargs.get('loss')
        self.metrics = kwargs.get('metrics')
        self.log_model = kwargs.get('log_model', False)
        self.project = kwargs.get('project_name', 'fedfairlab')
        self.id = kwargs.get('server_name', 'server')
        self.checkpoint_dir = kwargs.get('checkpoint_dir','checkpoints')
        self.checkpoint_name = kwargs.get('checkpoint_name','global_model.h5')
        self.patience = kwargs.get('server_patience', 5)
        self.verbose = kwargs.get('verbose', False)
        self.num_federated_iterations = kwargs.get('num_federated_iterations', 1)
        self.num_classes = kwargs.get('num_classes', 2)
        
        self.original_metrics_list = kwargs.get('metrics_list')
        self.original_groups_list = kwargs.get('groups_list')
        self.original_threshold_list = kwargs.get('threshold_list')
        self.sensitive_attributes = kwargs.get('sensitive_attributes')
        self.performance_constraint = kwargs.get('performance_constraint')
        self.history_global=[]
        self.current_global_model_in_history = False
        self.fraction = kwargs.get('fraction', 0.5)
        
        self.callbacks = [
            EarlyStopping(patience=self.patience,
                          monitor='val_global_score',
                          mode='max'
                          ),
            ModelCheckpoint(save_dir=self.checkpoint_dir,
                            save_name = self.checkpoint_name,
                            monitor='val_global_score',
                            mode='max')
                          ]
        
        
        self.logger = WandbLogger(
            project=self.project,
            config= None,
            id=self.id,
            checkpoint_dir= self.checkpoint_dir,
            checkpoint_path = self.checkpoint_name,
            data_module=self.data if self.log_model else None
        )
        

        self.aggregator = AggregatorFactory().create('FedAvgAggregator')
        self.problem = self._init_constrained_problem(use_adaptive_aggregation=True,**kwargs)
        self.global_problem = self._init_global_constrained_problem(**kwargs)
        self.aggregation_problem = self._init_aggregation_problem(**kwargs)
        
        self.aggregator = AggregatorFactory().create('FedAvgAggregator')
        self.teachers = []
        
        self.global_model = None
        self.global_score = None
        self.global_model_idx = None
        self.history_per_client = {k:{} for k in range(len(self.clients_init_fn_list))}
        for i in self.history_per_client.keys():
            default_dict = {j:[] for j in range(len(self.clients_init_fn_list))}
            self.history_per_client[i] = default_dict
        # history_per_client[i][j] = list of scores of models evaluated on client i, coming from client j 
        #print('History per client:',self.history_per_client)
        print('Server initialized')

    
    def greedy_aggregator(self,**kwargs):
        params_list = kwargs.get('params')
        c1 = 100
        tau=0.5
        scores = []
        for param in params_list:
            score = self.evaluate(model_params=param['params'])
            #print('Score:',score.keys())
            scores.append(score['metrics']['val_global_score'])
        scores = torch.tensor(scores)
        argmax = torch.argmax(scores).item()
        inv_scores = 1.0 / (scores + 1e-6) 
        probabilities = torch.nn.functional.softmax(inv_scores / tau, dim=0)
        selected= torch.multinomial(probabilities, num_samples=1).item()
        return params_list[selected]['params'],params_list[argmax]['params'],scores[argmax].item()

    def greedy_aggregator_list(self,**kwargs):
        params_list = kwargs.get('params')
        tau=kwargs.get('tau',0.5)
        score = self.evaluate_list(model_params_list=[p['params'] for p in params_list])
        scores=[s['metrics']['val_global_score'] for s in score]
        scores = torch.tensor(scores)
        selected,argmin = select_from_scores(scores,tau=tau)
        return params_list[selected]['params'],params_list[argmin]['params'],scores[argmin].item(),argmin

    def aggregation_phase(self, **kwargs):
        aggregation_epochs = 1
        num_local_epochs = 10
        params = kwargs.get('params')
        model_params_list = [p['params'] for p in params]
        selected_clients = kwargs.get('selected_clients', self.clients)
        assert len(model_params_list) > 0, "Model parameters are required"

        # === Step 1: valutazione di tutti i modelli in un solo broadcast
        scores, model_eval_list = self.evaluate_list(
            model_params_list=model_params_list,
            return_metrics=True,
            log_results=True
        )
        print(f"[Server] Scores: {scores}")
        best_idx = torch.tensor(scores).argmax().item()
        final_model_params = model_params_list[best_idx]
        final_model_eval = model_eval_list[best_idx]
        final_model_score = scores[best_idx]

        # === Step 2: Setup problema di aggregazione
        current_agg_problem = copy.deepcopy(self.aggregation_problem)
        current_agg_problem['aggregation_teachers_list'] = model_params_list
        for key in ['objective_function', 'original_objective_function', 'batch_objective_function']:
            current_agg_problem[key].set_weights([s*100 for s in scores])
        scores = torch.tensor(scores,dtype=torch.double)*100
        #aggregation_weights = torch.nn.functional.softmax(scores,dim=0)
        current_agg_problem['aggregation_weights'] = scores.tolist()
        
        candidate_model_params = copy.deepcopy(self.model.state_dict())
        best_aggregated_score = -float('inf')
        
        # === Step 3: Aggregazione dei modelli per ensemble
        for _ in range(aggregation_epochs):
            handlers = [
                client.fit.remote(
                    model_params=candidate_model_params,
                    problem=current_agg_problem,
                    num_local_epochs=num_local_epochs,
                    num_global_epochs=1
                )
                for client in selected_clients
            ]
            results = ray.get(handlers)
            new_model_list = [res['params'] for res in results]

            # Rivalutazione dei modelli aggregati
            agg_scores = self.evaluate_list(
                model_params_list=new_model_list,
                return_metrics=False,
                log_results=True
            )

            agg_scores = torch.tensor(agg_scores)
            best_agg_idx = torch.argmax(agg_scores).item()

            if agg_scores[best_agg_idx] > best_aggregated_score:
                best_aggregated_score = agg_scores[best_agg_idx].item()
                candidate_model_params = copy.deepcopy(new_model_list[best_agg_idx])

        # === Step 4: Confronto finale
        final_eval = self._broadcast_fn(
            'evaluate_constraints',
            model_params=candidate_model_params,
            problem=self.global_problem,
            first_performance_constraint=self.performance_constraint < 1.0,
            performance_constraint=self.performance_constraint,
            original_threshold_list=self.original_threshold_list,
            log_results=True
        )
        
        print(f'Final evaluation: {final_eval}')
        final_score = compute_global_score(
            performance_constraint=self.performance_constraint,
            original_threshold_list=self.original_threshold_list,
            eval_results=final_eval
        )
        final_score_value = final_score['metrics']['val_global_score']

        print(f"[FedFairLab] Best original score:   {final_model_score:.4f}")
        print(f"[FedFairLab] Ensemble model score:  {final_score_value:.4f}")

        if final_score_value > final_model_score:
            final_model_params = candidate_model_params
            final_model_eval = final_score
            final_model_score = final_score_value

        print(f"[FedFairLab] Final selected score:  {final_model_score:.4f}")
        return final_model_params, final_model_eval


    def aggregation_phase_old(self,**kwargs):
        aggregation_epochs = 3 #kwargs.get('aggregation_epochs',5)
        num_aggregation_local_epochs = 3
        params = kwargs.get('params')
        model_params_list=[p['params'] for p in params]
        assert len(model_params_list) > 0, "Model parameters are required"
        #print('Length of model params list:',len(model_params_list))
        results = self._broadcast_fn('evaluate_constraints_list',
                                     model_params_list=copy.deepcopy(model_params_list),
                            problem=self.global_problem,
                            first_performance_constraint=self.performance_constraint<1.0,
                            performance_constraint=self.performance_constraint,
                            original_threshold_list=self.original_threshold_list,
                            log_results=True)
        #for r in results:
        #    print('Results of eval: ',r)
        b = [list(group) for group in zip(*results)]
        #for br in b:
        #    print('New results: ',br)
        global_scores = []
        original_scores = []
        model_eval_list = []
        for br in b:
            scores = compute_global_score(
                performance_constraint=self.performance_constraint,
                original_threshold_list=self.original_threshold_list,
                eval_results=br)
            #print('Scores:',scores)
            original_scores.append(scores['metrics']['val_global_score'])
            global_scores.append(scores['metrics']['val_global_score']*10)
            model_eval_list.append(scores)
        
        #global_scores += [h['score'] for h in self.history_global]
        #model_params_list += [h['params'] for h in self.history_global]

        #print('Setting scores for aggregation problem: ',global_scores)
        current_aggregation_problem = copy.deepcopy(self.aggregation_problem)
        current_aggregation_problem['aggregation_teachers_list'] = model_params_list 
        current_aggregation_problem['objective_function'].set_weights(global_scores)
        current_aggregation_problem['original_objective_function'].set_weights(global_scores)
        current_aggregation_problem['batch_objective_function'].set_weights(global_scores)
        current_aggregation_problem['aggregation_weights'] = global_scores
        
        global_scores = torch.tensor(global_scores)
        
        print('Global scores: ',global_scores)
        
        #print()
        print('Aggregation....')
        #print()

        sorted_global_scores = torch.argsort(torch.tensor(original_scores),descending=True)
        selected = sorted_global_scores[0].item()
        
        final_aggregated_model_params = copy.deepcopy(model_params_list[selected])
        final_aggregated_model_score = original_scores[selected] 
        final_aggregated_model_eval = model_eval_list[selected]
        candidate_model_params = copy.deepcopy(self.model.state_dict())
        
        
        aggregated_model_params = copy.deepcopy(self.model.state_dict())
        aggregated_model_score = -np.infty 
        
        for _ in range(aggregation_epochs):
            aggregation_results = []
            handlers = []
            selected_clients=self._select_clients(fraction=self.fraction)
            for _,client in enumerate(selected_clients):
                handlers.append(getattr(client,'fit').remote(
                    model_params=copy.deepcopy(aggregated_model_params),
                            problem=current_aggregation_problem,
                            num_local_epochs=num_aggregation_local_epochs,
                            num_global_epochs=1
                            
                            ))
            
            for handler in handlers:
                aggregation_results.append(ray.get(handler))

            aggregated_models_list = [res['params'] for res in aggregation_results]
            aggregated_model_params = self.aggregator(model=aggregated_model_params,
                                                      params=aggregation_results)
            eval_results = self._broadcast_fn('evaluate_constraints_list',
                                        model_params_list=[copy.deepcopy(aggregated_model_params)],
                                problem=self.global_problem,
                                first_performance_constraint=self.performance_constraint<1.0,
                                performance_constraint=self.performance_constraint,
                                original_threshold_list=self.original_threshold_list,
                                 log_results=True)
            
            b = [list(group) for group in zip(*eval_results)]
            global_aggregation_scores = []
            for br in b:
                scores = compute_global_score(
                    performance_constraint=self.performance_constraint,
                    original_threshold_list=self.original_threshold_list,
                    eval_results=br)
                #print('Scores:',scores)
                global_aggregation_scores.append(scores['metrics']['val_global_score'])
                #global_scores.append((1-scores['metrics']['val_global_score'])*100)
                #model_eval_list.append(scores)
            #global_aggregation_scores = []

            """
            for result in eval_results:
                scores = compute_global_score(
                    performance_constraint=None,
                    original_threshold_list=[],
                    eval_results=result)
                global_aggregation_scores.append(scores)
            """
            #print('Global aggregation scores:',global_aggregation_scores[0])
            global_aggregation_scores = torch.tensor(global_aggregation_scores)
            sorted_global_aggregation_scores = torch.argsort(global_aggregation_scores,descending=True)
            selected = sorted_global_aggregation_scores[0].item()
            selected_score = global_aggregation_scores[selected].item()
            #aggregated_model_params = aggregated_models_list[selected]
            
            #print('Scores of aggregated models:',global_aggregation_scores)
            #print('Selected model:',selected)
            if selected_score > aggregated_model_score:
                aggregated_model_score = selected_score
                candidate_model_params = copy.deepcopy(aggregated_models_list[selected])
                 
        global_eval = self.evaluate(model_params=candidate_model_params)
        print('Ensemble model score: ',(global_eval['metrics']['val_global_score']*100))
        if global_eval['metrics']['val_global_score'] > final_aggregated_model_score:
            final_aggregated_model_params = copy.deepcopy(candidate_model_params)
            final_aggregated_model_score = global_eval['metrics']['val_global_score']
            final_aggregated_model_eval = global_eval
        print('Final global model score: ',(final_aggregated_model_score*100))
        print('End of the aggregation')
        #final_aggregated_model_params = copy.deepcopy(final_aggregated_model_params)
        return final_aggregated_model_params,final_aggregated_model_eval
    
    def evaluate_list(self, model_params_list, *, return_metrics=False, log_results=False):
        """
        Valuta una lista di modelli su tutti i client.

        Args:
            model_params_list (List[dict]): Lista di state_dict dei modelli da valutare.
            return_metrics (bool): Se True, restituisce anche i dizionari di metriche per ciascun modello.
            log_results (bool): Se True, abilita il logging lato client.

        Returns:
            List[float] se return_metrics=False
            (List[float], List[Dict]) se return_metrics=True
        """
        assert len(model_params_list) > 0, "model_params_list non può essere vuota"

        # Broadcast a tutti i client: ogni client restituisce una valutazione per ciascun modello
        eval_results = self._broadcast_fn(
            'evaluate_constraints_list',
            model_params_list=model_params_list,
            problem=self.global_problem,
            first_performance_constraint=self.performance_constraint < 1.0,
            performance_constraint=self.performance_constraint,
            original_threshold_list=self.original_threshold_list,
            log_results=log_results
        )

        # Aggrega per modello
        zipped_results = list(zip(*eval_results))  # shape: num_models × num_clients
        global_scores = []
        model_eval_list = []

        for clientwise_eval in zipped_results:
            score_dict = compute_global_score(
                performance_constraint=self.performance_constraint,
                original_threshold_list=self.original_threshold_list,
                eval_results=clientwise_eval
            )
            global_scores.append(score_dict['metrics']['val_global_score'])
            model_eval_list.append(score_dict)

        if return_metrics:
            return global_scores, model_eval_list
        return global_scores

    
    def evaluate(self,**kwargs):
        model_params = kwargs.get('model_params')
        client_id = kwargs.get('client_id')     
       
        assert model_params is not None, "Model parameters are required"
        if client_id is None:
            results = self._broadcast_fn('evaluate_constraints',
                            model_params=copy.deepcopy(model_params),
                            problem=self.global_problem,
                            first_performance_constraint=self.performance_constraint<1.0,
                            performance_constraint=self.performance_constraint,
                            original_threshold_list=self.original_threshold_list)
        
        else: 
            handler = self.clients[client_id].evaluate_constraints.remote(model_params=copy.deepcopy(model_params),
                            problem=self.global_problem,
                            first_performance_constraint=self.performance_constraint<1.0,
                            performance_constraint=self.performance_constraint,
                            original_threshold_list=self.original_threshold_list)
        
            results = [ray.get(handler)]
        #print('Length of global results:',len(results))  
        global_scores = compute_global_score(
            performance_constraint=self.performance_constraint,
            original_threshold_list=self.original_threshold_list,
            eval_results=results)
        return global_scores

    def _create_clients(self,clients_init_fn_list):
        client_list = [client_init_fn() 
                for client_init_fn in clients_init_fn_list]
        #print('Clients:',client_list)
        return client_list
    
    def _init_constrained_problem(self,**kwargs):
        use_adaptive_aggregation = kwargs.get('use_adaptive_aggregation',False)
        if use_adaptive_aggregation:
            objective_function = SurrogateFactory.create(name='adaptive_aggregation_f1_10', surrogate_name='adaptive_aggregation', weight=1, average='weighted')
            #objective_function = SurrogateFactory.create(name='performance', surrogate_name='adaptive_aggregation', weight=1, average='weighted')
            original_objective_function = SurrogateFactory.create(name='multiclass_f1', mode='max',surrogate_name='multiclass_f1', weight=1, average='weighted')
            #batch_objective_function = SurrogateFactory.create(name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted')
            batch_objective_function = SurrogateFactory.create(name='batch_adaptive_aggregation_f1_10', surrogate_name='multiclass_f1', weight=1, average='weighted')
        else:
            objective_function = SurrogateFactory.create(name='performance', surrogate_name='cross_entropy', weight=1, average='weighted')
            original_objective_function = SurrogateFactory.create(name='multiclass_f1', mode='max',surrogate_name='multiclass_f1', weight=1, average='weighted')
            batch_objective_function = SurrogateFactory.create(name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted')
        
        
        inequality_constraints = []
        macro_constraints = []
        shared_macro_constraints = []
        idx_constraint = 0
        all_group_ids = {}
        if self.performance_constraint < 1.0:
            inequality_constraints = [SurrogateFactory.create(name='multiclass_f1', 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=self.performance_constraint,
                                    use_max=False)] 
            idx_constraint = 1
            macro_constraints = [[0]]
            shared_macro_constraints = [0]

        for metric,group,threshold in zip(self.original_metrics_list,
                                          self.original_groups_list,
                                          self.original_threshold_list):
            
            group_cardinality = compute_group_cardinality(group,sensitive_attributes=self.sensitive_attributes)
            macro_constraint = []
            current_group_ids = {group: list(range(group_cardinality))}
            all_group_ids.update(current_group_ids)
            for i in range(group_cardinality):
                for j in range(i+1,group_cardinality):
                    if self.num_classes == 2:
                        constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                        surrogate_name=f'diff_{metric}_{group}',
                                                        surrogate_weight=1,
                                                        average='weighted',
                                                        group_name=group,
                                                        unique_group_ids={group: list(range(group_cardinality))},
                                                        lower_bound=threshold,
                                                        use_max=False,
                                                        target_groups=torch.tensor([i, j]))
                    
                        inequality_constraints.append(constraint)
                        macro_constraint.append(idx_constraint)
                        idx_constraint += 1
                    else:
                        for c in range(self.num_classes):
                            constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                            surrogate_name=f'diff_{metric}_{group}',
                                                            surrogate_weight=1,
                                                            average='weighted',
                                                            group_name=group,
                                                            unique_group_ids={group: list(range(group_cardinality))},
                                                            lower_bound=threshold,
                                                            use_max=False,
                                                            target_groups=torch.tensor([i, j]),
                                                            target_class = c
                                                            )
                            inequality_constraints.append(constraint)
                            macro_constraint.append(idx_constraint)
                            idx_constraint += 1
            macro_constraints.append(macro_constraint)       

        inequality_constraints = inequality_constraints
        macro_constraints = macro_constraints
        shared_macro_constraints = shared_macro_constraints
        all_group_ids = all_group_ids


        problem = {
            'name': 'local_problem',
            'original_objective_function': original_objective_function,
            'objective_function': objective_function,
            'batch_objective_function': batch_objective_function,
            'inequality_constraints': inequality_constraints,
            'macro_constraints_list': macro_constraints,
            'shared_macro_constraints': shared_macro_constraints,
            'all_group_ids': all_group_ids,
            'aggregation_teachers_list': [],
            'num_classes': self.num_classes
        }
        print('All group ids: ', all_group_ids)
        print('Macro constraints: ', macro_constraints)
        print('Num of macro constraints: ', len(macro_constraints)) 
        #print('Inequality constraints: ', inequality_constraints)
        print('Num of inequality constraints: ', len(inequality_constraints))
        return problem           

    def _init_aggregation_problem(self,**kwargs):
       
        objective_function = SurrogateFactory.create(name='adaptive_aggregation_f1_8', surrogate_name='adaptive_aggregation', weight=1, average='weighted')
        original_objective_function = SurrogateFactory.create(name='adaptive_aggregation_f1_8_max', surrogate_name='multiclass_f1', weight=1, average='weighted')
        batch_objective_function = SurrogateFactory.create(name='adaptive_aggregation_f1_8', surrogate_name='cross_entropy', weight=1, average='weighted')
        all_group_ids = {}
        for metric,group,threshold in zip(self.original_metrics_list,
                                          self.original_groups_list,
                                          self.original_threshold_list):
            
            group_cardinality = compute_group_cardinality(group,sensitive_attributes=self.sensitive_attributes)
            current_group_ids = {group: list(range(group_cardinality))}
            all_group_ids.update(current_group_ids)
           
        problem = {
            'name': 'aggregation_problem',
            'original_objective_function': original_objective_function,
            'objective_function': objective_function,
            'batch_objective_function': batch_objective_function,
            'inequality_constraints': [],
            'macro_constraints_list': [],
            'shared_macro_constraints': [],
            'all_group_ids': all_group_ids,
            'aggregation_teachers_list': [],
            'num_classes': self.num_classes
            
        }
        print('Aggregation: All group ids: ', all_group_ids)
        return problem
    
    def _init_global_constrained_problem(self,**kwargs):
        objective_function = SurrogateFactory.create(name='performance', surrogate_name='cross_entropy', weight=1, average='weighted')
        batch_objective_function = SurrogateFactory.create(name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted')
        original_objective_function = SurrogateFactory.create(name='multiclass_f1', mode='max',surrogate_name='multiclass_f1', weight=1, average='weighted')
        
        inequality_constraints = []
        macro_constraints = []
        shared_macro_constraints = []
        idx_constraint = 0
        all_group_ids = {}
        if self.performance_constraint < 1.0:
            inequality_constraints = [SurrogateFactory.create(name='multiclass_f1', 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=1.0,
                                    use_max=False)] 
            idx_constraint = 1
            macro_constraints = [[0]]
            shared_macro_constraints = [0]

        for metric,group,_ in zip(self.original_metrics_list,
                                          self.original_groups_list,
                                          self.original_threshold_list):
            group_cardinality = compute_group_cardinality(group,sensitive_attributes=self.sensitive_attributes)
            macro_constraint = []
            current_group_ids = {group: list(range(group_cardinality))}
            all_group_ids.update(current_group_ids)
            for i in range(group_cardinality):
                for j in range(i+1,group_cardinality):
                    if self.num_classes == 2:
                        constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                        surrogate_name=f'diff_{metric}_{group}',
                                                        surrogate_weight=1,
                                                        average='weighted',
                                                        group_name=group,
                                                        unique_group_ids={group: list(range(group_cardinality))},
                                                        lower_bound=0.0,
                                                        use_max=True,
                                                        target_groups=torch.tensor([i, j])
                                                        )
                        inequality_constraints.append(constraint)
                        macro_constraint.append(idx_constraint)
                        idx_constraint += 1
                    else:
                        for c in range(self.num_classes):
                            constraint = SurrogateFactory.create(name=f'diff_{metric}',
                                                            surrogate_name=f'diff_{metric}_{group}',
                                                            surrogate_weight=1,
                                                            average='weighted',
                                                            group_name=group,
                                                            unique_group_ids={group: list(range(group_cardinality))},
                                                            lower_bound=0.0,
                                                            use_max=True,
                                                            target_groups=torch.tensor([i, j]),
                                                            target_class = c
                                                            )
                            inequality_constraints.append(constraint)
                            macro_constraint.append(idx_constraint)
                            idx_constraint += 1
            macro_constraints.append(macro_constraint)       

        inequality_constraints = inequality_constraints
        macro_constraints = macro_constraints
        shared_macro_constraints = shared_macro_constraints
        all_group_ids = all_group_ids


        problem = {
            'name': 'global_problem',
            'original_objective_function': original_objective_function,
            'objective_function': objective_function,
            'batch_objective_function': batch_objective_function,
            'inequality_constraints': inequality_constraints,
            'macro_constraints_list': macro_constraints,
            'shared_macro_constraints': shared_macro_constraints,
            'all_group_ids': all_group_ids,
            'aggregation_teachers_list': [],
            'num_classes': self.num_classes
        }
        print('All group ids: ', all_group_ids)
        print('Macro constraints: ', macro_constraints)
        print('Num of macro constraints: ', len(macro_constraints)) 
        #print('Inequality constraints: ', inequality_constraints)
        print('Num of inequality constraints: ', len(inequality_constraints))
        return problem 
    
    def _select_clients(self,fraction=1.0):
        assert 0 < fraction <= 1, "Fraction must be between 0 and 1"
        num_clients = max(1,math.ceil(len(self.clients)* fraction))
        selected_indices = random.sample(range(len(self.clients)), num_clients)
        selected_clients = [self.clients[i] for i in selected_indices]
        print(f'[Server] Selected clients {[s+1 for s in selected_indices]} for this round')
        return selected_clients
    
    def _broadcast_fn(self,fn_name,**kwargs):
        assert isinstance(fn_name,str), "fn_name must be a string"
        selected_clients = kwargs.get('selected_clients', self.clients)
        handlers = []
        results = []
        
        for client in selected_clients:
            assert hasattr(client,fn_name), f"Client does not have {fn_name} method"
            handlers.append(getattr(client,fn_name).remote(**kwargs))
        for handler in handlers:
            results.append(ray.get(handler))
        return results
    
    def _evaluate_best_model(self):
        global_model = torch.load(self.checkpoint_path)
        self.model.load_state_dict(global_model)
        global_scores = self.evaluate(model_params=self.model.state_dict())
        final_scores ={}
        for key,v in global_scores['metrics'].items():
            final_scores[f'final_{key}'] = v
        self.logger.log(final_scores)

    def _evaluate_global_model(self,best_model=False):
        if best_model:
            scores = self._broadcast_fn('evaluate_best_model',
                            global_model=self.model)
        else: 
            scores = self._broadcast_fn('evaluate',
                            global_model=self.model)    
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
    
    
    def _update_history(self,results):
        # in posizione i di results ci sono le valutazioni del modello del client i sui vari clients
        # results[i][j] è il risultato della valutazione del modello del client i sul client j
        for i, res in enumerate(results):
            for j, score in enumerate(res):
                self.history_per_client[i][j].append(score)
    
    def select_new_teachers(self,**kwargs):
        #global_score = kwargs.get('global_score')
        global_model_idx = kwargs.get('global_model_idx')
        if global_model_idx is None:
            print('No global model index provided, returning empty list of teachers')
            return []
        
        teachers = {k:[] for k in self.history_per_client.keys()}
        for i in self.history_per_client.keys():
            teachers_per_client = []
            scores_per_client = []
            indexes = []
            global_model_score = self.last_round_scores[i][global_model_idx]

            for j in self.history_per_client.keys(): 
                history = self.history_per_client[i][j]
                if len(history) > 0:
                    if j==global_model_idx:
                        scores = [h['score'] for idx,h in enumerate(history) if h['score'] < global_model_score and idx < len(history)-1 and idx != i] 
                        indexes = [idx for idx,h in enumerate(history) if h['score'] < global_model_score and idx < len(history)-1 and idx != i] 
                    else:
                        scores = [h['score'] for idx,h in enumerate(history) if h['score'] < global_model_score  and idx != i]
                        indexes = [idx for idx,h in enumerate(history) if h['score'] < global_model_score  and idx != i] 
                    if len(scores) > 0:
                        selected,_ = select_from_scores(torch.tensor(scores))
                        teachers_per_client.append(copy.deepcopy(history[indexes[selected]]['model_params']))
                        scores_per_client.append(history[indexes[selected]]['score']*100)
                    #real_scores_per_client.append(history[selected]['score']*10)
            
            if len(scores_per_client) >0:
                sorted_scores_idx = np.argsort(scores_per_client)
                selected_teachers = [teachers_per_client[k] for k in sorted_scores_idx] 
                teachers[i] = selected_teachers[:1]

            """
            if len(teachers_per_client) > 0:
                #print(f'Real scores per client (target = {i}):',real_scores_per_client)
                #print(f'Scores per client (target = {i}):',scores_per_client)
                scores_per_client = torch.tensor(scores_per_client)
                #probs = torch.softmax(-scores_per_client / 0.5, dim=0)
                probs =(scores_per_client - scores_per_client.min()) / (scores_per_client.max() - scores_per_client.min())
                alpha = 2.0  # controlla la "bruschezza"
                probs = torch.exp(-probs * alpha)
                #print(f'Probabilities (target = {i}):',probs)
                selected_teachers_mask = torch.bernoulli(probs).bool().view(-1)
                #print('Selected teachers mask:',selected_teachers_mask)
                selected_teachers = [teachers_per_client[k] for k in range(len(teachers_per_client)) if selected_teachers_mask[k]]
                #selected_teachers = [teachers_per_client[k] for k in range(len(teachers_per_client)) ]
                teachers[i] = selected_teachers
            """
        
        return teachers  
    
    def save(self,metrics,path):
        result_to_save = {
            'model_params': self.model.state_dict(),
            'metrics': metrics
        }
        torch.save(result_to_save, path)
    
    def step(self,**kwargs):
        handlers = []
        results = []
        #teachers = self.select_new_teachers(global_score=self.global_score,
        #                                    global_model_idx=self.global_model_idx)

        selected_clients = self._select_clients(fraction=self.fraction)
        for i,client in enumerate(selected_clients):
            
            #if len(self.problem['aggregation_teachers_list']) >0:
                #problem['aggregation_teachers_list'] = [copy.deepcopy(self.model.state_dict())]+teachers[i]  
                #print(f'[SERVER] Client {i} teachers:',len(problem['aggregation_teachers_list']))
                #assert len(problem['aggregation_teachers_list']) == len(teachers[i]) + 1, f"Teachers list length mismatch: {len(problem['aggregation_teachers_list'])} != {len(teachers[i]) + 1}"
            #else:
            #    problem['aggregation_teachers_list'] = []
                #print('NO TEACHERS')
            #if self.global_model is not None:
            #    problem['aggregation_teachers_list'] = [copy.deepcopy(self.global_model.state_dict())]
            current_problem = copy.deepcopy(self.problem)
            """
            if i in [0,1,3]:
                for idx,constraint in enumerate(current_problem['inequality_constraints']):
                    if self.performance_constraint < 1.0:
                        if idx > 0:
                            constraint.set_lower_bound(0.2)
                            #print('Lower bound:',constraint.lower_bound)
            
            elif i in [2,4,5]:
                for idx,constraint in enumerate(current_problem['inequality_constraints']):
                    if self.performance_constraint < 1.0:
                        if idx > 0:
                            constraint.set_lower_bound(0.1)
                            #print('Lower bound:',constraint.lower_bound)
            """
            if not self.first_round:
                
                current_problem['aggregation_teachers_list'] = [copy.deepcopy(g['params']) for g in self.history_global] 
                #if len(current_problem['aggregation_teachers_list']) == 1 and not self.current_global_model_in_history:
                #    print('Current global model is not in history, adding it to the aggregation teachers list')
                #    current_problem['aggregation_teachers_list'].append(copy.deepcopy(self.history_global[0]['params']))
                
                handlers.append(getattr(client,'fit').remote(model_params=copy.deepcopy(self.history[i]),
                           problem=current_problem,selected_clients=selected_clients,))
            else:
                handlers.append(getattr(client,'fit').remote(model_params=copy.deepcopy(self.model.state_dict()),
                            problem=current_problem,selected_clients=selected_clients,))
        self.first_round = False     
        for handler in handlers:
            results.append(ray.get(handler))
        self.history = [r['params'] for r in results]
       
        #results = self._broadcast_fn('fit',
        #                   model_params=copy.deepcopy(self.model.state_dict()),
        #                   problem=self.problem)
        #self.model.load_state_dict(results[0]['params'])
        #global_model = copy.deepcopy(self.model)
        #new_params = self.aggregator(model=global_model,
        #                params=results)
        
        #aggregated_model_params = self.aggregator(model=self.model,
        #               params=results)
        #self.model.load_state_dict(aggregated_model_params)
        #global_eval = self.evaluate(model_params=self.model.state_dict())
        #print('Global Score:',(1-global_eval['metrics']['val_global_score'])*100)
        aggregated_model_params,global_eval = self.aggregation_phase(aggregation_epochs=1,
                                                           params=results,
                                                           selected_clients=selected_clients)
        
        
        #new_params,best_model_params,global_score,global_model_idx = self.greedy_aggregator_list(params=results)
        #self.global_score = global_score
        #self.global_model_idx = global_model_idx
        #self.problem['aggregation_teachers_list'] = [res['params'] for res in results]
        self.model.load_state_dict(aggregated_model_params)
        global_score = global_eval['metrics']['val_global_score']
        if len(self.history_global) > 0:
            if global_score > torch.tensor([h['score'] for h in self.history_global]).min().item():
                self.current_global_model_in_history = True
            else: 
                self.current_global_model_in_history = False
        self.history_global.append({'params':aggregated_model_params,
                                    'score':(global_eval['metrics']['val_global_score']),
                                    })
        
        self.history_global.sort(key=lambda x: x['score'],reverse=True)
        #print('History global:',[s['score'] for s in self.history_global])
        self.history_global = self.history_global[:1]
        #global_eval = self.evaluate(model_params=self.model.state_dict())
         
        try:
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    stop,counter = callback(metrics=global_eval['metrics'])
                    global_eval['metrics']['global_early_stopping'] = counter
                    if stop:
                        self.logger.log(global_eval['metrics'])  
                        raise EarlyStoppingException  
                elif isinstance(callback,ModelCheckpoint):
                    model_checkpoint = callback(save_fn=partial(self.save,
                                                              global_eval['metrics']),
                                                metrics = global_eval['metrics']
                                                )
                            
                    global_eval['metrics']['global_checkpoint'] = 1 if model_checkpoint else 0
                    self.global_model = copy.deepcopy(self.model)
            self.logger.log(global_eval['metrics'])
            #self.model.load_state_dict(new_params)
        
        except EarlyStoppingException:
            raise EarlyStoppingException 

    def personalization_phase(self,**kwargs):
        #print('Personalization phase')
        #print('Checkpoint path:',self.checkpoint_path)
        handlers = []
        for callback in self.callbacks:
          if isinstance(callback, ModelCheckpoint):
            best_results = callback.get_best_model()  
        global_model_params = best_results['model_params']
        current_problem = copy.deepcopy(self.problem)
        current_problem['aggregation_teachers_list'] = [h['params'] for h in self.history_global]
        self._broadcast_fn('personalize',
                           model_params=global_model_params,
                           problem=current_problem,
                           first_performance_constraint=self.performance_constraint<1.0,
                           performance_constraint=self.performance_constraint,
                           original_threshold_list=self.original_threshold_list)
        
        return          
    
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
    
    
    def execute(self,**kwargs):
        
        self.first_round = True
        global_eval = self.evaluate(model_params=self.model.state_dict())
        
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping):
                stop,counter = callback(metrics=global_eval['metrics'])
                global_eval['metrics']['global_early_stopping'] = counter
                if stop:
                    self.logger.log(global_eval)  
                    raise EarlyStoppingException  
            elif isinstance(callback,ModelCheckpoint):
                model_checkpoint = callback(save_fn=partial(self.save,
                                                              global_eval['metrics']),
                                                metrics = global_eval['metrics']
                                                )
                global_eval['metrics']['global_checkpoint'] = 1 if model_checkpoint else 0
        
        self.logger.log(global_eval['metrics'])
        
        try:
            pbar = tqdm(range(self.num_federated_iterations), desc="Global rounds", unit="round")
            for i in pbar:
                pbar.set_postfix_str(f"Global Round {i+1}")
                self.step(round=i)
        except EarlyStoppingException:
            pass
        print('End of the global rounds')
        print('Starting the personalization phase')
        self.personalization_phase()
        print('End of the personalization phase')

    def evaluate_model_from_ckpt(self,**kwargs):
        checkpoint_path = kwargs.get('checkpoint_path')
        client_id = kwargs.get('client_id')
        assert checkpoint_path is not None, "Checkpoint Path is required"
        

        model_params = torch.load(checkpoint_path)
       
        if client_id is None:
            try:
                self.model.load_state_dict(model_params['model_params'])
            except KeyError:
                self.model.load_state_dict(model_params)
        else:
            try:
                model_params = model_params['model_state_dict']
                if isinstance(model_params, dict):
                    self.model.load_state_dict(model_params)
                else:
                    self.model = copy.deepcopy(model_params)
            except KeyError:
                self.model.load_state_dict(model_params)

        print('Evaluating model from checkpoint:',checkpoint_path)
       
        global_scores = self.evaluate(model_params=self.model.state_dict(),
                                      client_id=client_id)
        final_scores = {}
        for key,v in global_scores['metrics'].items():
                final_scores[f'final_{key}'] = v
        #print('Final scores:',final_scores)
        return final_scores
    
    
    def fine_tune(self,**kwargs):
        handlers = []
        if os.path.exists(self.checkpoint_path):
            global_model = torch.load(self.checkpoint_path)
            self.model.load_state_dict(global_model)
        
        self._broadcast_fn('fine_tune',global_model=self.model)
        return
    
    def shutdown(self,**kwargs):
        log_results = kwargs.get('log_results',True)
        if log_results:
            self.log_final_results()
        
        self.logger.close()
        self._broadcast_fn('shutdown',
                           log_results=log_results)