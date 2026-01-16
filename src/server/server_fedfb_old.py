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

@register_server("server_fedfb")
class ServerFedFB(BaseServer):
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
       
        self.fair_metric = kwargs.get('fair_metric', 'demographic_parity')
        self.federated_rounds = kwargs.get('num_federated_iterations')
        self.monitor = kwargs.get('monitor')
        self.mode = kwargs.get('mode')
        print(f"[INFO] Initializing ServerFedFB with id: {self.id}, metrics: {self.metrics}, monitor: {self.monitor}, num_federated_rounds: {self.federated_rounds}")
        self.verbose = kwargs.get('verbose', False)
        self.alpha = kwargs.get('alpha',0.01)

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
                            model=self.model)    
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
    
    def get_number_of_groups(self):
        group_info_list = self._broadcast_fn('get_groups_ids')
        group_ids = set()
        for group_info in group_info_list:
            #print(f"[INFO] Group IDs from client: {group_info['group_ids']}")
            group_ids.update(group_info['group_ids'])
        
        self.group_ids = list(group_ids)
        #print(f"[INFO] Number of groups: {len(self.group_ids)}")
        #print(f"[INFO] Group IDs: {self.group_ids}")
        self.group_cardinality = {} 
        for y in [0,1]:
            for group_id in self.group_ids:
                self.group_cardinality[(y,group_id)] = 0
        for y,group_id in self.group_cardinality.keys():
            #print(f'Getting group cardinality for group {group_id} and target {y}') 
            num_elements_list = self._broadcast_fn('get_group_cardinality',
                                                y=y,group_id=group_id
                                                )
            self.group_cardinality[(y,group_id)] = sum([n['group_cardinality'] for n in num_elements_list])
        
        self.total_records = sum([v for k,v in self.group_cardinality.items()])
    
    
    def setup(self, **kwargs):
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        self.clients = self._create_clients(self.clients_init_fn_list)
        self._broadcast_fn('setup', global_model_ckpt_path=self.checkpoint_path)

        # Popola group_ids, group_cardinality, total_records ecc.
        self.get_number_of_groups()

        # Calcola n_z = n_{0,z} + n_{1,z} e controlla che il task sia binario
        self.group_cardinality_vector = {}
        labels = sorted({y for (y, z) in self.group_cardinality.keys()})
        assert len(labels) == 2, "FedFB baseline: expected binary label task (y in {0,1})."
        y0, y1 = labels

        for z in self.group_ids:
            n_z = self.group_cardinality.get((y0, z), 0) + self.group_cardinality.get((y1, z), 0)
            self.group_cardinality_vector[z] = n_z

        # λ_{0,z} = λ_{1,z} = p(z) con p(z) = n_z / N  (cap = 2*p(z))
        N = float(self.total_records) if self.total_records > 0 else 1.0
        self.lambdas = {}
        for z in self.group_ids:
            pz = self.group_cardinality_vector[z] / N
            self.lambdas[(y0, z)] = pz
            self.lambdas[(y1, z)] = pz

        
    
    def _update_lambdas(self, **kwargs):
        """
        Update dei pesi λ per FedFB:
        - DP: differenze dei tassi positivi tra gruppi
        - EOD: differenze di TPR e FPR tra gruppi (aggiorna λ_{1,z} con TPR e λ_{0,z} con FPR)
        """
        fair_metric = self.fair_metric

        results = self._broadcast_fn(
           'inference',
           global_model=self.model,
           group_cardinality_vector=self.group_cardinality_vector,
           group_cardinality=self.group_cardinality,
           fair_metric=fair_metric
        )

        if len(self.group_ids) == 0:
           return
        z0 = min(self.group_ids)  # gruppo di riferimento

        if fair_metric == 'demographic_parity':
           # 1) Aggrega contatori DP
            pos, tot = {}, {}
            for z in self.group_ids:
                pos[z], tot[z] = 0, 0
            for res in results:
                if res.get('metric') != 'demographic_parity':
                   continue
                dp = res.get('dp_counts', {})
                if not dp:
                   continue
                for z in self.group_ids:
                   pos[z] += int(dp['pos'].get(int(z), 0))
                   tot[z] += int(dp['tot'].get(int(z), 0))

           # 2) Tassi e violazioni
            dp_rate = {z: (pos[z] / max(1, tot[z])) for z in self.group_ids}
            viol = {z: dp_rate[z] - dp_rate[z0] for z in self.group_ids}

            # 3) μ somma-zero
            mu_vec = []
            for z in self.group_ids:
                if z == z0:
                    mu = -sum(viol[zz] for zz in self.group_ids if zz != z0)
                else:
                    mu = viol[z]
                mu_vec.append(mu)
            mu_norm = np.linalg.norm(mu_vec) + 1e-12

            # 4) Update λ_{0,z}, λ_{1,z} complementare
            for z, mu in zip(self.group_ids, mu_vec):
                self.lambdas[(0, z)] += (self.alpha / mu_norm) * float(mu)
                cap = 2.0 * (self.group_cardinality_vector[z] / self.total_records)
                self.lambdas[(0, z)] = float(np.clip(self.lambdas[(0, z)], 0.0, cap))
                self.lambdas[(1, z)] = float(cap - self.lambdas[(0, z)])

        elif fair_metric == 'equalized_odds':
            # 1) Aggrega TP, Pos, FP, Neg
            tp, pos, fp, neg = {}, {}, {}, {}
            for z in self.group_ids:
                tp[z] = pos[z] = fp[z] = neg[z] = 0
            for res in results:
                if res.get('metric') != 'equalized_odds':
                    continue
                eod = res.get('eod_counts', {})
                if not eod:
                    continue
                for z in self.group_ids:
                    zi = int(z)
                    tp[z]  += int(eod['tp'].get(zi, 0))
                    pos[z] += int(eod['pos'].get(zi, 0))
                    fp[z]  += int(eod['fp'].get(zi, 0))
                    neg[z] += int(eod['neg'].get(zi, 0))

            # 2) TPR/FPR e violazioni
            tpr = {z: (tp[z] / max(1, pos[z])) for z in self.group_ids}
            fpr = {z: (fp[z] / max(1, neg[z])) for z in self.group_ids}
            viol_tpr = {z: tpr[z] - tpr[z0] for z in self.group_ids}
            viol_fpr = {z: fpr[z] - fpr[z0] for z in self.group_ids}
            # 3) μ somma-zero per TPR e FPR
            mu_tpr, mu_fpr = [], []
            for z in self.group_ids:
                if z == z0:
                    mu_tpr_z = -sum(viol_tpr[zz] for zz in self.group_ids if zz != z0)
                    mu_fpr_z = -sum(viol_fpr[zz] for zz in self.group_ids if zz != z0)
                else:
                    mu_tpr_z = viol_tpr[z]
                    mu_fpr_z = viol_fpr[z]
                mu_tpr.append(mu_tpr_z)
                mu_fpr.append(mu_fpr_z)

            mu_tpr_norm = np.linalg.norm(mu_tpr) + 1e-12
            mu_fpr_norm = np.linalg.norm(mu_fpr) + 1e-12

            # 4) Update separati: λ_{1,z} con μ_tpr, λ_{0,z} con μ_fpr
            for (z, mt, mf) in zip(self.group_ids, mu_tpr, mu_fpr):
                self.lambdas[(1, z)] += (self.alpha / mu_tpr_norm) * float(mt)  # parte "positivi"
                self.lambdas[(0, z)] += (self.alpha / mu_fpr_norm) * float(mf)  # parte "negativi"

            # 5) Proiezione: mantieni λ in [0, cap] e rinormalizza coppia alla massa target
            for z in self.group_ids:
                cap = 2.0 * (self.group_cardinality_vector[z] / self.total_records)
                self.lambdas[(0, z)] = float(np.clip(self.lambdas[(0, z)], 0.0, cap))
                self.lambdas[(1, z)] = float(np.clip(self.lambdas[(1, z)], 0.0, cap))
                pair_sum = self.lambdas[(0, z)] + self.lambdas[(1, z)]
                target_sum = cap
                if pair_sum <= 1e-12:
                    self.lambdas[(0, z)] = target_sum / 2.0
                    self.lambdas[(1, z)] = target_sum / 2.0
                else:
                    scale = target_sum / pair_sum
                    self.lambdas[(0, z)] *= scale
                    self.lambdas[(1, z)] *= scale

        else:
            raise ValueError(f"fair_metric sconosciuto: {fair_metric}")

    def step(self,**kwargs):
        results = self._broadcast_fn('update',
                           global_model=self.model,
                           lambdas=self.lambdas,
                           group_cardinality_vector=self.group_cardinality_vector   
                           )
        global_model = copy.deepcopy(self.model)
        new_params = self.aggregator(model=global_model,
                        params=results)
        self.model.load_state_dict(new_params)
        round_ = kwargs.get('round')
        self._update_lambdas(round=round_)
        
        global_scores = self._evaluate_global_model()
        global_scores['global_round'] = round_ + 1
       
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
        self._broadcast_fn('fine_tune',global_model=self.model,
                           lambdas=self.lambdas,
                           group_cardinality_vector=self.group_cardinality_vector
                           )
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