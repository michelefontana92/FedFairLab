from .torch_nn_wrapper import TorchNNWrapper
import torch
import tqdm
from callbacks import EarlyStopping, ModelCheckpoint
import os
from entmax import entmax_bisect
from metrics import Performance,GroupFairnessMetric
import copy


class EarlyStoppingException(Exception):
    pass

class LocalLearner(TorchNNWrapper):
    """
    LocalLearner is a class that implements a local learner with ALM optimization.
    Methods:
        compute_groups_cardinality(): Computes the cardinality of groups.
        _init_inequality_lambdas(): Initializes the inequality lambdas.
        _init_alm_parameters(): Initializes the ALM parameters.
        update_lambdas_inequality(constraints): Updates the inequality lambdas.
        update_lambdas_equality(constraints): Updates the equality lambdas.
        update_alm_parameters_and_metrics(update_alm=True, **kwargs): Updates the ALM parameters and computes metrics.
        compute_constraints(**kwargs): Computes the constraints.
        compute_score(**kwargs): Computes the score.
        compute_loss_fn(**kwargs): Computes the loss function.
        _compute_metrics(metrics, prefix='val', **kwargs): Computes metrics.
        _training_step(batch, batch_idx): Performs a training step.
        _train_eval_step(**kwargs): Performs a training evaluation step.
        _validation_step(**kwargs): Performs a validation step.
        set_constraints(inequality_constraints_fn_list, equality_constraints_fn_list, macro_constraints_list, inequality_lambdas, equality_lambdas): Sets the constraints.
        evaluate(model_dict, **kwargs): Evaluates the model.
        compute_val_kwargs(model_dict, use_training=False): Computes validation kwargs.
        compute_violations(val_kwargs, **kwargs): Computes the violations.
        fit(**kwargs): Fits the model.
    """
    def __init__(self, *args, **kwargs):
      
        super(LocalLearner, self).__init__(*args, **kwargs)
        self.id = kwargs.get('id','LagrangianWrapper')
        self.compute_only_score =kwargs.get('compute_only_score',False)
        self.optimizer_fn: callable = kwargs.get('optimizer_fn')
        self.lagrangian_checkpoints = kwargs.get('lagrangian_checkpoints', [])
        #self.training_group_name: str = kwargs.get('training_group_name')
        
        self.teacher_model = kwargs.get('teacher_model')
        #self.distillation_loss_fn:callable = kwargs.get('distillation_loss_fn')
        self.batch_objective_function = kwargs.get('batch_objective_fn')
        self.original_objective_fn:callable = kwargs.get('original_objective_fn')
        self.objective_fn: callable = kwargs.get('objective_fn')
        self.inequality_constraints_fn_list: list = kwargs.get('inequality_constraints')
        self.equality_constraints_fn_list: list = kwargs.get('equality_constraints')
        
        self.mu_max = kwargs.get('mu_max', 1e3)
        self.nu_max = kwargs.get('nu_max', 100)
        self.lambda_equality_max = kwargs.get('lambda_equality_max', 100)
        self.lambda_inequality_max = kwargs.get('lambda_inequality_max', 100)

        self.rho = kwargs.get('rho', 2)
        self.mu_0 = kwargs.get('mu_0', 2)
        self.damping_factor = kwargs.get('damping_factor', 1.0)  # Valore di damping per rallentare l'aggiornamento

        self.gamma_objective = kwargs.get('gamma_objective', 0.8)
        self.gamma_constraint = kwargs.get('gamma_constraint', 10.0)

        self.inequality_lambdas_0_value = kwargs.get('inequality_lambdas_0_value', 0.1)
        self.equality_lambdas_0_value = kwargs.get('equality_lambdas_0_value', 0.)
        self.objective_multiplier_0_value = kwargs.get('objective_multiplier_0_value', 1)
        self.macro_constraints_list= kwargs.get('macro_constraints_list')
        # Assicurati che tutti i tensori siano su device
        self.inequality_lambdas_0 = torch.ones(len(self.inequality_constraints_fn_list), device=self.device) * self.inequality_lambdas_0_value
        self.equality_lambdas_0 = torch.ones(len(self.equality_constraints_fn_list), device=self.device) * self.equality_lambdas_0_value
        self.objective_multiplier_0 = torch.tensor(self.objective_multiplier_0_value, device=self.device)
        self.lambda0_max_value = kwargs.get('lambda0_max_value', 0.1)
        self.target_groups = set()

        self.teachers_kwargs = {'train':{},
                                'val':{}
                                }

        self.wasserstein_kwargs = {'train':{},
                                   'val':{}
                                  }
        
        self.verbose = kwargs.get('verbose', False)
        #self.compute_groups_cardinality()
        self.state_path = None
        self.active_groups = {}
        self.teacher_model_list = kwargs.get('teacher_model_list', [])
        for constraint in self.inequality_constraints_fn_list:
            if constraint.group_name is not None:
                self.target_groups.add(constraint.group_name)
                if constraint.group_name not in self.active_groups:
                    self.active_groups[constraint.group_name] = []
                for c in constraint.target_groups:
                    if c not in self.active_groups[constraint.group_name]:
                        self.active_groups[constraint.group_name].append(c.item())
                  
        assert self.macro_constraints_list is not None, f'{self.macro_constraints_list} has to be provided'
        self.group_cardinality = None
        #print('Device:',self.device)
        self._init_alm_parameters()
        
       

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
    
    def set_wasserstein_teachers_kwargs(self,teachers_kwargs):
        #print('Setting Wasserstein kwargs:',teachers_kwargs)
        self.wasserstein_kwargs.update(teachers_kwargs)

    def set_teachers_kwargs(self,teachers_kwargs):
         #print('Setting Teachers kwargs:',teachers_kwargs)
         self.teachers_kwargs.update(teachers_kwargs)
    
    def compute_groups_cardinality(self):
        groups = next(iter(self.get_train_loader_eval()))['groups']
        self.group_cardinality = {group_name: {} for group_name in self.target_groups} 
        self.max_cardinality = {group_name: 0 for group_name in self.target_groups}
        for group_name in self.target_groups:
            for group in groups[group_name].unique():
                self.group_cardinality[group_name].update({group.item(): len(groups[group_name][groups[group_name] == group])})               
                if len(groups[group_name][groups[group_name] == group]) > self.max_cardinality[group_name]:
                    self.max_cardinality[group_name] = len(groups[group_name][groups[group_name] == group])
        
    
    def _init_inequality_lambdas(self):
        self.inequality_lambdas = torch.ones_like(self.inequality_lambdas_0, device=self.device) * self.inequality_lambdas_0_value
    
    def _init_alm_parameters(self):
        self._init_inequality_lambdas()
        self.equality_lambdas = self.equality_lambdas_0
        self.mu = self.mu_0
        self.objective_multiplier = self.objective_multiplier_0

    def update_lambdas_inequality(self, constraints):
       
        if constraints is None:
            return self.inequality_lambdas
        
        new_lambdas = torch.max(
            torch.ones_like(self.inequality_lambdas, device=self.device)*self.inequality_lambdas_0_value,
            self.inequality_lambdas + self.mu * torch.max(constraints,torch.zeros_like(constraints, 
                                                                                       device=self.device))
        )
       
        assert torch.all(new_lambdas >= self.inequality_lambdas_0_value), 'Negative Lagrange multipliers!'
        return new_lambdas

    def update_lambdas_equality(self, constraints):
        
        if constraints is None:
            return self.equality_lambdas
        new_lambdas = self.equality_lambdas + self.mu * constraints * self.damping_factor
        new_lambdas = torch.clamp(new_lambdas, min=self.equality_lambdas_0_value, max=self.lambda_equality_max)
    
        return new_lambdas

    
   
    
    def update_alm_parameters_and_metrics(self, update_alm=True,**kwargs):
        metrics = {}
        self.model.eval()
        with torch.no_grad():
            kwargs = {}   
            val_loader = self.data_module.val_loader(batch_size=None)
            val_kwargs = self._compute_kwargs_in_batches(val_loader, self.model,use_entmax=False,use_training=False)
            kwargs['val_kwargs'] = val_kwargs
              
            if update_alm:
                train_loader = self.data_module.train_loader_eval(batch_size=None)
                train_kwargs = self._compute_kwargs_in_batches(train_loader, self.model,use_entmax=False,use_training=True)          
                kwargs['train_kwargs'] = train_kwargs
                inequality_constraints = train_kwargs['inequality_constraints']
                equality_constraints = train_kwargs['equality_constraints']
                self._apply_early_stopping(inequality_constraints, equality_constraints)

               
                if inequality_constraints is not None:
                    inequality_constraints = inequality_constraints * self.inequality_mask

                if equality_constraints is not None:
                    equality_constraints = equality_constraints * self.equality_mask

               
                if inequality_constraints is not None:
                    self.inequality_lambdas = self.update_lambdas_inequality(inequality_constraints)
                if equality_constraints is not None:
                    self.equality_lambdas = self.update_lambdas_equality(equality_constraints)

            val_score = self.compute_score(**val_kwargs)    
            metrics['val_constraints_score'] = val_score 
            val_loss = self.compute_loss_fn(**val_kwargs)
            metrics['val_loss'] = val_loss
            if not self.compute_only_score:
                metrics.update(self._compute_metrics(self.metrics,  prefix='val', **val_kwargs))
            return metrics
        
    def _apply_early_stopping(self, inequality_constraints, equality_constraints):
        n_inequality_constraints = len(self.inequality_constraints_fn_list)
        self.inequality_mask = torch.ones_like(self.inequality_lambdas, device=self.device)
        self.equality_mask = torch.ones_like(self.equality_lambdas, device=self.device)
        cached_scores = {}  
        for i, checkpoint in enumerate(self.lagrangian_checkpoints):
            if isinstance(checkpoint, EarlyStopping):
                if i < n_inequality_constraints:
                    if i not in cached_scores:
                        cached_scores[i] = {'violations': inequality_constraints[i]}
                    update, _ = checkpoint(metrics=cached_scores[i])
                    if not update:
                        self.inequality_mask[i] = 0  # Ferma l'aggiornamento per questo vincolo
                    else:
                        checkpoint.reset(keep_best=True)
                else:
                    eq_index = i - n_inequality_constraints
                    if eq_index not in cached_scores:
                        cached_scores[eq_index] = {'violations': equality_constraints[eq_index]}
                    update, _ = checkpoint(metrics=cached_scores[eq_index])
                    if not update:
                        self.equality_mask[eq_index] = 0  # Ferma l'aggiornamento per questo vincolo
                    else:
                        checkpoint.reset()
    
    def compute_constraints(self, **kwargs):
        device = self.device
        if len(self.inequality_constraints_fn_list)>0:
            inequality_constraints = torch.stack(
                [torch.clamp(constraint_fn(**kwargs),min=0) for constraint_fn in self.inequality_constraints_fn_list], dim=0
            ).to(device)
        else:
            inequality_constraints = torch.tensor([], device=device)

        if len(self.equality_constraints_fn_list)>0:
            equality_constraints = torch.stack(
                [constraint_fn(**kwargs) for constraint_fn in self.equality_constraints_fn_list], dim=0
            ).to(device)
        else:
            equality_constraints = torch.tensor([], device=device)


        return inequality_constraints, equality_constraints


    def compute_score(self, **kwargs):
        objective_function = kwargs.get('original_objective_function')
        inequality_constraints = kwargs.get('inequality_constraints')
        equality_constraints = kwargs.get('equality_constraints')
        #print('Original_objective function',objective_function.item())
        # Inizializza lo score con la funzione obiettivo
        score = 1-objective_function.clone()
        
        # Inizializza una variabile per il conteggio delle violazioni dei vincoli
        total_penalty = 0
        # Penalità per vincoli di disuguaglianza
        if len(inequality_constraints) > 0:
            for _,macro_constraint in enumerate(self.macro_constraints_list):
                if len(macro_constraint) > 0:
                    inequality_penalty = torch.max(torch.clamp(inequality_constraints[macro_constraint], min=0))   
                    total_penalty += inequality_penalty*self.gamma_constraint
                   
        if len(equality_constraints)> 0:
            equality_penalty = torch.max(torch.abs(equality_constraints))
            total_penalty += equality_penalty * self.gamma_constraint
        score -= total_penalty
        #if total_penalty > 0:
        #    print('Total Penalty:',total_penalty)
        #print('Total Penalty:',total_penalty)
        #assert score >= 0, 'Score non può essere negativo!'
        return score

       
    def compute_loss_fn(self, **kwargs):
        objective_function = kwargs['objective_function']
        batch_objective_function = kwargs['batch_objective_function']
        equality_constraints = kwargs.get('equality_constraints')
        inequality_constraints = kwargs.get('inequality_constraints')
        group_ids = kwargs.get('group_ids')

        assert group_ids is not None, 'Group ids must be provided'
        
        loss = objective_function.clone()
        if torch.isnan(loss).any():
            raise ValueError("NaN trovato nella loss!")
        group_losses = []

        for group_name in self.target_groups:
            group_tensor = group_ids[group_name]  # shape [B]
            unique_group_ids = torch.unique(group_tensor)

            total_weight = 0.0
            for group_id in unique_group_ids:
                mask = group_tensor == group_id
                if mask.sum() > 0:
                    group_loss = batch_objective_function[mask].mean()
                    weight = 1.0 - mask.sum().item() / batch_objective_function.shape[0]
                    group_losses.append(weight * group_loss)
                    total_weight += weight

        if len(group_losses) > 0:
            stacked_group_losses = torch.stack(group_losses)
            if torch.isnan(stacked_group_losses).any():
                print("NaN in group_losses:", stacked_group_losses)
            if total_weight > 0:
                loss += stacked_group_losses.sum() / total_weight
            else:
                print("total_weight is zero — skipping group loss contribution")

        
        # === Equality constraints
        if equality_constraints is not None and len(self.equality_constraints_fn_list) > 0:
            equality_penalty = torch.mean(torch.abs(equality_constraints)) * self.mu
            equality_lagrange = (self.equality_lambdas * equality_constraints).sum()
            loss += equality_penalty + equality_lagrange

        # === Inequality constraints
        if inequality_constraints is not None and len(self.inequality_constraints_fn_list) > 0:
            if torch.any(self.inequality_lambdas > 0):
                penalty = torch.sum(torch.clamp(inequality_constraints, min=0))
                lambdas_updated = torch.clamp(self.inequality_lambdas + self.mu * inequality_constraints, min=0)
                lagrange = (lambdas_updated.pow(2) - self.inequality_lambdas.pow(2)).sum() / (2 * self.mu)
                loss += penalty + lagrange

        if torch.isnan(loss).any():
            raise ValueError("NaN trovato nella loss!")

        return loss

    
    def _compute_kwargs_in_batches(self, loader, model, use_entmax=True,use_training=False):
        all_logits = []
        all_labels = []
        all_group_ids = {group_name: [] for group_name in loader.dataset[0]['groups'].keys()}
        all_group_ids_list = {group_name: [] for group_name in loader.dataset[0]['groups_ids_list'].keys()}
        all_group_masks = {group_name: [] for group_name in loader.dataset[0]['groups'].keys()}
        all_positive_masks = []
        all_indices = []

        
        
       
        for batch in loader:
            inputs = batch['data'].float().to(self.device)  
            outputs = model(inputs)  
            all_indices.append(batch['index'])

            all_logits.append(outputs)
            all_labels.append(batch['labels'].to(self.device))
            

           
            for group_name in batch['groups'].keys():
                all_group_ids[group_name].append(batch['groups'][group_name].to(self.device))
                all_group_masks[group_name].append(batch['groups'][group_name].to(self.device))
            
            
            for group_name in batch['groups_ids_list'].keys():
                all_group_ids_list[group_name].append(batch['groups_ids_list'][group_name].to(self.device))
            
            all_positive_masks.append(batch['positive_mask'].to(self.device))

        
        final_logits = torch.cat(all_logits, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        
        final_group_ids = {group_name: torch.cat(all_group_ids[group_name], dim=0) for group_name in all_group_ids}
        final_group_ids_list = {group_name: torch.cat(all_group_ids_list[group_name], dim=0) for group_name in all_group_ids_list}
        final_positive_masks = torch.cat(all_positive_masks, dim=0)
        final_indices = torch.cat(all_indices, dim=0)

       
        kwargs = {
            'logits': final_logits,
            'labels': final_labels,
            'groups': final_group_ids,
            'groups_ids_list': final_group_ids_list,
            'positive_mask': final_positive_masks,
            'index': final_indices,
            
        }

        kwargs = self._compute_kwargs(kwargs, final_logits, use_entmax=use_entmax,use_training=use_training)
        
        return kwargs

    def _compute_kwargs(self, batch, outputs, use_entmax=True,use_training=False):
        
        device = self.device
        
        group_ids = {group_name: batch['groups'][group_name].to(device) for group_name in batch['groups'].keys()}
        
       
        if 'groups_ids_list' in batch:
            group_ids_list = {group_name: batch['groups_ids_list'][group_name].to(device) for group_name in batch['groups_ids_list'].keys()}
        else:
            raise ValueError("'groups_ids_list' non è presente nel batch. Verifica la struttura del dataset.")

        
        positive_mask = batch.get('positive_mask', None)
        if positive_mask is None:
            raise ValueError("'positive_mask' non è presente nel batch.")
        positive_mask = positive_mask.to(device)

        labels = batch.get('labels', None)
        if labels is None:
            raise ValueError("'labels' non è presente nel batch.")
        labels = labels.to(device)

        predictions = torch.argmax(outputs, dim=-1)
        
        if use_entmax:
            probabilities = entmax_bisect(outputs, alpha=1.5, dim=-1)
        else:
            probabilities = torch.nn.functional.one_hot(predictions, num_classes=outputs.size(-1)).float()
        
        output_distribution = torch.nn.functional.softmax(outputs/1.0, dim=-1)
       
        if torch.isnan(probabilities).any():
            raise ValueError('Probabilità contiene NaN!')
       

        kwargs = {
            'group_ids': group_ids,  
            'group_ids_list': group_ids_list, 
            'group_masks': group_ids,  
            'positive_mask': positive_mask,  
            'logits': outputs,  
            'labels': labels,  
            'probabilities': probabilities,  
            'predictions': predictions,
            'output_distribution': output_distribution,
        }

        teacher_kwargs = self.teachers_kwargs['train'] if use_training else self.teachers_kwargs['val']
        if 'index' in batch:
            indices = batch['index']
        else:
            raise ValueError("Missing 'index' in batch — required to align teacher_kwargs")
        
        #print('Indices shape:',indices.shape)
        #print('Teacher kwargs:',teacher_kwargs)
        #print('Max indices: ',torch.max(indices))
        #for key,v in teacher_kwargs.items():
         #   print(f"Key: {key}, Shape: {v.shape}")
        filtered_teacher_kwargs = {}
        for k, v in teacher_kwargs.items():
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                if v.dim() >= 2 and v.shape[1] >= indices.max().item() + 1:
                    filtered_teacher_kwargs[k] = v[:, indices, ...]  # seleziona il batch corretto
                else:
                    print(f"[WARNING] Skipping teacher key {k}: shape={v.shape}, indices max={indices.max().item()}")
                    filtered_teacher_kwargs[k] = v
            else:
                filtered_teacher_kwargs[k] = v

        wasserstein_kwargs = self.wasserstein_kwargs['train'] if use_training else self.wasserstein_kwargs['val']
        filtered_wasserstein_kwargs = {}
        for k, v in wasserstein_kwargs.items():
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                if v.dim() >= 2 and v.shape[1] >= indices.max().item() + 1:
                    filtered_wasserstein_kwargs[f'wasserstein_{k}'] = v[:, indices, ...]  # seleziona il batch corretto
                else:
                    print(f"[WARNING] Skipping teacher key {k}: shape={v.shape}, indices max={indices.max().item()}")
                    filtered_wasserstein_kwargs[f'wasserstein_{k}'] = v
            else:
                filtered_wasserstein_kwargs[f'wasserstein_{k}'] = v
        #print('Filtered teacher kwargs:',filtered_teacher_kwargs)
        #for key in filtered_teacher_kwargs:
            #print(f"Key: {key}, Shape: {filtered_teacher_kwargs[key].shape}")
        kwargs.update(filtered_teacher_kwargs)
        kwargs.update(filtered_wasserstein_kwargs)
        #kwargs.update(teacher_kwargs)

        objective_fn_value = self.objective_fn(**kwargs)
        inequality_constraints, equality_constraints = self.compute_constraints(**kwargs)
        
        original_objective_fn_value = self.original_objective_fn(**kwargs)
        batch_objective_fn_value = self.batch_objective_function(**kwargs)
       
        kwargs['inequality_constraints'] = inequality_constraints
        kwargs['equality_constraints'] = equality_constraints
        kwargs['objective_function'] = objective_fn_value
        kwargs['batch_objective_function'] = batch_objective_fn_value
        kwargs['original_objective_function'] = original_objective_fn_value
    
        return kwargs

    def _compute_metrics(self,metrics,prefix='val',**kwargs):
        group_ids = kwargs['group_ids']
        y_pred = kwargs['predictions']
        y_true = kwargs['labels']

        tmp_result = {}
        final_result = {}
        
                
        for metric in metrics:
            metric.reset()
            if issubclass(metric.__class__,GroupFairnessMetric):
                            group_ids_detached = {group_name:group_ids[group_name].detach().cpu() for group_name in group_ids.keys()}
                            metric.calculate(y_pred.detach().cpu(),
                                            y_true.detach().cpu(),
                                            group_ids_detached)
                           
            elif isinstance(metric,Performance):
                metric.calculate(y_pred.detach().cpu(),
                                 y_true.detach().cpu())
            else:
                raise ValueError(f"{metric} is an invalid metric")
            tmp_result.update(metric.get())
            
      
        for key, value in tmp_result.items():
            if prefix == '':
                final_result[key] = value
            else:
                final_result[f'{prefix}_{key}'] = value
        return final_result 

    def _training_step(self, batch, batch_idx):
        self.model.train()
        inputs = batch['data'].float().to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        kwargs = self._compute_kwargs(batch, outputs,use_entmax=True,use_training=True)
        loss = self.compute_loss_fn(**kwargs)

        if torch.isnan(loss).any():
            raise ValueError('Loss contiene NaN!')
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        return loss.item()

    def _train_eval_step(self,**kwargs):
        self.model.eval()
        with torch.no_grad():
            train_kwargs = kwargs['train_kwargs']
            outputs = train_kwargs['logits']
            targets = train_kwargs['labels']
            loss = self.compute_loss_fn(**train_kwargs)
            predictions = torch.argmax(outputs, dim=1)

            return loss.item(), outputs, targets, predictions
            
    def _validation_step(self,**kwargs):
        self.model.eval()
        with torch.no_grad():
            val_kwargs = kwargs['val_kwargs']
            outputs = val_kwargs['logits']
            targets = val_kwargs['labels']
            loss = self.compute_loss_fn(**val_kwargs)
            predictions = torch.argmax(outputs, dim=1)

            return loss.item(), outputs, targets, predictions

    def set_constraints(self, inequality_constraints_fn_list, equality_constraints_fn_list,macro_constraints_list,inequality_lambdas, equality_lambdas):
        self.inequality_constraints_fn_list = inequality_constraints_fn_list
        self.equality_constraints_fn_list = equality_constraints_fn_list
        self.macro_constraints_list = macro_constraints_list
        self.inequality_lambdas=inequality_lambdas
        self.equality_lambdas=equality_lambdas

    def evaluate(self, model_dict, **kwargs):
       
        # Impostiamo il modello in modalità valutazione
        original_model_dict = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(self.device)
        metrics = self.update_alm_parameters_and_metrics(update_alm=False) 
        self.model.load_state_dict(original_model_dict)
      
        return metrics 
    
    def compute_val_kwargs(self,model_dict,use_training=False):
        original_model_dict = self.model.state_dict()
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(self.device)
        if use_training:
           loader = self.data_module.train_loader_eval(batch_size=None)
        else:
            loader = self.data_module.val_loader(batch_size=None)
        kwargs = self._compute_kwargs_in_batches(loader, self.model,use_entmax=False,use_training=use_training)
        self.model.load_state_dict(original_model_dict)
        self.model.to(self.device)
        return kwargs
    
    def compute_violations(self,val_kwargs,**kwargs):
        inequality_constraints, equality_constraints = self.compute_constraints(**val_kwargs)
        results = {}

        violations = {k:None for k,_ in enumerate(self.macro_constraints_list)}
        
        violations_per_group_list = {}
        violations_per_group = {}
        for key,value_dict in self.group_cardinality.items():
            violations_per_group_list[key] = {k:[] for k in value_dict.keys()}
        
        for i,constraint_violation in enumerate(inequality_constraints):
            constraint = self.inequality_constraints_fn_list[i]
            target_groups = constraint.target_groups
            group_name = constraint.group_name
            if group_name is not None:
                for group in target_groups:
                    try:
                        violations_per_group_list[group_name][group.item()].append(constraint_violation)
                    except KeyError:
                        if group_name not in violations_per_group_list:
                            violations_per_group_list[group_name] = {}
                        violations_per_group_list[group_name][group.item()] = [torch.tensor(0.0, device=self.device)]
        for key,value_dict in violations_per_group_list.items():
            try:
                violations_per_group[key] = {k:torch.stack(v).max().item() for k,v in value_dict.items()}
            except RuntimeError:
                violations_per_group[key] = 0
            
        
        results['violations_per_group'] = copy.deepcopy(violations_per_group)
        
        for i,macro_constraint in enumerate(self.macro_constraints_list):
            violations[i] = inequality_constraints[macro_constraint].detach().cpu().numpy()
        
        results['inequality_constraints_violations'] = inequality_constraints.detach().cpu().numpy()

        macro_constraints_violation = copy.deepcopy(violations)
        
        for i,_ in enumerate(self.macro_constraints_list):
            if len(macro_constraints_violation[i]) > 0:
                macro_constraints_violation[i] = [macro_constraints_violation[i].max()]
            else: 
                macro_constraints_violation[i] = []
        results['macro_constraints_violations'] = copy.deepcopy(macro_constraints_violation)
    
        return results
    
    
    def _progress_bar(self,iterable, **kwargs):
        if self.verbose:
            return tqdm.tqdm(iterable, **kwargs)
        else:
            return iterable

   
    def fit(self, **kwargs):
        #print('Current lambdas:',self.inequality_lambdas)
        if self.verbose:
            print(f'[{self.id}]:Number of inequality constraints:',len(self.inequality_constraints_fn_list))
            print(f'Macro constraints:',self.macro_constraints_list)
        num_epochs = kwargs.get('num_epochs', -1)
        disable_log = kwargs.get('disable_log', False)
        evaluate_best_model = kwargs.get('evaluate_best_model', True)
        n_rounds = self.num_epochs if num_epochs == -1 else num_epochs
        
        self.teacher_model_list = kwargs.get('teacher_model_list',[])
        #print(f'[LL {self.id}]:Number of teacher models:',len(self.teacher_model_list))
        start_model_dict = kwargs.get('start_model_dict')
        #print('[LL {self.id}]:Length of teacher model list:',len(self.teacher_model_list))
       
        if start_model_dict is not None:
            self.model.load_state_dict(copy.deepcopy(start_model_dict))
        

        self.model.to(self.device)
       
        """
        metrics = self.update_alm_parameters_and_metrics(update_alm=True) 
        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, EarlyStopping):
                stop, counter = checkpoint(metrics=metrics)
                metrics['early_stopping'] = counter
                if stop:
                    if not disable_log:
                        self.logger.log(metrics)
                    raise EarlyStoppingException

            elif isinstance(checkpoint, ModelCheckpoint):
                model_checkpoint = checkpoint(save_fn=self.save, metrics=metrics)
                metrics['model_checkpoint'] = 1 if model_checkpoint else 0
        
        if not disable_log:
            self.logger.log(metrics)
        """
        self.model.train()
        self.optimizer = self.optimizer_fn(self.model.parameters())
        
        try:
            for epoch in self._progress_bar(range(n_rounds), desc=f'Epoch 0/{n_rounds}', total=n_rounds, unit='epoch'):
                train_loader = self.data_module.train_loader()
                batch_iterator = self._progress_bar(train_loader, desc=f'Epoch {epoch+1}/{n_rounds}', leave=False)
                for batch_idx, batch in enumerate(batch_iterator):
                    self._training_step(batch, batch_idx)

                with torch.no_grad():                    
                    metrics = self.update_alm_parameters_and_metrics(update_alm=True,**kwargs)
                    
                    # Early stopping e model checkpoint
                    for checkpoint in self.checkpoints:
                        if isinstance(checkpoint, EarlyStopping):
                           
                            stop, counter = checkpoint(metrics=metrics)
                            metrics['early_stopping'] = counter
                            if stop:
                                if not disable_log:
                                    self.logger.log(metrics)
                                raise EarlyStoppingException

                        elif isinstance(checkpoint, ModelCheckpoint):
                           
                            model_checkpoint = checkpoint(save_fn=self.save, metrics=metrics)
                            metrics['model_checkpoint'] = 1 if model_checkpoint else 0
                            if model_checkpoint:
                                self.final_inequality_lambdas = copy.deepcopy(self.inequality_lambdas)
                                self.final_equality_lambdas = copy.deepcopy(self.equality_lambdas)

                    if not disable_log:
                        self.logger.log(metrics)
                    
                    if self.verbose and hasattr(batch_iterator, 'set_description'):
                        batch_iterator.set_description(f'Epoch {epoch+1}/{n_rounds}')
        except EarlyStoppingException:
            pass

        
        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, ModelCheckpoint):
                if os.path.exists(checkpoint.get_model_path()):
                    self.load(checkpoint.get_model_path())
        
        if evaluate_best_model:
            self.model.eval()
           
            metrics = self.update_alm_parameters_and_metrics(update_alm=True,**kwargs)
            final_metrics = {f'final_{name}': value for name, value in metrics.items()}
            if not disable_log:
               self.logger.log(final_metrics)

        #print('Final lambdas:',self.inequality_lambdas)  
        return copy.deepcopy(self.model.state_dict()),self.final_inequality_lambdas, self.final_equality_lambdas

    def query_teachers(self, teacher_model_dict_list, use_training=False, **kwargs):
        use_entmax = kwargs.get('use_entmax', True)

        all_teacher_logits = []
        teachers_probabilities_list = []
        teachers_predictions_list = []
        teachers_softmax_list = []
        teachers_logits_list = []
        if teacher_model_dict_list is not None and len(teacher_model_dict_list) > 0:
            if use_training:
                loader = self.data_module.train_loader_eval(batch_size=None)
            else:
                loader = self.data_module.val_loader(batch_size=None)
            
            for teacher_model_dict in teacher_model_dict_list:
                teacher_model = copy.deepcopy(self.model)
                teacher_model.load_state_dict(copy.deepcopy(teacher_model_dict))
                teacher_model.to(self.device)
                teacher_model.eval()

                teacher_outputs_list = []
                for batch in loader:
                    inputs = batch['data'].float().to(self.device)
                    with torch.no_grad():
                        outputs = teacher_model(inputs)
                    teacher_outputs_list.append(outputs)
                    all_teacher_logits.append(outputs)

                if len(teacher_outputs_list) > 0:
                    stacked_outputs = torch.cat(teacher_outputs_list, dim=0)
                else:
                    stacked_outputs = torch.tensor([], device=self.device)

                if stacked_outputs.numel() > 0:
                    teacher_softmax = torch.nn.functional.softmax(stacked_outputs / 1.0, dim=-1)
                    teacher_predictions = torch.argmax(stacked_outputs, dim=-1)

                    if use_entmax:
                        teacher_probabilities = entmax_bisect(stacked_outputs, alpha=1.5, dim=-1)
                    else:
                        teacher_probabilities = torch.nn.functional.one_hot(
                            teacher_predictions, num_classes=stacked_outputs.size(-1)
                        ).float()

                    if torch.isnan(teacher_probabilities).any():
                        raise ValueError("Teacher Probabilities contiene NaN!")

                    teachers_probabilities_list.append(teacher_probabilities)
                    teachers_predictions_list.append(teacher_predictions)
                    teachers_softmax_list.append(teacher_softmax)
                    teachers_logits_list.append(stacked_outputs)

        if len(teachers_probabilities_list)> 0:
            teacher_probabilities = torch.stack(teachers_probabilities_list, dim=0)
            teacher_predictions = torch.stack(teachers_predictions_list, dim=0)
            teacher_softmax = torch.stack(teachers_softmax_list, dim=0)
            teacher_logits = torch.stack(teachers_logits_list, dim=0)
        else:
            teacher_probabilities = torch.tensor([], device=self.device)
            teacher_predictions = torch.tensor([], device=self.device)
            teacher_softmax = torch.tensor([], device=self.device)
            teacher_logits = torch.tensor([], device=self.device)

        return {
            'teacher_probabilities': teacher_probabilities,
            'teacher_softmax_list': teacher_softmax,
            'teacher_logits_list': teacher_logits,
            'teacher_predictions_list': teacher_predictions,
        }
