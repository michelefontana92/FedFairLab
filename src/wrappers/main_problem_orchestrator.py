from debug_utils import debug_print
import math
import random
import torch
from callbacks import EarlyStopping, ModelCheckpoint, EarlyStoppingException
import copy
from dataclasses import dataclass
import os
from loggers import WandbLogger
from .subproblem_config import SubProblemConfig
from checkpoint_utils import load_trusted_checkpoint

@dataclass
class MainProblemOrchestrator:
    """
    Coordinate FairLAB local learners for one client.

    The orchestrator partitions fairness constraints into subproblems, selects
    which local learner to update, manages ALM state, queries teacher models or
    precomputed teacher logits, and tracks the best local model.
    """
    model : torch.nn.Module
    inequality_constraints: list
    equality_constraints: list
    macro_constraints: list
    checkpoints_config: dict
    all_group_ids: dict
    num_subproblems: int
    options: dict
    logger:WandbLogger
    checkpoints: list
    shared_macro_contraints: list
    delta: float
    max_constraints_in_subproblem: int
    aggregation_teachers_list: list
    min_samples: int=2 
    verbose: bool=False
    num_classes: int=2
    

   

    # Save the model to the specified path. The path should include the file name and extension.
    def save(self, path):
        """
        Save the current orchestrator model state.

        Args:
            path: Destination checkpoint path.
        """
        save_dict = {'model_state_dict': copy.deepcopy(self.model.state_dict()),
                     'inequality_lambdas': {},
                     'equality_lambdas': {}}
        
        
        torch.save(save_dict, path)

    def set_state(self,state):
       """
       Placeholder for restoring serialized orchestrator state.

       Args:
           state: Serialized state dictionary.
       """
       pass
    
    # Load the model from the specified path. The path should include the file name and extension.
    def load(self, path):
        """
        Load a saved orchestrator model state.

        Args:
            path: Checkpoint path.

        Returns:
            Loaded checkpoint dictionary.
        """
        save_dict = load_trusted_checkpoint(path)
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        return save_dict
    
    def __post_init__(self):
        """
        Complete dataclass initialization.

        Builds constraint assignments, evaluation subproblems, teacher caches and
        local subproblems required for subsequent iterations.
        """
        self.constraints_assignment = {
            'inequality_constraints': [],
            'equality_constraints': [],
        }

        self.split_problem = True
        self.violations_dict = None
        self.teacher_history = []
        self.performance_budget = self.options.get('performance_budget')
        self.performance_step = self.options.get('performance_step', 0.0)
        self.performance_reference = None
        self.c = 10
        self.current_model_idx = -1
        self.shock = False
        self.empty_state = True
        self.aggregation_teacher_logits = None

        self.unconstrained_mode = (
            len(self.inequality_constraints) == 0
            and len(self.equality_constraints) == 0
        )
        if self.unconstrained_mode:
            self.num_subproblems = 0
            self.instanciate_subproblems(full_instance=True)
            self.query_teachers()
            self.eval_subproblem.instance.set_teachers_kwargs(self.teachers_kwargs)
            return

        self.assign_constraints()
        self.instanciate_subproblems(full_instance=True)
        self.query_teachers()
        self.eval_subproblem.instance.set_teachers_kwargs(self.teachers_kwargs)

        #print('After init Teachers kwargs:',self.teachers_kwargs)  
        #print('Main Problem Orchestrator initialized')

    def has_performance_budget(self):
        """
        Check whether FairLAB dynamic performance constraints are enabled.

        Returns:
            ``True`` when beta was provided and the first shared
            macro-constraint contains the two performance constraints used by
            FairLAB.
        """
        return (
            self.performance_budget is not None
            and len(self.inequality_constraints) >= 2
            and len(self.macro_constraints) > 0
            and self.macro_constraints[0][:2] == [0, 1]
        )

    def compute_performance_value(self, model):
        """
        Evaluate the predictive performance metric ``P`` on validation data.

        Args:
            model: Model whose parameters are evaluated.

        Returns:
            Floating-point value of the original performance objective, typically
            F1 in this codebase.
        """
        val_kwargs = self.eval_subproblem.instance.compute_val_kwargs(
            model.state_dict(),
            use_training=False,
        )
        performance = self.eval_subproblem.instance.original_objective_fn(**val_kwargs)
        return float(performance.detach().cpu().item())

    def update_performance_reference(self, model):
        """
        Update FairLAB's ``p*`` with the best validation performance seen so far.

        Args:
            model: Candidate model used to refresh the reference.

        Returns:
            Updated reference performance.
        """
        if not self.has_performance_budget():
            return None
        current_performance = self.compute_performance_value(model)
        if self.performance_reference is None:
            self.performance_reference = current_performance
        else:
            self.performance_reference = max(self.performance_reference, current_performance)
        return self.performance_reference

    def update_performance_constraints(self, model):
        """
        Rebuild FairLAB performance constraints from the current ``p*``.

        The budget constraint enforces ``P(m) >= p* - beta``. The improvement
        constraint seeks
        ``P(m) >= p* + rho_step`` and follows the same lifecycle as the budget
        constraint. The reference ``p*`` is initialized and updated from
        validation performance.

        Args:
            model: Current model at the beginning of the FairLAB iteration.
        """
        if not self.has_performance_budget():
            return
        p_star = self.update_performance_reference(model)
        improvement_target = min(1.0, p_star + self.performance_step)
        budget_target = max(0.0, p_star - self.performance_budget)
        self.inequality_constraints[0].upper_bound = improvement_target
        self.inequality_constraints[1].upper_bound = budget_target
        for subproblem in getattr(self, 'subproblems', {}).values():
            if len(subproblem.current_inequality_constraints) >= 2:
                subproblem.current_inequality_constraints[0].upper_bound = improvement_target
                subproblem.current_inequality_constraints[1].upper_bound = budget_target
            if hasattr(subproblem, 'instance') and len(subproblem.instance.inequality_constraints_fn_list) >= 2:
                subproblem.instance.inequality_constraints_fn_list[0].upper_bound = improvement_target
                subproblem.instance.inequality_constraints_fn_list[1].upper_bound = budget_target
        if hasattr(self, 'eval_subproblem'):
            self.eval_subproblem.current_inequality_constraints[0].upper_bound = improvement_target
            self.eval_subproblem.current_inequality_constraints[1].upper_bound = budget_target
            if hasattr(self.eval_subproblem, 'instance'):
                self.eval_subproblem.instance.inequality_constraints_fn_list[0].upper_bound = improvement_target
                self.eval_subproblem.instance.inequality_constraints_fn_list[1].upper_bound = budget_target
     
    def update_teacher_history(self,teacher_model,metric,violations_dict):
        """
        Store a teacher snapshot with its score and constraint violations.

        Args:
            teacher_model: Model to archive.
            metric: Selection metric associated with the teacher.
            violations_dict: Constraint violation summary for the teacher.
        """
        config = {
            'model': copy.deepcopy(teacher_model.state_dict()),
            'metric': metric,
            'violations_per_group': violations_dict['violations_per_group'],
            'violations_per_macro_constraint': violations_dict['macro_constraints_violations']
        }

        self.teacher_history.append(config)
        max_num_teachers =1
        self.current_model_idx += 1
        if len(self.teacher_history) > max_num_teachers:
            self.teacher_history = self.teacher_history[-max_num_teachers:]

    def select_teacher_model(self):
        """
        Select a teacher index from the local teacher history.

        Returns:
            Index of the selected teacher. The current implementation returns the
            most recent teacher for deterministic local behavior.
        """
  
        metrics = torch.tensor([config['metric'] for i,config in enumerate(self.teacher_history)])
        tau=0.5
        probabilities = torch.nn.functional.softmax(-metrics / tau, dim=0)
        selected= torch.multinomial(probabilities, num_samples=1).item()
        selected =-1
        return selected
    
    def reset(self):
        """Reset callbacks and clear local teacher history before a new fit."""
        for checkpoint in self.checkpoints:
            checkpoint.reset()
        self.teacher_history = []

    def instanciate_subproblems(self,full_instance=True):
        """
        Instantiate evaluation or trainable subproblems.

        Args:
            full_instance: If true, create only the full evaluation problem;
                otherwise create local learner and violation subproblems.
        """
        if full_instance:
            self.eval_subproblem = self.build_subproblem(-1,eval_problem=True)
            self.eval_subproblem.instanciate(self.model)
            self.eval_subproblem.instance.compute_groups_cardinality()
        else:
            self.subproblems = {i:self.build_subproblem(i) for i in range(self.num_subproblems)}
            self.violation_subproblems = {i:self.build_subproblem(i) for i in range(self.num_subproblems)}
            self.attempts = [1 for _ in range(self.num_subproblems)]
            for subproblem in self.subproblems.values():
                subproblem.instanciate(self.model)
                subproblem.set_alm()
            for subproblem in self.violation_subproblems.values():
                subproblem.instanciate(self.model)
 
        
    def query_teachers(self):
        """
        Populate cached teacher tensors for train and validation data.

        If ``aggregation_teacher_logits`` is set, it is used directly as the
        distillation target for the aggregation phase.
        """
        self.teachers_kwargs = {'train':{},'val':{}}
        if self.aggregation_teacher_logits is not None:
            self.eval_subproblem.instance.set_ensemble_teacher_logits(self.aggregation_teacher_logits)
            self.teachers_kwargs['train'] = copy.deepcopy(self.eval_subproblem.instance.teachers_kwargs['train'])
            self.teachers_kwargs['val'] = copy.deepcopy(self.eval_subproblem.instance.teachers_kwargs['val'])
            return
        debug_print(f'Querying {len(self.aggregation_teachers_list)} teachers')
        train_kwargs = self.eval_subproblem.instance.query_teachers(self.aggregation_teachers_list,use_training=True)
        val_kwargs = self.eval_subproblem.instance.query_teachers(self.aggregation_teachers_list,use_training=False)
        self.teachers_kwargs['train'] = train_kwargs
        self.teachers_kwargs['val'] = val_kwargs

    def compute_weighted_ensemble_logits(self, teacher_model_dict_list, weights, use_training=True):
        """
        Compute weighted ensemble logits through the evaluation local learner.

        Args:
            teacher_model_dict_list: Candidate model state dicts.
            weights: Ensemble weights supplied by the server.
            use_training: Whether to use the training split.

        Returns:
            Tensor of weighted local logits.
        """
        return self.eval_subproblem.instance.compute_weighted_ensemble_logits(
            teacher_model_dict_list=teacher_model_dict_list,
            weights=weights,
            use_training=use_training,
        )

    def iterate_without_constraints(self,num_local_epochs,send_teacher_model=False,state=None,aggregation_weights=None):
        """
        Train the model when no fairness constraints are active.

        Args:
            num_local_epochs: Number of epochs for the local learner.
            send_teacher_model: Compatibility flag for teacher transmission.
            state: Optional persisted local state.
            aggregation_weights: Optional distillation surrogate weights.

        Returns:
            Updated model state and an empty state dictionary.
        """
        debug_print('Using iterate_without_constraints')
        learner = self.eval_subproblem.instance
        learner.reset()
        
        self.query_teachers()
        learner.set_teachers_kwargs(self.teachers_kwargs)
        
        if aggregation_weights is not None:
            #print('Setting aggregation weights:',aggregation_weights)
            learner.batch_objective_function.set_weights(aggregation_weights)
            learner.objective_fn.set_weights(aggregation_weights)
            learner.original_objective_fn.set_weights(aggregation_weights)
        
        updated_model, _, _ = learner.fit(
            start_model_dict=self.model.state_dict(),
            num_epochs=num_local_epochs,
            disable_log=True  
        )
        self.model.load_state_dict(updated_model)
        metrics = self.evaluate(self.model)

        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, EarlyStopping):
                stop, counter = checkpoint(metrics=metrics)
                metrics['early_stopping'] = counter
                if stop and self.logger is not None:
                    self.logger.log(metrics)
                    raise EarlyStoppingException
            elif isinstance(checkpoint, ModelCheckpoint):
                model_checkpoint = checkpoint(save_fn=self.save, metrics=metrics)
                metrics['model_checkpoint'] = 1 if model_checkpoint else 0

       
        if self.logger is not None:
            self.logger.log(metrics)
        #print('New model state:',self.model.state_dict()['fc1.weight'][:5,:])
        return {'model': copy.deepcopy(self.model.state_dict()), 'state': {}}
    
    def iterate(self,num_local_epochs=1,add_proximity_constraints=True,
                send_teacher_model=False,
                state=None,aggregation_weights=None):
        """
        Run one FairLAB local orchestration step.

        Args:
            num_local_epochs: Epochs for the selected learner.
            add_proximity_constraints: Whether proximity constraints may be used.
            send_teacher_model: Compatibility flag for teacher transmission.
            state: Persisted ALM state from previous rounds.
            aggregation_weights: Optional aggregation/distillation weights.

        Returns:
            Dictionary containing updated model parameters and local state.
        """
        #print('[BEFORE ITERATE] Number of subproblems:',self.num_subproblems)
        
        
        if len(self.inequality_constraints) == 0 and len(self.equality_constraints) == 0:
            return self.iterate_without_constraints(num_local_epochs=num_local_epochs,
                                                    send_teacher_model=send_teacher_model,
                                                    state=state,
                                                    aggregation_weights=aggregation_weights)

        if state is not None and 'performance_reference' in state:
            self.performance_reference = state['performance_reference']
        self.update_performance_constraints(self.model)
        
        
        if self.violations_dict is None:
            self.val_violations_dict,self.violations_dict = self.compute_violations(self.model)
            self.instanciate_subproblems(full_instance=False)
            self._set_violation_per_subproblem(self.violations_dict,self.val_violations_dict)
            self.delta_max = self.delta
            self.delta_min=self.delta
            self.delta_step = self.delta
            self.delta_per_subproblem = {i:self.delta_min for i in range(self.num_subproblems)}
            self.is_eligible = {i:True for i in range(self.num_subproblems)}
        else: 
            self.val_violations_dict,self.violations_dict = self.compute_violations(self.model)
            self._set_violation_per_subproblem(self.violations_dict,self.val_violations_dict)    
            #if state is not None:
                #self.model.load_state_dict(state['model_state_dict'])   
                #for learner, inequality_lambdas, equality_lambdas in zip(self.subproblems.values(),
                #                                                        state['inequality_lambdas'], 
                #                                                        state['equality_lambdas']):
                #    learner.instance.inequality_lambdas = inequality_lambdas
                #    learner.instance.equality_lambdas = equality_lambdas    
        #print('[BEFORE SELECT] Number of subproblems:',self.num_subproblems)
        selected = self.select_subproblem(c1=10)
       
        
        problem = self.subproblems[selected]
       
        problem.reset()
        original_constraint_count = len(problem.instance.inequality_constraints_fn_list)
        
        problem.instance.set_teachers_kwargs(self.teachers_kwargs)
        
        if problem.instance.group_cardinality is None:
            problem.instance.compute_groups_cardinality()
        
       
        
        
        max_violation = torch.max(torch.tensor([v for v in self.violation_per_subproblem.values()])).item()
        max_violation_val = torch.max(torch.tensor([v for v in self.val_violation_per_subproblem.values()])).item()
        if self.verbose:
            if not send_teacher_model:
                debug_print(50*'-')
                debug_print(f'\nSelected subproblem {selected} with violation (train) {self.violation_per_subproblem[selected]} (val) {self.val_violation_per_subproblem[selected]}')
                debug_print(f'Max violation (train) {max_violation} (val) {max_violation_val}')
                debug_print()
                debug_print(50*'-')
                debug_print()
        num_epochs = num_local_epochs
        updated_state = {}
        if send_teacher_model:
            if self.verbose:
                debug_print(50*'-')
                debug_print(f'\nSelected subproblem {selected} with violation (train) {self.violation_per_subproblem[selected]} (val) {self.val_violation_per_subproblem[selected]}')
                debug_print(f'Max violation (train) {max_violation} (val) {max_violation_val}')
                debug_print()
                debug_print(50*'-')
                debug_print()

            if state is not None:
                updated_state = copy.deepcopy(state)
                try: 
                    current_state = state[selected]   
                    new_inequality_lambdas = current_state['inequality_lambdas']
                    new_equality_lambdas = current_state['equality_lambdas']
                    problem.set_alm(new_inequality_lambdas=new_inequality_lambdas,
                                new_equality_lambdas=new_equality_lambdas)
                except KeyError:
                    problem.set_alm()

            updated_model,self.inequality_lambda,self.equality_lambda = problem.instance.fit(start_model_dict = self.model.state_dict(),
                                                 num_epochs=num_epochs,
                                                 disable_log=True,
                                                 teacher_model_list=self.aggregation_teachers_list,
                                                 use_first_model = False,
                                                )
        
            
            updated_state.update({selected: {
                'inequality_lambdas': copy.deepcopy(self.inequality_lambda[:original_constraint_count]),
                'equality_lambdas': copy.deepcopy(self.equality_lambda[:original_constraint_count])
            }})
            #print('Updated state:',updated_state.keys())
        else:
            updated_model,self.inequality_lambda,self.equality_lambda = problem.instance.fit(start_model_dict = self.model.state_dict(),
                                                 num_epochs=num_epochs,
                                                 disable_log=True,
                                                 use_first_model = False
                                                 )
        
            updated_state.update({selected: {
                'inequality_lambdas': copy.deepcopy(self.inequality_lambda[:original_constraint_count]),
                'equality_lambdas': copy.deepcopy(self.equality_lambda[:original_constraint_count])
            }})

        self.model.load_state_dict(updated_model)
        metrics = self.evaluate(self.model)
        self.update_performance_reference(self.model)
        old_violation_per_subproblem = copy.deepcopy(self.violation_per_subproblem)
        
        self.instanciate_subproblems(full_instance=False)
        val_new_violations_dict,new_violations_dict = self.compute_violations(self.model)
        self._set_violation_per_subproblem(new_violations_dict,val_violations_dict=val_new_violations_dict)
        same_violations = True
        
        for i in range(self.num_subproblems):
            if old_violation_per_subproblem[i] != self.violation_per_subproblem[i]:
                same_violations = False
                break
        
        if same_violations:
            self.delta_per_subproblem[selected] += self.delta_step
            self.delta_per_subproblem[selected] = min(self.delta_max,self.delta_per_subproblem[selected])
            self.is_eligible[selected] = True
            self.shock = True            
        else:
            self.delta_per_subproblem[selected] = max(self.delta_min,self.delta_per_subproblem[selected] - self.delta_step)
            for i in range(self.num_subproblems):
                self.is_eligible[i] = True
            self.shock = False
        self.violations_dict = copy.deepcopy(new_violations_dict)
        self.val_violations_dict = copy.deepcopy(val_new_violations_dict)
        self.update_teacher_history(self.model,metrics['val_constraints_score'],self.violations_dict)
        
        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, EarlyStopping):
                stop, counter = checkpoint(metrics=metrics)
                metrics['early_stopping'] = counter
                if stop:
                    if self.logger is not None:
                        self.logger.log(metrics)
                    raise EarlyStoppingException

            elif isinstance(checkpoint, ModelCheckpoint):
                model_checkpoint = checkpoint(save_fn=self.save, metrics=metrics)
                metrics['model_checkpoint'] = 1 if model_checkpoint else 0
        if self.logger is not None:
            self.logger.log(metrics)            
        
        updated_state.update({
            'teacher_history': self.teacher_history,
            'performance_reference': self.performance_reference,
        })
        return {'model': copy.deepcopy(self.model.state_dict()),
                'state': updated_state
                }
    
    def _compute_macro_constraints_violations_subproblems(self, val_kwargs):
        """
        Compute maximum macro-constraint violation per subproblem.

        Args:
            val_kwargs: Validation tensors and metadata.

        Returns:
            Tensor with one violation value per subproblem.
        """
        final_violations = []
        for i in range(self.num_subproblems):
            current_violations=self.violation_subproblems[i].instance.compute_violations(val_kwargs)
            total_violations = 0 
            for key,value in current_violations['macro_constraints_violations'].items():
                if key not in self.shared_macro_contraints:
                    if len(value)>0:
                        if value[0] > total_violations:
                            total_violations = value[0]
            final_violations.append(total_violations)
        
        final_violations = torch.tensor(final_violations)
        return final_violations
    
   
    def compute_violations(self,model):
        """
        Compute validation and training violations for a model.

        Args:
            model: Model instance to evaluate.

        Returns:
            Tuple ``(validation_violations, training_violations)``.
        """
        self.eval_subproblem.instance.set_teachers_kwargs(self.teachers_kwargs)
        val_kwargs = self.eval_subproblem.instance.compute_val_kwargs(model.state_dict(),use_training=False)
        eval_subproblem_violations = self.eval_subproblem.instance.compute_violations(val_kwargs)

        train_kwargs = self.eval_subproblem.instance.compute_val_kwargs(model.state_dict(),use_training=True)
        train_eval_subproblem_violations = self.eval_subproblem.instance.compute_violations(train_kwargs)
        return eval_subproblem_violations,train_eval_subproblem_violations
    
   

    def _random_assign_constraints(self):
        """
        Randomly assign macro-constraints to subproblems.

        Returns:
            Assignment dictionary for inequality constraints.
        """
        inequality_constraints_assignment = {}
        
        for macro_idx,macro_constraint in enumerate(self.macro_constraints):
            if macro_idx in self.shared_macro_contraints:
                if self.num_subproblems == 0:
                    self.num_subproblems = 1
                for inequality_constraint_idx in macro_constraint:
                    inequality_constraints_assignment[inequality_constraint_idx] = {
                        'to': [i for i in range(self.num_subproblems)],
                        'macro_constraint': macro_idx
                    }
            else:
                assignment = [random.randint(0, self.num_subproblems - 1) for _ in range(len(macro_constraint))]
                for inequality_constraint_idx in macro_constraint:
                    inequality_constraints_assignment[inequality_constraint_idx] = {
                        'to': [assignment[macro_constraint.index(inequality_constraint_idx)]],
                        'macro_constraint': macro_idx
                    }
        return inequality_constraints_assignment
    
    def _group_assign_constraints(self):
        """
        Assign binary fairness constraints to subproblems by group structure.

        Returns:
            Assignment dictionary for inequality constraints.
        """
        inequality_constraints_assignment = {}
        self.num_subproblems = 0
        for group_name,_ in self.all_group_ids.items():
            num_subproblems = 0
            for macro_idx,macro_constraint in enumerate(self.macro_constraints):
                if macro_idx not in self.shared_macro_contraints:
                    for inequality_constraint_idx in macro_constraint:
                        current_constraint = self.inequality_constraints[inequality_constraint_idx]
                        if (current_constraint.group_name is not None) and  (current_constraint.group_name==group_name):
                            inequality_constraints_assignment[inequality_constraint_idx] = {
                                'to': [ self.num_subproblems+g.item() for g in self.inequality_constraints[inequality_constraint_idx].target_groups],
                                'macro_constraint': macro_idx
                            }
                            num_subproblems = max(self.num_subproblems,max(inequality_constraints_assignment[inequality_constraint_idx]['to']))
            
            self.num_subproblems += num_subproblems +1
        for macro_idx,macro_constraint in enumerate(self.macro_constraints):
            if macro_idx in self.shared_macro_contraints:
                if self.num_subproblems == 0:
                    self.num_subproblems = 1
                for inequality_constraint_idx in macro_constraint:
                    inequality_constraints_assignment[inequality_constraint_idx] = {
                        'to': [i for i in range(self.num_subproblems)],
                        'macro_constraint': macro_idx
                    }
        #print('Number of subproblems:',self.num_subproblems)
        return inequality_constraints_assignment
    
    def _group_assign_constraints_multiclass(self):
        """
        Assign multiclass fairness constraints to class-aware subproblems.

        Returns:
            Assignment dictionary for inequality constraints.
        """
        inequality_constraints_assignment = {}
        self.num_subproblems = 0
        for group_name,_ in self.all_group_ids.items():
            for current_class in range(self.num_classes):
                num_subproblems = 0
                for macro_idx,macro_constraint in enumerate(self.macro_constraints):
                    if macro_idx not in self.shared_macro_contraints:
                        for inequality_constraint_idx in macro_constraint:
                            current_constraint = self.inequality_constraints[inequality_constraint_idx]
                            if (current_constraint.group_name is not None) and  (current_constraint.group_name==group_name) and (current_constraint.target_class == current_class):
                                inequality_constraints_assignment[inequality_constraint_idx] = {
                                    'to': [ self.num_subproblems+g.item() for g in self.inequality_constraints[inequality_constraint_idx].target_groups],
                                    'macro_constraint': macro_idx
                                }
                                num_subproblems = max(self.num_subproblems,max(inequality_constraints_assignment[inequality_constraint_idx]['to']))
                
                self.num_subproblems += num_subproblems +1
        for macro_idx,macro_constraint in enumerate(self.macro_constraints):
            if macro_idx in self.shared_macro_contraints:
                if self.num_subproblems == 0:
                    self.num_subproblems = 1
                for inequality_constraint_idx in macro_constraint:
                    inequality_constraints_assignment[inequality_constraint_idx] = {
                        'to': [i for i in range(self.num_subproblems)],
                        'macro_constraint': macro_idx
                    }
        #print('Number of subproblems:',self.num_subproblems)
        return inequality_constraints_assignment
    
    def _split_assignments(self,assignment):
        """
        Split oversized assignments into smaller subproblems.

        Args:
            assignment: Initial constraint-to-subproblem assignment.

        Returns:
            Assignment dictionary respecting ``max_constraints_in_subproblem``.
        """
        new_assignments = copy.deepcopy(assignment)
        for _,value in new_assignments.items():
            value['to'] = []
        
        
        
        num_constraints_per_subproblem = {i:0 for i in range(self.num_subproblems)}
        constraints_per_subproblem = {i:[] for i in range(self.num_subproblems)}
        
        
        for key,value in assignment.items():
            for subproblem in value['to']:
                if value['macro_constraint'] not in self.shared_macro_contraints:
                    num_constraints_per_subproblem[subproblem] += 1
                constraints_per_subproblem[subproblem].append(key)
        
        constraints_per_subproblem_cpy = copy.deepcopy(constraints_per_subproblem)
        for key,value in constraints_per_subproblem_cpy.items():
            if len(value) == 0:
                del constraints_per_subproblem[key]
            else:
                if value == self.shared_macro_contraints:
                    del constraints_per_subproblem[key]
        num_subproblems = 0
    
        for problem_id,constraints in constraints_per_subproblem.items():
            
            if num_constraints_per_subproblem[problem_id] > self.max_constraints_in_subproblem:
                n_new_problems = math.ceil(len(constraints) / self.max_constraints_in_subproblem)
                idx = 0
                for _ in range(n_new_problems):
                    current_constraints = constraints[idx:idx+self.max_constraints_in_subproblem]
                    idx += self.max_constraints_in_subproblem
                    for constraint in current_constraints:
                       macro_constraint = new_assignments[constraint]['macro_constraint']
                       if macro_constraint not in self.shared_macro_contraints:
                        new_assignments[constraint]['to'].append(num_subproblems)
                    
                    num_subproblems += 1
            else:
                for constraint in constraints:
                    macro_constraint = new_assignments[constraint]['macro_constraint']
                    if macro_constraint not in self.shared_macro_contraints:
                        new_assignments[constraint]['to'].append(num_subproblems)
                
                
                num_subproblems += 1
       
        self.num_subproblems = max(1, num_subproblems) if len(self.macro_constraints) > 0 else num_subproblems
       
        for macro_idx,macro_constraint in enumerate(self.macro_constraints):
            if macro_idx in self.shared_macro_contraints:
                for inequality_constraint_idx in macro_constraint:
                    new_assignments[inequality_constraint_idx] = {
                        'to': [i for i in range(self.num_subproblems)],
                        'macro_constraint': macro_idx
                    }
        #print('Number of subproblems:',self.num_subproblems)
        return new_assignments
    
    def _set_violation_per_subproblem(self,violations_dict, val_violations_dict):
        """
        Attach current violation magnitudes to each subproblem.

        Args:
            violations_dict: Training violation dictionary.
            val_violations_dict: Validation violation dictionary.

        Returns:
            Tuple of training and validation violation maps per subproblem.
        """
        self.violation_per_subproblem = {i:0 for i in range(self.num_subproblems)}
        self.val_violation_per_subproblem = {i:0 for i in range(self.num_subproblems)}
        for key,value in enumerate(violations_dict['inequality_constraints_violations']):
            for subproblem in self.constraints_assignment['inequality_constraints'][key]['to']:
                if self.constraints_assignment['inequality_constraints'][key]['macro_constraint'] not in self.shared_macro_contraints:
                    if value > self.violation_per_subproblem[subproblem]:
                        self.violation_per_subproblem[subproblem] = value
        
        for key,value in enumerate(val_violations_dict['inequality_constraints_violations']):
            for subproblem in self.constraints_assignment['inequality_constraints'][key]['to']:
                if self.constraints_assignment['inequality_constraints'][key]['macro_constraint'] not in self.shared_macro_contraints:
                    if value > self.val_violation_per_subproblem[subproblem]:
                        self.val_violation_per_subproblem[subproblem] = value
        
        return self.violation_per_subproblem,self.val_violation_per_subproblem
    

    def _unique_assignment(self):
        """
        Create a single assignment containing all constraints.

        Returns:
            Assignment dictionary with one subproblem.
        """
        inequality_constraints_assignment = {}
        self.num_subproblems = 1
        
        for macro_idx,macro_constraint in enumerate(self.macro_constraints):
            for inequality_constraint_idx in macro_constraint:
                inequality_constraints_assignment[inequality_constraint_idx] = {
                    'to': [0],
                    'macro_constraint': macro_idx
                }
                            
        return inequality_constraints_assignment
    
    def assign_constraints(self,violations_dict=None):
        """
        Assign fairness constraints to local subproblems.

        Args:
            violations_dict: Optional violation information for adaptive assignment.
        """
        if self.split_problem and len(self.macro_constraints) > 0:
            if self.num_classes > 2:
                debug_print('Using group assignment for multiclass problem')
                group_assignment = self._group_assign_constraints_multiclass()
            else:
                group_assignment = self._group_assign_constraints()
            assignment = self._split_assignments(group_assignment)
        else: 
            assignment = self._unique_assignment()

        self.constraints_assignment['inequality_constraints']=assignment
        #print('Constraints assignment:',self.constraints_assignment['inequality_constraints'])
        #print('Number of subproblems:',self.num_subproblems)
        if violations_dict is not None:
            self._set_violation_per_subproblem(violations_dict)

    def build_subproblem(self,problem_id,eval_problem=False):
        """
        Build a subproblem configuration.

        Args:
            problem_id: Subproblem identifier, or ``-1`` for evaluation.
            eval_problem: Whether to include all constraints for evaluation.

        Returns:
            ``SubProblemConfig`` ready to instantiate a local learner.
        """
        
        if eval_problem:
            return SubProblemConfig(id=problem_id,
                         inequality_constraints=self.inequality_constraints,
                         equality_constraints=self.equality_constraints,
                         macro_constraints=self.macro_constraints,
                         checkpoints_config=self.checkpoints_config,
                         options=self.options,
                         num_constraints=len(self.inequality_constraints),
                         compute_only_score=False,
                         all_group_ids=self.all_group_ids,
                         aggregation_teachers_list=self.aggregation_teachers_list) 
        
        inequality_constraints = []
        sub_macro_constraints = []
        num_constraints = 0
        
        for _,macro_constraint in enumerate(self.macro_constraints):
            constraints_indices = [idx for idx in macro_constraint if problem_id in self.constraints_assignment['inequality_constraints'][idx]['to']]
            #print('Subproblem',problem_id,'macro constraints:',constraints_indices)
            inequality_constraints.extend([self.inequality_constraints[idx] for idx in constraints_indices])
            sub_macro_constraints.append(list(range(num_constraints,num_constraints+len(constraints_indices))))
            num_constraints += len(constraints_indices)
        
        return SubProblemConfig(id=problem_id,
                         inequality_constraints=inequality_constraints,
                         equality_constraints=self.equality_constraints,
                         macro_constraints=sub_macro_constraints,
                         checkpoints_config=self.checkpoints_config,
                         options=self.options,
                         num_constraints=num_constraints,
                         compute_only_score=True,
                         all_group_ids=self.all_group_ids,
                         aggregation_teachers_list=self.aggregation_teachers_list)

    def evaluate(self, model, split='val', eval_kwargs=None):
        """
        Evaluate model metrics through the full evaluation subproblem.

        Args:
            model: Model instance to evaluate.
            split: Evaluation split. Must be ``val`` or ``test``.
            eval_kwargs: Optional payload precomputed for ``split``.

        Returns:
            Metric dictionary.
        """
        self.eval_subproblem.instance.set_teachers_kwargs(self.teachers_kwargs)
        payload = {f'{split}_kwargs': eval_kwargs} if eval_kwargs is not None else {}
        metrics = self.eval_subproblem.instance.evaluate(
            model.state_dict(), split=split, **payload)
        return metrics
    
    def select_subproblem(self, c1=100.0, c2=1.0):
        """
        Select the next subproblem to optimize.

        Args:
            c1: Scale factor for violation-based selection.
            c2: Reserved exploration scale.

        Returns:
            Selected subproblem id.
        """
  
        #print('Selecting subproblems among',self.num_subproblems,'subproblems')
        violations_per_subproblem_tensor = torch.tensor([self.violation_per_subproblem[i] for i in range(self.num_subproblems)])
        for i in range(self.num_subproblems):
            if not self.is_eligible[i]:
                violations_per_subproblem_tensor[i] = 0
            
        alpha = torch.clamp(c1 * violations_per_subproblem_tensor, min=0)
        tau=0.5
           
        if torch.sum(violations_per_subproblem_tensor) == 0:
            eligible_subproblems = [i for i in range(self.num_subproblems) if self.is_eligible[i]] 
            if len(eligible_subproblems) > 0:
                selected = random.choice(eligible_subproblems)
                return selected
            else:
               
                selected = torch.randint(0,self.num_subproblems,(1,)).item()
                return selected
            
        probabilities = torch.nn.functional.softmax(alpha / tau, dim=0)

        stop=False    
        while not stop:
            selected= torch.multinomial(probabilities, num_samples=1).item()
            if violations_per_subproblem_tensor[selected] > 0:
                if self.is_eligible[selected]:
                    stop = True
                    
        return selected
    
    def load_final_model(self):
        """
        Load the best checkpointed model from local learner checkpoints.

        Returns:
            Loaded checkpoint dictionary, or an empty dictionary if unavailable.
        """
        save_dict ={}
        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, ModelCheckpoint):
                if self.verbose:
                    debug_print('Loading best model from:',checkpoint.get_model_path())
                if os.path.exists(checkpoint.get_model_path()):
                    save_dict=self.load(checkpoint.get_model_path())
                else:
                    if self.verbose:
                        debug_print('No model found in:',checkpoint.get_model_path())
                    break
        #print('Loading final model:',save_dict['inequality_lambdas'])
        
        return save_dict
    
    def eval_final_model(self):
        """
        Evaluate and log final metrics for the current model.
        """
        self.load_final_model()
        self.model.eval()
        metrics = self.evaluate(self.model)
        if self.verbose:
            debug_print('Best model evaluated: ', metrics)
        final_metrics = {f'final_{name}': value for name, value in metrics.items()}
        if self.logger is not None:
            self.logger.log(final_metrics)
