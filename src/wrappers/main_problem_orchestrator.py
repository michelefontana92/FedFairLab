import math
import random
import torch
from callbacks import EarlyStopping, ModelCheckpoint, EarlyStoppingException
import copy
from dataclasses import dataclass
import os
from loggers import WandbLogger
from .subproblem_config import SubProblemConfig

@dataclass
class MainProblemOrchestrator:
    """
    This class handles the logic of the orchestrator. It is responsible for selecting the learners,
    assigning the constraints to the learners, and updating the model.
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
        save_dict = {'model_state_dict': copy.deepcopy(self.model.state_dict()),
                     'inequality_lambdas': {},
                     'equality_lambdas': {}}
        
        
        torch.save(save_dict, path)

    def set_state(self,state):
       pass
    
    # Load the model from the specified path. The path should include the file name and extension.
    def load(self, path):
        save_dict = torch.load(path)
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        return save_dict
    
    def __post_init__(self):
        self.constraints_assignment = {
            'inequality_constraints': [],
            'equality_constraints': [],
        }
    
        self.split_problem = True
        self.assign_constraints()
        self.instanciate_subproblems(full_instance=True)
        
        self.active_groups = None
        self.violations_dict = None
        self.teacher_history = []
        self.c = 10
        self.current_model_idx = -1
        self.shock = False
        self.empty_state = True
        self.wasserstein_teachers_list = []
        self.query_teachers()
        self.query_wasserstein_teachers()
        self.eval_subproblem.instance.set_teachers_kwargs(self.teachers_kwargs)
        if len(self.inequality_constraints) == 0:
            self.num_subproblems = 1
            self.constraints_assignment['inequality_constraints'] = self._unique_assignment()
            self.instanciate_subproblems(full_instance=False)

        #print('After init Teachers kwargs:',self.teachers_kwargs)  
        #print('Main Problem Orchestrator initialized')
     
    def update_teacher_history(self,teacher_model,metric,violations_dict):
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
  
        metrics = torch.tensor([config['metric'] for i,config in enumerate(self.teacher_history)])
        tau=0.5
        probabilities = torch.nn.functional.softmax(-metrics / tau, dim=0)
        selected= torch.multinomial(probabilities, num_samples=1).item()
        selected =-1
        return selected
    
    def reset(self):
        for checkpoint in self.checkpoints:
            checkpoint.reset()
        self.teacher_history = []

    def compute_active_groups(self,selected_teacher_idx):
        tolerance = 0.05
        active_groups = {}
        
        violations_dict = self.teacher_history[selected_teacher_idx]['violations_per_group']
        
        for group_name, violations in violations_dict.items():
            active_groups[group_name] = []
            for idx,violation in violations.items():
                if violation <= tolerance:
                    
                        
                    active_groups[group_name].append({
                            'group_id': idx,
                            'delta': self.delta
                     })
                   
        active_groups_cpy = copy.deepcopy(active_groups)
        for group_name,group_list in active_groups_cpy.items():
            if len(group_list) == 0:
                del active_groups[group_name]

        self.active_groups = copy.deepcopy(active_groups) 


    def instanciate_subproblems(self,full_instance=True):
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
        self.teachers_kwargs = {'train':{},'val':{}}
        print(f'Querying {len(self.aggregation_teachers_list)} teachers')
        train_kwargs = self.eval_subproblem.instance.query_teachers(self.aggregation_teachers_list,use_training=True)
        val_kwargs = self.eval_subproblem.instance.query_teachers(self.aggregation_teachers_list,use_training=False)
        self.teachers_kwargs['train'] = train_kwargs
        self.teachers_kwargs['val'] = val_kwargs

    def query_wasserstein_teachers(self):
        self.wasserstein_teachers_kwargs = {'train':{},'val':{}}
        train_kwargs = self.eval_subproblem.instance.query_teachers(self.wasserstein_teachers_list,use_training=True)
        val_kwargs = self.eval_subproblem.instance.query_teachers(self.wasserstein_teachers_list,use_training=False)
        self.wasserstein_teachers_kwargs['train'] = train_kwargs
        self.wasserstein_teachers_kwargs['val'] = val_kwargs


    def iterate_without_constraints(self,num_local_epochs,send_teacher_model=False,state=None,aggregation_weights=None):
        print('Using iterate_without_constraints')
        problem = self.subproblems[0]
        problem.instance.reset()
        #print('Current model state:',self.model.state_dict()['fc1.weight'][:5,:])
        
        self.query_teachers()
        problem.instance.set_teachers_kwargs(self.teachers_kwargs)
        
        if aggregation_weights is not None:
            #print('Setting aggregation weights:',aggregation_weights)
            problem.instance.batch_objective_function.set_weights(aggregation_weights)
            problem.instance.objective_fn.set_weights(aggregation_weights)
            problem.instance.original_objective_fn.set_weights(aggregation_weights)
        
        updated_model, _, _ = problem.instance.fit(
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
        #print('[BEFORE ITERATE] Number of subproblems:',self.num_subproblems)
        
        
        if len(self.inequality_constraints) == 0 and len(self.equality_constraints) == 0:
            return self.iterate_without_constraints(num_local_epochs=num_local_epochs,
                                                    send_teacher_model=send_teacher_model,
                                                    state=state,
                                                    aggregation_weights=aggregation_weights)
        
        
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
                print(50*'-')
                print(f'\nSelected subproblem {selected} with violation (train) {self.violation_per_subproblem[selected]} (val) {self.val_violation_per_subproblem[selected]}')
                print(f'Max violation (train) {max_violation} (val) {max_violation_val}')
                print()
                print(50*'-')
                print()
        num_epochs = num_local_epochs
        updated_state = {}
        if send_teacher_model:
            delta = self.delta_per_subproblem[selected]
            if self.verbose:
                print(50*'-')
                print(f'\nSelected subproblem {selected} with delta {delta} and violation (train) {self.violation_per_subproblem[selected]} (val) {self.val_violation_per_subproblem[selected]}')
                print(f'Max violation (train) {max_violation} (val) {max_violation_val}')
                print()
                print(50*'-')
                print()
            
            for i in range(len(self.teacher_history)):
                #problem.add_global_proximity_constraint(i,
                #                                        delta,
                #                                        i==0)
                self.compute_active_groups(i)
                if self.verbose:
                    print('Active groups:',self.active_groups)
                
                if len(self.active_groups) > 0:
                    is_first = True
                    for group_name,group_list in self.active_groups.items():
                        for group in group_list:
                            problem.add_local_proximity_constraint(i,
                                                                    group_name,
                                                                    group['group_id'],
                                                                    delta,
                                                                    is_first)
                            is_first = False
            
            
            if state is not None:
                updated_state = copy.deepcopy(state)
                try: 
                    current_state = state[selected]   
                    new_inequality_lambdas = current_state['inequality_lambdas']
                    new_equality_lambdas = current_state['equality_lambdas']
                    problem.set_alm(new_inequality_lambdas=new_inequality_lambdas,
                                new_equality_lambdas=new_equality_lambdas)
                except KeyError:
                    #print('No state found for subproblem',selected)
                    #print('State:',state.keys())
                    problem.set_alm()
            #if state is not None:
            #    print('Problem: ',problem.instance.id)
           #     print('Loading state: Inequality lambdas: ',state['inequality_lambdas'])
                #self.set_state(state)
            #print(f'Sending teacher model of {len(self.aggregation_teachers_list)} teachers')    
            
            self.wasserstein_teachers_list = [self.teacher_history[i]['model'] for i in range(len(self.teacher_history))]
            
            self.query_wasserstein_teachers()
            
            self.eval_subproblem.instance.set_wasserstein_teachers_kwargs(self.wasserstein_teachers_kwargs)
            problem.instance.set_wasserstein_teachers_kwargs(self.wasserstein_teachers_kwargs)

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
        })
        return {'model': copy.deepcopy(self.model.state_dict()),
                'state': updated_state
                }
    
    def _compute_macro_constraints_violations_subproblems(self, val_kwargs):
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
        self.eval_subproblem.instance.set_teachers_kwargs(self.teachers_kwargs)
        val_kwargs = self.eval_subproblem.instance.compute_val_kwargs(model.state_dict(),use_training=False)
        eval_subproblem_violations = self.eval_subproblem.instance.compute_violations(val_kwargs)

        train_kwargs = self.eval_subproblem.instance.compute_val_kwargs(model.state_dict(),use_training=True)
        train_eval_subproblem_violations = self.eval_subproblem.instance.compute_violations(train_kwargs)
        return eval_subproblem_violations,train_eval_subproblem_violations
    
   

    def _random_assign_constraints(self):
        inequality_constraints_assignment = {}
        
        for macro_idx,macro_constraint in enumerate(self.macro_constraints):
            if macro_idx in self.shared_macro_contraints:
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
                for inequality_constraint_idx in macro_constraint:
                    inequality_constraints_assignment[inequality_constraint_idx] = {
                        'to': [i for i in range(self.num_subproblems)],
                        'macro_constraint': macro_idx
                    }
        #print('Number of subproblems:',self.num_subproblems)
        return inequality_constraints_assignment
    
    def _group_assign_constraints_multiclass(self):
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
                for inequality_constraint_idx in macro_constraint:
                    inequality_constraints_assignment[inequality_constraint_idx] = {
                        'to': [i for i in range(self.num_subproblems)],
                        'macro_constraint': macro_idx
                    }
        #print('Number of subproblems:',self.num_subproblems)
        return inequality_constraints_assignment
    
    def _split_assignments(self,assignment):
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
       
        self.num_subproblems = num_subproblems
       
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
        if self.split_problem and len(self.macro_constraints) > 0:
            if self.num_classes <0 :
                print('Using group assignment for multiclass problem')
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

    def evaluate(self,model,val_kwargs=None):
        self.eval_subproblem.instance.set_teachers_kwargs(self.teachers_kwargs)
        metrics = self.eval_subproblem.instance.evaluate(model.state_dict(),val_kwargs=val_kwargs)
        return metrics
    
    def select_subproblem(self, c1=100.0, c2=1.0):
  
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
        save_dict ={}
        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, ModelCheckpoint):
                if self.verbose:
                    print('Loading best model from:',checkpoint.get_model_path())
                if os.path.exists(checkpoint.get_model_path()):
                    save_dict=self.load(checkpoint.get_model_path())
                else:
                    if self.verbose:
                        print('No model found in:',checkpoint.get_model_path())
                    break
        #print('Loading final model:',save_dict['inequality_lambdas'])
        
        return save_dict
    
    def eval_final_model(self):
        self.load_final_model()
        self.model.eval()
        metrics = self.evaluate(self.model)
        if self.verbose:
            print('Best model evaluated: ', metrics)
        final_metrics = {f'final_{name}': value for name, value in metrics.items()}
        if self.logger is not None:
            self.logger.log(final_metrics)

