import torch
from wrappers import LocalLearner
from callbacks import EarlyStopping, ModelCheckpoint
import copy
from dataclasses import dataclass

@dataclass
class SubProblemConfig:
    """Configuration object that builds and owns a FairLAB local subproblem."""
    id: int
    inequality_constraints: list
    equality_constraints: list
    macro_constraints: list
    checkpoints_config: dict
    options: dict
    num_constraints: int
    compute_only_score: bool
    aggregation_teachers_list: list
    all_group_ids: dict

    def __post_init__(self):
        """Finalize initialization after dataclass fields have been populated."""
        self.reset()
       
    def reset(self):
        """Reset."""
        self.current_inequality_constraints = self.inequality_constraints
        self.current_macro_constraints = self.macro_constraints
        self.current_num_constraints = self.num_constraints
        self._init_checkpoints()
        
    def _init_checkpoints(self):
        """Handle init checkpoints."""
        self.checkpoints = [
                EarlyStopping(patience=5, 
                            monitor='val_constraints_score', 
                            mode='max'),
                ModelCheckpoint(save_dir=f"{self.checkpoints_config['checkpoint_dir']}/subproblem_{self.id}", 
                                save_name=f"{self.checkpoints_config['checkpoint_name']}", 
                                monitor='val_constraints_score', 
                                mode='max')
            ]
        
        self.lagrangian_checkpoints = [EarlyStopping(patience=2, 
                            monitor='violations', 
                            mode='min') for _ in range(len(self.current_inequality_constraints))]
    def set_alm(self,new_inequality_lambdas=None,new_equality_lambdas=None):
        """Set alm.
        
        Args:
            new_inequality_lambdas: Optional inequality Lagrange multipliers to restore.
            new_equality_lambdas: Optional equality Lagrange multipliers to restore.
        """
        if new_inequality_lambdas is not None:
            #print(f'New inequality lambdas: {new_inequality_lambdas}')
            self.instance.inequality_lambdas = new_inequality_lambdas
        
        inequality_lambdas = self.instance.inequality_lambdas 
        
        if new_equality_lambdas is not None:
            self.instance.equality_lambdas = new_equality_lambdas
        
        #print(f'Length of current_inequality_constraints: {len(self.current_inequality_constraints)}') 
        #print(f'Length of inequality_lambdas: {len(inequality_lambdas)}')
        additional_values = torch.full(
            (len(self.current_inequality_constraints) - len(inequality_lambdas),), self.instance.inequality_lambdas_0_value
            ).to(inequality_lambdas.device)
        inequality_lambdas = torch.cat([inequality_lambdas, additional_values])
        self.instance.inequality_lambdas = inequality_lambdas
        self.instance.inequality_constraints_fn_list = self.current_inequality_constraints
        self.instance.macro_constraints_list = self.current_macro_constraints
        self._init_checkpoints()
        self.instance.checkpoints = self.checkpoints
        self.instance.lagrangian_checkpoints = self.lagrangian_checkpoints

    def instanciate(self,model):
        """Handle instanciate.
        
        Args:
            model: Model instance or state to use.
        """
        self._init_checkpoints()
        config = self.options
        config['inequality_constraints'] = self.current_inequality_constraints
        config['equality_constraints'] = self.equality_constraints
        config['lagrangian_checkpoints'] = self.lagrangian_checkpoints
        config['macro_constraints_list'] = self.current_macro_constraints
        config['checkpoints'] = self.checkpoints
        config['compute_only_score'] = self.compute_only_score
        config['id'] = f'Subproblem {self.id}'
        config['teacher_model_list'] = self.aggregation_teachers_list
        config['subtract_upper_bound'] = self.options.get('subtract_upper_bound', False)
        config['all_group_ids'] = self.all_group_ids
        self.instance = LocalLearner(model=copy.deepcopy(model),**config)
    
