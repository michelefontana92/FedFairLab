from debug_utils import debug_print

from wrappers import TorchNNWrapper
import copy
from .main_problem_orchestrator import MainProblemOrchestrator
from callbacks import EarlyStoppingException 



class OrchestratorWrapper(TorchNNWrapper):
    """
    Wrapper around the FairLAB main-problem orchestrator.

    It owns the model instance used by a client and exposes a compact API for
    training, evaluation, constraint evaluation, and ensemble-logit computation.
    """
    def __init__(self, *args,**kwargs):
        """
        Initialize the orchestrator wrapper and build the main FairLAB problem.

        Args:
            *args: Forwarded to ``TorchNNWrapper``.
            **kwargs: Model, constraints, objectives, optimizer, data module,
                metrics, callbacks, and FedFairLab aggregation settings.
        """
        super(OrchestratorWrapper, self).__init__(*args, **kwargs)
        # Estrarre i parametri necessari da kwargs, con valori di default ove appropriato
        self.loss_fn = kwargs.get('loss')
        self.inequality_constraints = kwargs.get("inequality_constraints", [])
        self.macro_constraints_list = kwargs.get("macro_constraints_list", [])
        self.target_groups = kwargs.get("target_groups", [])
        self.all_group_ids = kwargs.get("all_group_ids")
        self.num_classes = kwargs.get("num_classes", 2)
        assert self.all_group_ids is not None, 'all_group_ids must be provided'
        
        self.aggregation_teachers_list = kwargs.get("aggregation_teachers_list", [])


        self.optimizer_fn: callable = kwargs.get('optimizer_fn')
        self.objective_function = kwargs.get("objective_function")
        self.original_objective_fn = kwargs.get("original_objective_function")
        self.batch_objective_function = kwargs.get("batch_objective_function")

        self.equality_constraints = kwargs.get("equality_constraints")
        self.metrics = kwargs.get("metrics", [])
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.logger = kwargs.get("logger")
        self.lagrangian_checkpoints = kwargs.get("lagrangian_checkpoints", [])
        
        self.checkpoints = kwargs.get("checkpoints")
        self.checkpoints_config = kwargs.get("checkpoints_config")
        self.delta = kwargs.get("delta")
        self.performance_budget = kwargs.get("performance_constraint")
        self.performance_step = kwargs.get("performance_step", 0.0)
       
        self.current_model = self.model
        self.shared_macro_constraints = kwargs.get("shared_macro_constraints",[])
        self.max_constraints_in_subproblem = kwargs.get("max_constraints_in_subproblem",5)
        self.verbose = kwargs.get("verbose",False)
        self.options = {
                'optimizer_fn': self.optimizer_fn,
                'objective_fn': self.objective_function,
                'batch_objective_fn': self.batch_objective_function,
                'original_objective_fn': self.original_objective_fn,
                'metrics': self.metrics,
                'num_epochs': self.num_epochs,
                'logger': self.logger,
                'loss': self.loss_fn,
                'optimizer':self.optimizer,
                'data_module':self.data_module,
                'verbose':self.verbose,  
                'inequality_lambdas_0_value': 0,
                'performance_budget': self.performance_budget,
                'performance_step': self.performance_step,
            }
        
        self._build_main_problem()
    
    def set_model_params(self,model_params):
        """
        Load model parameters into the wrapped model.

        Args:
            model_params: State dict to load.
        """
        self.model.load_state_dict(model_params)
    
    def _build_main_problem(self,num_subproblems=5):
        """
        Instantiate the FairLAB main-problem controller.

        Args:
            num_subproblems: Number of subproblems used when partitioning
                fairness constraints.
        """
        for checkpoint in self.checkpoints:
            checkpoint.reset()
        #print('Teacher list:',len(self.aggregation_teachers_list))
        self.main_problem = MainProblemOrchestrator(
                                            model=copy.deepcopy(self.model),
                                            inequality_constraints=self.inequality_constraints,
                                            equality_constraints=self.equality_constraints,
                                            macro_constraints=self.macro_constraints_list,
                                            checkpoints_config=self.checkpoints_config,
                                            all_group_ids=self.all_group_ids,
                                            num_subproblems=num_subproblems,
                                            options=self.options,
                                            logger=self.logger,
                                            checkpoints=self.checkpoints,
                                            shared_macro_contraints=self.shared_macro_constraints,
                                            delta=self.delta,
                                            max_constraints_in_subproblem=self.max_constraints_in_subproblem,                                            
                                            aggregation_teachers_list = self.aggregation_teachers_list,
                                            num_classes = self.num_classes
                                           )

    
    
        
    
    def fit(self,model_params, num_global_iterations=1,num_local_epochs=5,num_subproblems=5,state=None,
            aggregation_teachers_list=[],aggregation_weights=None,aggregation_teacher_logits=None):
        """
        Train the wrapped model for a local or aggregation problem.

        Args:
            model_params: Starting model state dict.
            num_global_iterations: Number of orchestrator iterations.
            num_local_epochs: Epochs per selected local learner.
            num_subproblems: Constraint partitions to use.
            state: Optional persisted ALM/local client state.
            aggregation_teachers_list: Teacher model state dicts for local KD.
            aggregation_weights: Optional surrogate weights.
            aggregation_teacher_logits: Precomputed global ensemble target for
                aggregation-phase distillation.

        Returns:
            Tuple ``(model, state)`` with the updated model and persisted state.
        """
        
        self.main_problem.reset()
        self.main_problem.model.load_state_dict(model_params)
        self.main_problem.aggregation_teachers_list = aggregation_teachers_list
        self.main_problem.aggregation_teacher_logits = aggregation_teacher_logits
        self.main_problem.query_teachers()
        self.main_problem.eval_subproblem.instance.set_teachers_kwargs(
            self.main_problem.teachers_kwargs)

        debug_print('Number of aggregation teachers:',len(self.main_problem.aggregation_teachers_list))
        if self.logger is not None:
            metrics = self.main_problem.evaluate(self.main_problem.model)
            self.logger.log(metrics)
        
        try:
            if state is None:
                current_state = {}
            else: 
                current_state = copy.deepcopy(state)
                if 'teacher_history' not in current_state:
                    current_state['teacher_history'] = [{'model':copy.deepcopy(self.main_problem.model)}]
                self.main_problem.teacher_history = current_state['teacher_history']
            for i in range(num_global_iterations):
                if self.verbose:
                    debug_print('Iteration',i)
                #print('Iteration',i)
                
                new_state = self.main_problem.iterate(
                                    num_local_epochs=num_local_epochs,
                                    add_proximity_constraints=True,
                                    send_teacher_model=True,
                                    state=current_state,
                                    aggregation_weights=aggregation_weights,)
                
                current_state.update(new_state['state'])
        
        except EarlyStoppingException:
            debug_print('Early stopping')

        
        #state = self.get_state()
        self.main_problem.load_final_model()
        #state = self.get_state()
        self.main_problem.aggregation_teacher_logits = None
        return self.main_problem.model,current_state
    
    def evaluate(self, model_params, split='val'):
        """
        Evaluate a model using the main problem metrics.

        Args:
            model_params: State dict to evaluate.

        Returns:
            Metric dictionary.
        """
        
        model = copy.deepcopy(self.model)
        model.load_state_dict(model_params)
        metrics = self.main_problem.evaluate(model, split=split)
        return metrics
    
    
    def evaluate_constraints(self,model_params):
        """
        Compute train and validation constraint violations.

        Args:
            model_params: State dict to evaluate.

        Returns:
            Dictionary containing train and validation constraint outputs.
        """
        
        model = copy.deepcopy(self.model)
        model.load_state_dict(model_params)
        val_constraints,train_constraints = self.main_problem.compute_violations(model)
        return {'train':train_constraints,
                'val':val_constraints}
    
    def compute_kwargs(self,model_params,use_training=False):
        """
        Build the tensor payload used by objectives, metrics, and constraints.

        Args:
            model_params: State dict to evaluate.
            use_training: If true, use the training split; otherwise validation.

        Returns:
            Dictionary of logits, labels, groups, probabilities and masks.
        """
        kwargs = self.main_problem.eval_subproblem.instance.compute_val_kwargs(model_params,use_training=use_training)
        return kwargs
    
    def compute_score(self,model_params,use_training=False):
        """
        Compute the orchestrator score for a model.

        Args:
            model_params: State dict to score.
            use_training: Whether to score on the training split.

        Returns:
            Scalar score tensor/value from the local learner.
        """
        kwargs = self.compute_kwargs(model_params,use_training=use_training)
        score = self.main_problem.eval_subproblem.instance.compute_score(**kwargs)
        return score

    def compute_weighted_ensemble_logits(self,model_params_list,weights,use_training=True):
        """
        Compute weighted ensemble logits on the client's local data.

        Args:
            model_params_list: Candidate model state dicts.
            weights: Ensemble weights associated with the candidates.
            use_training: Whether to use the training split.

        Returns:
            Tensor of weighted logits.
        """
        return self.main_problem.compute_weighted_ensemble_logits(
            teacher_model_dict_list=model_params_list,
            weights=weights,
            use_training=use_training,
        )
    
    def evaluate_constraints2(self, model_params, split='val'):
        """
        Evaluate validation constraints and task metrics for server scoring.

        Args:
            model_params: State dict to evaluate.

        Returns:
            Dictionary with validation constraint violations, objective value and
            metric values.
        """
       
        model = copy.deepcopy(self.model)
        model.load_state_dict(model_params)
        
        eval_kwargs = self.main_problem.eval_subproblem.instance.compute_eval_kwargs(
            model_params, split=split)
        #train_kwargs=main_problem.eval_subproblem.instance.compute_val_kwargs(model_params,use_training=True)
        eval_constraints = self.main_problem.eval_subproblem.instance.compute_violations(eval_kwargs)
        #train_constraints = main_problem.eval_subproblem.instance.compute_violations(train_kwargs)
        eval_objective_fn = self.main_problem.eval_subproblem.instance.original_objective_fn(**eval_kwargs)
        #train_objective_fn = main_problem.eval_subproblem.instance.original_objective_fn(**train_kwargs)
        metrics = self.main_problem.evaluate(
            model, split=split, eval_kwargs=eval_kwargs)
        #print('Metrics:',metrics)
        #val_constraints,train_constraints = main_problem.compute_violations(model)
        return {#'train_constraints':train_constraints,
                f'{split}_constraints':eval_constraints,
                #'train_objective_fn':train_objective_fn.detach().cpu().item(),
                f'{split}_objective_fn':eval_objective_fn.detach().cpu().item(),
                'metrics':metrics}
    
