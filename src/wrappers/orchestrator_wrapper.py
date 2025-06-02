
from wrappers import TorchNNWrapper
import copy
from .main_problem_orchestrator import MainProblemOrchestrator
from callbacks import EarlyStoppingException 



class OrchestratorWrapper(TorchNNWrapper):
    """
    Implementation of the orchestrator.

    Methods:
        fit(num_global_iterations=5, num_local_epochs=5, num_subproblems=5):
            Fits the model using the specified number of global iterations, local epochs, and subproblems.
            Args:
                num_global_iterations (int): Number of global iterations. Default is 5.
                num_local_epochs (int): Number of local epochs. Default is 5.
                num_subproblems (int): Number of subproblems. Default is 5.
            Returns:
                The trained model.
    """
    def __init__(self, *args,**kwargs):
        super(OrchestratorWrapper, self).__init__(*args, **kwargs)
        # Estrarre i parametri necessari da kwargs, con valori di default ove appropriato
        self.loss_fn = kwargs.get('loss')
        self.inequality_constraints = kwargs.get("inequality_constraints", [])
        self.macro_constraints_list = kwargs.get("macro_constraints_list", [])
        self.target_groups = kwargs.get("target_groups", [])
        self.all_group_ids = kwargs.get("all_group_ids")
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
            }
        
        
    def set_model_params(self,model_params):
        self.model.load_state_dict(model_params)
    
    def _build_main_problem(self,num_subproblems=5):
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
                                           )

    
    
    def fit(self, num_global_iterations=5,num_local_epochs=5,num_subproblems=5,state=None):
        self._build_main_problem(num_subproblems=num_subproblems)
        
        metrics = self.main_problem.evaluate(self.main_problem.model)
        if self.logger is not None:
            self.logger.log(metrics)
        
       
        try:
            if state is None:
                current_state = {}
            else: 
                current_state = copy.deepcopy(state)
                self.main_problem.teacher_history = current_state['teacher_history']
            for i in range(num_global_iterations):
                if self.verbose:
                    print('Iteration',i)
                #print('Iteration',i)
                new_state = self.main_problem.iterate(num_local_epochs=num_local_epochs,
                                    add_proximity_constraints=True,
                                    send_teacher_model=True,
                                    state=current_state)
                current_state.update(new_state['state'])
        
        except EarlyStoppingException:
            print('Early stopping')

        
        #state = self.get_state()
        self.main_problem.load_final_model()
        #state = self.get_state()
        return self.main_problem.model,current_state
    
    def evaluate(self,model_params):
        self._build_main_problem()
        model = copy.deepcopy(self.model)
        model.load_state_dict(model_params)
        metrics = self.main_problem.evaluate(model)
        return metrics
    
    
    def evaluate_constraints(self,model_params):
        self._build_main_problem()
        model = copy.deepcopy(self.model)
        model.load_state_dict(model_params)
        val_constraints,train_constraints = self.main_problem.compute_violations(model)
        return {'train':train_constraints,
                'val':val_constraints}
    
    def compute_kwargs(self,model_params,use_training=False):
        self._build_main_problem()
        model = copy.deepcopy(self.model)
        model.load_state_dict(model_params)
        kwargs = self.main_problem.eval_subproblem.instance.compute_val_kwargs(model_params,use_training=use_training)
        return kwargs
    
    def compute_score(self,model_params,use_training=False):
        kwargs = self.compute_kwargs(model_params,use_training=use_training)
        score = self.main_problem.eval_subproblem.instance.compute_score(**kwargs)
        return score
    
    def evaluate_constraints2(self,model_params):
        self._build_main_problem()
        model = copy.deepcopy(self.model)
        model.load_state_dict(model_params)
        
        val_kwargs=self.main_problem.eval_subproblem.instance.compute_val_kwargs(model_params,use_training=False)
        #train_kwargs=main_problem.eval_subproblem.instance.compute_val_kwargs(model_params,use_training=True)
        val_constraints = self.main_problem.eval_subproblem.instance.compute_violations(val_kwargs)
        #train_constraints = main_problem.eval_subproblem.instance.compute_violations(train_kwargs)
        val_objective_fn = self.main_problem.eval_subproblem.instance.original_objective_fn(**val_kwargs)
        #train_objective_fn = main_problem.eval_subproblem.instance.original_objective_fn(**train_kwargs)
        metrics = self.main_problem.evaluate(model)
        #print('Metrics:',metrics)
        #val_constraints,train_constraints = main_problem.compute_violations(model)
        return {#'train_constraints':train_constraints,
                'val_constraints':val_constraints,
                #'train_objective_fn':train_objective_fn.detach().cpu().item(),
                'val_objective_fn':val_objective_fn.detach().cpu().item(),
                'metrics':metrics}
    