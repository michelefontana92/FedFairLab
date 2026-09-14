from debug_utils import debug_print
from .base_builder import Base_Builder
from metrics import MetricsFactory
from surrogates import SurrogateFactory
from wrappers import OrchestratorWrapper
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger
from torch.nn import CrossEntropyLoss
import os
import torch
import copy
from client import ClientFactory
from server import ServerFactory
import ray
import math


class FedFairLabBuilder(Base_Builder):
    """
    Builder for FedFairLab experiments.

    The builder converts run configuration into concrete data modules, metrics,
    objectives, clients, server, callbacks, loggers, and Ray resource settings.
    """
    @staticmethod
    def _resolve_checkpoint_dir(checkpoint_dir, validation_strategy,
                                cv_folds=5, fold_id=None):
        """Return a fold-isolated checkpoint directory for K-fold runs."""
        if validation_strategy != 'kfold':
            return checkpoint_dir
        cv_folds = int(cv_folds)
        if fold_id is None or not 0 <= int(fold_id) < cv_folds:
            raise ValueError(
                f"fold_id must be between 0 and {cv_folds - 1} for K-fold CV")
        return os.path.join(checkpoint_dir, f'fold_{int(fold_id)}')

    def _assign_resources(self):
        """
        Compute Ray CPU/GPU resources for server and client actors.

        ``clients_per_core`` controls fractional CPU allocation per client. For
        example, two clients per core gives each client actor ``0.5`` CPU.
        """
        num_clients = self.num_clients
        self.clients_per_core = max(1, getattr(self, 'clients_per_core', 1))
        available_cpus = max(1, os.cpu_count() or 1)
        requested_cpus = math.ceil(num_clients / self.clients_per_core)
        self.num_cpus = min(available_cpus, requested_cpus)
        # All client actors are long-lived, so their combined CPU reservations
        # must fit in Ray at once. This also uses one full CPU per client when
        # the machine has enough cores (e.g. 10 cores for 10 clients).
        self.num_cpus_per_client = min(
            1.0 / self.clients_per_core,
            self.num_cpus / num_clients,
        )
        self.num_gpus = len(self.gpu_devices)
     
        self.num_gpus_per_client = self.num_gpus/num_clients if self.num_gpus > 0 else 0
        
    def compute_group_cardinality(self,group_name,sensitive_attributes):
        """
        Compute the number of groups induced by a sensitive attribute.

        Args:
            group_name: Name of the sensitive attribute or intersectional group.
            sensitive_attributes: Dataset metadata describing possible values.

        Returns:
            Cardinality of the requested group.

        Raises:
            KeyError: If the group is not present in ``sensitive_attributes``.
        """
        for name,group_dict in sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 
    
    def __init__(self,**kwargs):
        """
        Build all experiment components except Ray runtime execution.

        Args:
            **kwargs: Run configuration produced by ``RunFactory`` and CLI
                options, including federation size, resources, constraints, and
                execution settings.
        """
        super(FedFairLabBuilder,self).__init__(**kwargs)
        self.num_clients = kwargs.get('num_clients', 1)
        self.clients_per_core = max(1, kwargs.get('clients_per_core', 1))
        self.gpu_devices = tuple(kwargs.get('gpu_devices') or ())
        self._assign_resources()
        self.id = kwargs.get('id')
        self.run_dict = kwargs.get('run_dict')
        self.common_client_params  = self._get_common_params(**kwargs)
        self.experiment_name = kwargs.get('experiment_name')
        self.clients = []
        self.evaluate_only = kwargs.get('evaluate_only', False)
        self.evaluate_test = kwargs.get('evaluate_test', False)
        for i in range(self.num_clients):
            client = self._build_client(f'{self.id}_client_{i+1}',i+1,**kwargs)
            self.clients.append(client)
        self.server = self._build_server(**kwargs)
        
    def _get_common_params(self,**kwargs):
        """
        Assemble parameters shared by clients and server.

        Args:
            **kwargs: Experiment configuration and run-specific defaults.

        Returns:
            Dictionary containing metrics, losses, constraints, optimizer,
            callbacks configuration, data/model settings, and FedFairLab options.
        """
        common_params = {}
        common_params['metrics_list'] = tuple(kwargs.get('metrics_list') or ())
        common_params['groups_list'] = tuple(kwargs.get('groups_list') or ())
        common_params['threshold_list'] = tuple(kwargs.get('threshold_list') or ())
        common_params['lr'] = self.run_dict['lr']
        common_params['loss'] = partial(CrossEntropyLoss)
        common_params['num_lagrangian_epochs'] = kwargs.get('num_lagrangian_epochs', 1)
        common_params['batch_size'] = self.run_dict['batch_size']
        common_params['random_seed'] = kwargs.get('random_seed')
        common_params['fraction'] = kwargs.get('client_fraction', 0.5)
        common_params['project_name'] = kwargs.get('project_name')
        base_checkpoint_dir = kwargs.get(
            'checkpoint_dir', f'checkpoints/{common_params["project_name"]}')
        validation_strategy = kwargs.get('validation_strategy')
        common_params['checkpoint_dir'] = self._resolve_checkpoint_dir(
            base_checkpoint_dir,
            validation_strategy,
            cv_folds=kwargs.get('cv_folds', 5),
            fold_id=kwargs.get('fold_id'),
        )
        
        common_params['verbose'] = kwargs.get('verbose', False)
        common_params['optimizer_fn'] = partial(Adam, lr=common_params['lr'])
        
        common_params['monitor'] = kwargs.get('monitor', 'val_constraints_score')
        common_params['mode'] = kwargs.get('mode', 'max')
        
        
        common_params['log_model'] = kwargs.get('log_model', False)
        common_params['num_global_iterations'] = kwargs.get('num_global_iterations')
        common_params['num_local_iterations'] = kwargs.get('num_local_iterations')
        common_params['aggregation_epochs'] = kwargs.get('aggregation_epochs', 1)
        common_params['aggregation_local_epochs'] = kwargs.get('aggregation_local_epochs', 10)
        common_params['aggregation_patience'] = kwargs.get(
            'aggregation_patience', 0)
        common_params['aggregation_min_delta'] = kwargs.get(
            'aggregation_min_delta', 1e-6)
        common_params['history_size'] = kwargs.get('history_size', 5)
        
        common_params['performance_constraint'] = kwargs.get('performance_constraint')
        common_params['performance_step'] = kwargs.get('performance_step', 0.0)
        common_params['delta'] = kwargs.get('delta', 0.2)
        common_params['max_constraints_in_subproblem'] = kwargs.get('max_constraints_in_subproblem')
        common_params['global_patience'] = kwargs.get('global_patience')
        common_params['local_patience'] = kwargs.get('local_patience')
        common_params['num_classes'] = self.run_dict['num_classes']
        self.num_classes = common_params['num_classes']
        debug_print('Number of classes:', self.num_classes)
        debug_print('Groups: ', common_params['groups_list'])
        # Callbacks
        
        # Metriche
        common_params['metrics'] = [MetricsFactory().create_metric('performance',num_classes=common_params['num_classes'])]

        # Funzione obiettivo e vincoli
        common_params['objective_function'] = SurrogateFactory.create(name='performance', surrogate_name='cross_entropy', weight=1, average='weighted',num_classes=common_params['num_classes'])
        common_params['batch_objective_function'] = SurrogateFactory.create(name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted',num_classes=common_params['num_classes'])
        if common_params['num_classes'] > 2:
            common_params['original_objective_fn'] = SurrogateFactory.create(name='multiclass_f1', surrogate_name='multiclass_f1', weight=1, average='weighted',num_classes=common_params['num_classes'])
        else:
            common_params['original_objective_fn'] = SurrogateFactory.create(name='binary_f1', surrogate_name='binary_f1', weight=1, average='weighted',num_classes=common_params['num_classes'])
        common_params['equality_constraints'] = []
        common_params['shared_macro_constraints'] = []
        debug_print()

        if common_params['performance_constraint'] is not None:
            debug_print('Performance budget beta: ', common_params['performance_constraint'])
            debug_print('Performance improvement step rho: ', common_params['performance_step'])
            if common_params['num_classes'] > 2:
                performance_surrogate = 'multiclass_f1'
            else:
                performance_surrogate = 'binary_f1'
            common_params['inequality_constraints'] = [
                SurrogateFactory.create(name=performance_surrogate, 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=1.0,
                                    use_max=False),
                SurrogateFactory.create(name=performance_surrogate, 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=1.0,
                                    use_max=False)
            ]
            common_params['lagrangian_callbacks'] = [EarlyStopping(patience=2, 
                                                                   monitor='score', 
                                                                   mode='max') for _ in range(2)]
            common_params['macro_constraints_list'] = [[0, 1]]
            common_params['shared_macro_constraints'] = [0]
          
        else:
            debug_print('No performance constraint')
            debug_print()
            common_params['inequality_constraints'] = []
            common_params['lagrangian_callbacks'] = []
            common_params['macro_constraints_list'] = []
         
        # Configurazione dei macro vincoli
        
        for key,value in self.run_dict.items():
           
            if key not in common_params:
                common_params[key] = value
        
       
        for metric, group, threshold in zip(common_params['metrics_list'], common_params['groups_list'], common_params['threshold_list']):
            common_params['threshold'] = threshold
            common_params['metric'] = metric
            common_params['training_group_name'] = group
            common_params['num_groups'] = self.compute_group_cardinality(common_params['training_group_name'],common_params['sensitive_attributes'])
            common_params['group_ids'] = {common_params['training_group_name']: list(range(common_params['num_groups']))}
            
            # Aggiunta della metrica
            common_params['metrics'] += [MetricsFactory().create_metric(metric, group_ids=common_params['group_ids'], group_name=common_params['training_group_name'],
                                                                        num_classes=common_params['num_classes'],)]
           
        common_params['optimizer'] = Adam(copy.deepcopy(self.run_dict['model']).parameters(),
                          lr=common_params['lr']
                          )
    
        
        return common_params

    
    def _build_client(self,client_name,client_idx,**kwargs):
        """
        Create a client initialization function for one federated silo.

        Args:
            client_name: Stable logger/checkpoint identifier for the client.
            client_idx: One-based client index used to select data partition files.
            **kwargs: Builder configuration overrides.

        Returns:
            Partial function that creates the configured Ray client actor.
        """
        client_params = copy.deepcopy(self.common_client_params)
        client_params['client_name'] = client_name
        experiment_seed = self.common_client_params.get('random_seed')
        client_params['random_seed'] = (
            None if experiment_seed is None else experiment_seed + client_idx)
        checkpoint_name = kwargs.get('checkpoint_name', f'{client_name}_local.h5')
        client_params['checkpoint_name'] = checkpoint_name   
        client_params['callbacks'] = [
            EarlyStopping(patience=client_params['local_patience'], monitor=client_params['monitor'], mode=client_params['mode']),
            ModelCheckpoint(save_dir=client_params['checkpoint_dir'], save_name=kwargs.get('checkpoint_name', checkpoint_name), 
                                                                                           monitor=client_params['monitor'], mode=client_params['mode'])
        ]

        client_params['client_checkpoint_name'] = kwargs.get('client_checkpoint_name', f'{client_name}_local_final.h5')  
        client_params['client_callbacks'] = [
            ModelCheckpoint(save_dir=client_params['checkpoint_dir'], 
                            save_name=client_params['client_checkpoint_name'], 
                            monitor=client_params['monitor'],
                            mode=client_params['mode'])
        ]


       
        
        config = {
            'hidden1': client_params['hidden1'],
            'hidden2': client_params['hidden2'],
            'dropout': client_params['dropout'],
            'lr': client_params['lr'],
            'batch_size': client_params['batch_size'],
            'random_seed': client_params['random_seed'],
            'dataset': client_params['dataset'],
            'optimizer': 'Adam',
            'num_lagrangian_epochs': client_params['num_lagrangian_epochs'],
            'num_epochs': client_params['num_local_iterations'],
            'patience': client_params['global_patience'],
            'monitor': client_params['monitor'],
            'mode': client_params['mode'],
            'log_model': client_params['log_model'],
            'validation_strategy': client_params.get(
                'validation_strategy', 'external'),
            'cv_folds': client_params.get('cv_folds', 5),
            'fold_id': client_params.get('fold_id'),
        }
        
        checkpoints_config = {
            'checkpoint_dir': client_params['checkpoint_dir'],
            'checkpoint_name': client_params['checkpoint_name'],
            'monitor': client_params['monitor'],
            'mode': client_params['mode'],
            'patience': client_params['global_patience']
        }
        client_params['checkpoints_config'] = checkpoints_config
        client_params['config'] = config
         # Creazione del DataModule
        path = self._resolve_client_data_prefix(client_params, client_idx)
        validation_strategy = client_params.get(
            'validation_strategy', 'external')
        train_set, validation_set, test_set = self._data_split_paths(path)
        client_params['data_module'] = DataModule(dataset=client_params["dataset"], 
                                               root=client_params["data_root"], 
                                               train_set=train_set,
                                                 val_set=validation_set,
                                                 test_set=test_set,
                                                 load_test_set=True,
                                                 validation_strategy=validation_strategy,
                                                 validation_fraction=client_params.get('validation_fraction', 0.2),
                                                 split_seed=client_params.get('split_seed', 42),
                                                 stratify_columns=client_params.get('stratify_columns', ()),
                                                 cv_folds=client_params.get('cv_folds', 5),
                                                 fold_id=client_params.get('fold_id'),
                                                 clean_data_path=client_params.get('clean_data_path'),
                                                 batch_size=client_params["batch_size"], 
                                                 num_workers=client_params.get(
                                                     'num_workers', 0),
                                                 use_local_weights=False,
                                                 sensitive_attributes=client_params["sensitive_attributes"])

        # Configurazione del logger
        client_params['logger'] = partial(WandbLogger,
                                          project=client_params["project_name"], 
                                  config=config, 
                                  id=client_name,
                                  checkpoint_dir=client_params["checkpoint_dir"], 
                                  checkpoint_path=client_params["checkpoint_name"],
                                  log_model=client_params["log_model"],
                                  data_module=client_params["data_module"] if client_params["log_model"] else None
                                  )

        orchestrator = partial(
            OrchestratorWrapper,
            model=copy.deepcopy(client_params['model']),
            inequality_constraints=client_params['inequality_constraints'],
            macro_constraints_list=client_params['macro_constraints_list'],
            optimizer_fn=client_params['optimizer_fn'],
            optimizer=client_params['optimizer'],
            objective_function=client_params['objective_function'],
            equality_constraints=client_params['equality_constraints'],
            metrics=client_params['metrics'],
            num_epochs=client_params['num_local_iterations'],
            loss=client_params['loss'],
            data_module=client_params['data_module'],
            lagrangian_checkpoints=client_params['lagrangian_callbacks'],
            checkpoints=client_params['callbacks'],
            checkpoints_config=client_params['checkpoints_config'],
            shared_macro_constraints=client_params['shared_macro_constraints'],
            delta=client_params['delta'],
            performance_constraint=client_params['performance_constraint'],
            performance_step=client_params['performance_step'],
            max_constraints_in_subproblem=(
                client_params['max_constraints_in_subproblem']),
        )

        return partial(
            ClientFactory().create,
            'client_fedfairlab',
            remote=True,
            num_cpus=self.num_cpus_per_client,
            num_gpus=self.num_gpus_per_client,
            orchestrator=orchestrator,
            client_name=client_name,
            logger=client_params['logger'],
            model=client_params['model'],
            num_global_iterations=client_params['num_global_iterations'],
            num_local_iterations=client_params['num_local_iterations'],
            client_callbacks=client_params['client_callbacks'],
            config=client_params,
        )

    @staticmethod
    def _data_split_paths(data_prefix):
        """Return the train and test CSV paths used by built-in runs.

        Holdout and K-fold validation rows are derived from ``*_train.csv``.
        The ``*_test.csv`` path is never used until explicit final evaluation.
        """
        return (
            f'{data_prefix}_train.csv',
            None,
            f'{data_prefix}_test.csv',
        )

    def _resolve_client_data_prefix(self, client_params, client_idx):
        """Resolve the experiment-nested or direct client-data layout."""
        data_file_prefix = client_params.get(
            'data_file_prefix', client_params['dataset'])
        root = client_params['data_root']
        candidates = [
            f'{self.experiment_name}/node_{client_idx}/{data_file_prefix}',
            f'node_{client_idx}/{data_file_prefix}',
        ]
        for prefix in candidates:
            if os.path.exists(os.path.join(root, f'{prefix}_train.csv')):
                return prefix
        return candidates[0]
    

    def _build_server(self,**kwargs):
        """
        Instantiate the federated server.

        Args:
            **kwargs: Server and experiment configuration.

        Returns:
            Server object or Ray actor handle.
        """
        server_params = copy.deepcopy(self.common_client_params)
        server_params['server_name'] = f'{self.id}_server'
        server_params['checkpoint_name'] = kwargs.get('checkpoint_name', f'{server_params["server_name"]}_global.h5')
        server_params['checkpoint_dir'] = self.common_client_params['checkpoint_dir']
       
        server_params['model'] = copy.deepcopy(server_params['model'])
        server_params['metrics'] = kwargs.get('metrics')
        server_params['num_federated_iterations'] = kwargs.get('num_federated_iterations')
        server_params['aggregation_epochs'] = kwargs.get('aggregation_epochs', server_params['aggregation_epochs'])
        server_params['aggregation_local_epochs'] = kwargs.get('aggregation_local_epochs', server_params['aggregation_local_epochs'])
        server_params['aggregation_patience'] = kwargs.get(
            'aggregation_patience',
            server_params.get('aggregation_patience', 0))
        server_params['aggregation_min_delta'] = kwargs.get(
            'aggregation_min_delta',
            server_params.get('aggregation_min_delta', 1e-6))
        server_params['history_size'] = kwargs.get('history_size', server_params['history_size'])
        server_params['num_classes'] =self.num_classes

        return ServerFactory().create(
            'server_fedfairlab',
            clients_init_fn_list=self.clients,
            **server_params,
        )
    
    def run(self):
        """
        Run training or evaluation for the configured experiment.

        This initializes Ray with computed resources, starts the server,
        executes global rounds, and shuts down all actors.
        """
        validation_strategy = getattr(
            self, 'common_client_params', {}).get(
                'validation_strategy', 'external')
        if self.evaluate_only and validation_strategy == 'kfold':
            raise ValueError(
                "Evaluation-only on the external test is disabled for CV folds")

        debug_print('Number of CPUs:',self.num_cpus)
        debug_print('Number of GPUs:',self.num_gpus)
        debug_print('Number of GPUs per client:',self.num_gpus_per_client)
        debug_print('Number of CPUs per client:',self.num_cpus_per_client)
        debug_print('Number of clients per core:',self.clients_per_core)
        ray.init(num_cpus=self.num_cpus,num_gpus=self.num_gpus)
        self.server.setup()
        if self.evaluate_only:
            self.server.evaluate_saved_checkpoints_on_test()
            self.server.shutdown(
                log_global_results=False,
                log_client_results=True,
            )
        else:
            self.server.execute()
            if validation_strategy == 'kfold':
                self.server.shutdown(final_split='val', metric_prefix='cv')
            elif getattr(self, 'evaluate_test', False):
                self.server.shutdown(final_split='test')
            else:
                self.server.shutdown(
                    final_split='val', metric_prefix='final')
        ray.shutdown()
    
    def shutdown(self):
        """
        Shut down server-side logging and Ray resources without final logging.
        """
        self.server.shutdown(log_results=False)
        ray.shutdown()
