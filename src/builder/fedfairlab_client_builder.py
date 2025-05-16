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
import torch
import copy
from client import ClientFactory
from server import ServerFactory
import ray


class FedFairLabBuilder(Base_Builder):
    def _assign_resources(self):
        num_clients = self.num_clients
        self.num_cpus = num_clients + 1
        self.num_gpus = len(self.gpu_devices)
     
        self.num_gpus_per_client = self.num_gpus/num_clients if self.num_gpus > 0 else 0
        
    def compute_group_cardinality(self,group_name,sensitive_attributes):
        for name,group_dict in sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 
    
    def __init__(self,**kwargs):
        super(FedFairLabBuilder,self).__init__(**kwargs)
        self.num_clients = kwargs.get('num_clients', 1)
        self.gpu_devices = kwargs.get('gpu_devices',[0])
        self._assign_resources()
        self.id = kwargs.get('id')
        self.run_dict = kwargs.get('run_dict')
        self.common_client_params = self._get_common_params(**kwargs)
        self.experiment_name = kwargs.get('experiment_name')
        self.clients = []
        self.algorithm = kwargs.get('algorithm', 'fedfairlab')
        for i in range(self.num_clients):
            client = self._build_client(f'{self.id}_client_{i+1}',i+1,**kwargs)
            self.clients.append(client)
        self.server = self._build_server(**kwargs)

    def _get_common_params(self,**kwargs):
        common_params = {}
        common_params['metrics_list'] = kwargs.get('metrics_list')
        common_params['groups_list'] = kwargs.get('groups_list')
        common_params['threshold_list'] = kwargs.get('threshold_list')
        common_params['lr'] = kwargs.get('lr', 1e-4)
        common_params['loss'] = partial(CrossEntropyLoss)
        common_params['num_lagrangian_epochs'] = kwargs.get('num_lagrangian_epochs', 1)
        common_params['batch_size'] = kwargs.get('batch_size', 128)
        common_params['project_name'] = kwargs.get('project_name')
        common_params['checkpoint_dir'] = kwargs.get('checkpoint_dir', f'checkpoints/{common_params["project_name"]}')
        
        common_params['verbose'] = kwargs.get('verbose', False)
        common_params['optimizer_fn'] = partial(Adam, lr=common_params['lr'])
        
        common_params['monitor'] = kwargs.get('monitor', 'val_constraints_score')
        common_params['mode'] = kwargs.get('mode', 'max')
        
        
        common_params['log_model'] = kwargs.get('log_model', False)
        common_params['num_global_iterations'] = kwargs.get('num_global_iterations')
        common_params['num_local_iterations'] = kwargs.get('num_local_iterations')
        common_params['num_personalization_iterations'] = kwargs.get('num_personalization_iterations')
        
        common_params['performance_constraint'] = kwargs.get('performance_constraint')
        common_params['delta'] = kwargs.get('delta', 0.2)
        common_params['max_constraints_in_subproblem'] = kwargs.get('max_constraints_in_subproblem')
        common_params['global_patience'] = kwargs.get('global_patience')
        common_params['local_patience'] = kwargs.get('local_patience')
        print('Groups: ', common_params['groups_list'])
        # Callbacks
        
        # Metriche
        common_params['metrics'] = [MetricsFactory().create_metric('performance')]

        # Funzione obiettivo e vincoli
        common_params['objective_function'] = SurrogateFactory.create(name='performance', surrogate_name='cross_entropy', weight=1, average='weighted')
        common_params['batch_objective_function'] = SurrogateFactory.create(name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted')
        common_params['original_objective_fn'] = SurrogateFactory.create(name='binary_f1', surrogate_name='binary_f1', weight=1, average='weighted')
        common_params['equality_constraints'] = []
        common_params['shared_macro_constraints'] = []
        print()

        if common_params['performance_constraint'] < 1.0:
            print('Performance constraint: ', common_params['performance_constraint'])
            common_params['inequality_constraints'] = [SurrogateFactory.create(name='binary_f1', 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=common_params['performance_constraint'],
                                    use_max=True)]
            common_params['lagrangian_callbacks'] = [EarlyStopping(patience=2, 
                                                                   monitor='score', 
                                                                   mode='max')]
            common_params['macro_constraints_list'] = [[0]]
            common_params['shared_macro_constraints'] = [0]
          
        else:
            print('No performance constraint')
            print()
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
            common_params['metrics'] += [MetricsFactory().create_metric(metric, group_ids=common_params['group_ids'], group_name=common_params['training_group_name'])]
           
        common_params['optimizer'] = Adam(self.run_dict['model'].parameters(),
                          lr=common_params['lr']
                          )
    
        
        return common_params

    
    def _build_client(self,client_name,client_idx,**kwargs):
        client_params = copy.deepcopy(self.common_client_params)
        client_params['client_name'] = client_name
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
            'dataset': client_params['dataset'],
            'optimizer': 'Adam',
            'num_lagrangian_epochs': client_params['num_lagrangian_epochs'],
            'num_epochs': client_params['num_local_iterations'],
            'patience': client_params['global_patience'],
            'monitor': client_params['monitor'],
            'mode': client_params['mode'],
            'log_model': client_params['log_model']
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
        path = f'{self.experiment_name}/node_{client_idx}/{client_params["dataset"]}' 
        client_params['data_module'] = DataModule(dataset=client_params["dataset"], 
                                               root=client_params["data_root"], 
                                               train_set=f'{path}_train.csv',
                                                 val_set=f'{path}_val.csv', 
                                                 test_set=f'{path}_val.csv', 
                                                 batch_size=client_params["batch_size"], 
                                                 num_workers=0, 
                                                 sensitive_attributes=client_params["sensitive_attributes"])

        # Configurazione del logger
        client_params['logger'] = partial(WandbLogger,
                                          project=client_params["project_name"], 
                                  config=config, 
                                  id=client_name,
                                  checkpoint_dir=client_params["checkpoint_dir"], 
                                  checkpoint_path=client_params["checkpoint_name"],
                                  data_module=client_params["data_module"] if client_params["log_model"] else None
                                  )

        if self.algorithm == 'fedfairlab':
            orchestrator = partial(OrchestratorWrapper,
                model=client_params['model'],
                #inequality_constraints=client_params['inequality_constraints'],
                #macro_constraints_list=client_params['macro_constraints_list'],
                optimizer_fn=client_params['optimizer_fn'],
                optimizer=client_params['optimizer'],
                #objective_function=client_params['objective_function'],
                equality_constraints=client_params['equality_constraints'],
                metrics=client_params['metrics'],
                num_epochs=client_params['num_local_iterations'],
                loss = client_params['loss'],
                data_module=client_params['data_module'],
                
                lagrangian_checkpoints=client_params['lagrangian_callbacks'],
                checkpoints=client_params['callbacks'],
                #all_group_ids=client_params['all_group_ids'],
                checkpoints_config=client_params['checkpoints_config'],
                #shared_macro_constraints=client_params['shared_macro_constraints'],
                delta=client_params['delta'],
                max_constraints_in_subproblem=client_params['max_constraints_in_subproblem'],
                #batch_objective_fn=client_params['batch_objective_function'],
            )
            

            return partial(ClientFactory().create,
                    'client_fedfairlab',
                    remote=True,
                    num_gpus=self.num_gpus_per_client,
                    orchestrator=orchestrator,
                    client_name=client_name,
                    logger = client_params['logger'],
                    model = client_params['model'],
                    num_global_iterations = client_params['num_global_iterations'],
                    num_local_iterations = client_params['num_local_iterations'],
                    client_callbacks = client_params['client_callbacks'],
                    num_personalization_iterations = client_params['num_personalization_iterations'],
                    #config = client_params
            )
        elif self.algorithm=='fedavg':
            fedavg_checkpoints_config = copy.deepcopy(checkpoints_config)
            fedavg_checkpoints_config['monitor'] = 'val_f1'
            fedavg_checkpoints_config['mode'] = 'max'
            return partial(ClientFactory().create,
                    'client_fedavg',
                    remote=True,
                    num_gpus=self.num_gpus_per_client,
                    client_name=client_name,
                    logger = client_params['logger'],
                    model = client_params['model'],
                    metrics=client_params['metrics'],
                    num_epochs=client_params['num_local_iterations'],
                    loss = client_params['loss'],
                    data=client_params['data_module'],
                    config = client_params,
                    checkpoint_dir = client_params['checkpoint_dir'],
                    checkpoint_name = client_params['checkpoint_name'],
                    checkpoint_config = fedavg_checkpoints_config,
                    optimizer_name = 'Adam',

            )
        
    

    def _build_server(self,**kwargs):
        server_params = copy.deepcopy(self.common_client_params)
        server_params['server_name'] = f'{self.id}_server'
        server_params['checkpoint_name'] = kwargs.get('checkpoint_name', f'{server_params["server_name"]}_global.h5')
        server_params['checkpoint_dir'] = kwargs.get('checkpoint_dir', f'checkpoints/{server_params["project_name"]}')
        server_params['monitor'] = 'global_val_f1'
        server_params['mode'] = 'max'
        server_params['model'] = server_params['model']
        server_params['metrics'] = kwargs.get('metrics')
        server_params['num_federated_iterations'] = kwargs.get('num_federated_iterations')
        
        if self.algorithm == 'fedfairlab':
            return ServerFactory().create(
                    'server_fedfairlab',
                    remote=True,
                    num_gpus=self.num_gpus,
                    clients_init_fn_list=self.clients,
                    **server_params)
        
        
        elif self.algorithm == 'fedavg':
            return ServerFactory().create(
                    'server_fedavg',
                    remote=True,
                    num_gpus=self.num_gpus,
                    clients_init_fn_list=self.clients,
                    **server_params)
        
    def run(self):
        print('Number of CPUs:',self.num_cpus)
        print('Number of GPUs:',self.num_gpus)
        print('Number of GPUs per client:',self.num_gpus_per_client)
        ray.init(num_cpus=self.num_cpus,num_gpus=self.num_gpus)
        self.server.setup()
        self.server.execute(num_federated_rounds=2)
        self.server.shutdown()
        ray.shutdown()
     
       
    
    