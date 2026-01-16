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
import wandb
import pprint
class FedFairLabBuilder(Base_Builder):
    def _assign_resources(self):
        num_clients = self.num_clients
        self.num_cpus = 1*(num_clients) + 1
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
        self.common_client_params  = self._get_common_params(**kwargs)
        self.experiment_name = kwargs.get('experiment_name')
        self.clients = []
        self.algorithm = kwargs.get('algorithm', 'fedfairlab')
        for i in range(self.num_clients):
            client = self._build_client(f'{self.id}_client_{i+1}',i+1,**kwargs)
            self.clients.append(client)
        self.server = self._build_server(**kwargs)
        self.eval_mode = kwargs.get('eval_mode',False)
        self.eval_dir = kwargs.get('checkpoint_path')
        self.eval_prefix = kwargs.get('eval_prefix')
        
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
        common_params['num_classes'] = kwargs.get('num_classes')
        self.num_classes = common_params['num_classes']
        print('Number of classes:', self.num_classes)
        print('Groups: ', common_params['groups_list'])
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
        print()

        if common_params['performance_constraint'] < 1.0:
            print('Performance constraint: ', common_params['performance_constraint'])
            if common_params['num_classes'] > 2:
                common_params['inequality_constraints'] = [SurrogateFactory.create(name='multiclass_f1', 
                                    surrogate_name='cross_entropy', 
                                    weight=1, average='weighted', 
                                    upper_bound=common_params['performance_constraint'],
                                    use_max=True)]
            else:
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
            common_params['metrics'] += [MetricsFactory().create_metric(metric, group_ids=common_params['group_ids'], group_name=common_params['training_group_name'],
                                                                        num_classes=common_params['num_classes'],)]
           
        common_params['optimizer'] = Adam(copy.deepcopy(self.run_dict['model']).parameters(),
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
                                                 num_workers=4, 
                                                 use_local_weights = self.algorithm == 'fedavg_lr',
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
                model=copy.deepcopy(client_params['model']),
                inequality_constraints=client_params['inequality_constraints'],
                macro_constraints_list=client_params['macro_constraints_list'],
                optimizer_fn=client_params['optimizer_fn'],
                optimizer=client_params['optimizer'],
                objective_function=client_params['objective_function'],
                equality_constraints=client_params['equality_constraints'],
                metrics=client_params['metrics'],
                num_epochs=client_params['num_local_iterations'],
                loss = client_params['loss'],
                data_module=client_params['data_module'],
                
                lagrangian_checkpoints=client_params['lagrangian_callbacks'],
                checkpoints=client_params['callbacks'],
                #all_group_ids=client_params['all_group_ids'],
                checkpoints_config=client_params['checkpoints_config'],
                shared_macro_constraints=client_params['shared_macro_constraints'],
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
                    config = client_params
            )
        
        elif self.algorithm=='fedavg':
            fedavg_checkpoints_config = copy.deepcopy(checkpoints_config)
            fedavg_checkpoints_config['monitor'] = 'val_f1'
            fedavg_checkpoints_config['mode'] = 'max'
            client_params['client_callbacks'] = [
            ModelCheckpoint(save_dir=client_params['checkpoint_dir'], 
                            save_name=client_params['client_checkpoint_name'], 
                            monitor= fedavg_checkpoints_config['monitor'],
                            mode=fedavg_checkpoints_config['mode'])
        ]

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
                    client_callbacks = client_params['client_callbacks'],

            )
        elif self.algorithm=='fedavg_lr':
            fedavg_checkpoints_config = copy.deepcopy(checkpoints_config)
            fedavg_checkpoints_config['monitor'] = f"val_{client_params['metrics_list'][-1]}_{client_params['groups_list'][-1]}"
            fedavg_checkpoints_config['mode'] = 'min'
            client_params['client_callbacks'] = [
            ModelCheckpoint(save_dir=client_params['checkpoint_dir'], 
                            save_name=client_params['client_checkpoint_name'], 
                            monitor= fedavg_checkpoints_config['monitor'],
                            mode=fedavg_checkpoints_config['mode'])
        ]
            return partial(ClientFactory().create,
                    'client_fedavg_lr',
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
                    client_callbacks = client_params['client_callbacks'],
                    training_group_name = client_params['training_group_name'],
            )
        elif self.algorithm=='fedfb':
            fedavg_checkpoints_config = copy.deepcopy(checkpoints_config)
            fedavg_checkpoints_config['monitor'] = f"val_{client_params['metrics_list'][-1]}_{client_params['groups_list'][-1]}"
            fedavg_checkpoints_config['mode'] = 'min'
            client_params['client_callbacks'] = [
            ModelCheckpoint(save_dir=client_params['checkpoint_dir'], 
                            save_name=client_params['client_checkpoint_name'], 
                            monitor= fedavg_checkpoints_config['monitor'],
                            mode=fedavg_checkpoints_config['mode'])
        ]
            return partial(ClientFactory().create,
                    'client_fedfb',
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
                    client_callbacks = client_params['client_callbacks'],
                    training_group_name = client_params['training_group_name'],
            )
    

    def _build_server(self,**kwargs):
        server_params = copy.deepcopy(self.common_client_params)
        server_params['server_name'] = f'{self.id}_server'
        server_params['checkpoint_name'] = kwargs.get('checkpoint_name', f'{server_params["server_name"]}_global.h5')
        server_params['checkpoint_dir'] = kwargs.get('checkpoint_dir', f'checkpoints/{server_params["project_name"]}')
       
        server_params['model'] = copy.deepcopy(server_params['model'])
        server_params['metrics'] = kwargs.get('metrics')
        server_params['num_federated_iterations'] = kwargs.get('num_federated_iterations')
        server_params['num_classes'] =self.num_classes

        if self.algorithm == 'fedfairlab':
            return ServerFactory().create(
                    'server_fedfairlab',
                    remote=True,
                    num_gpus=1,
                    num_cpus=1,
                    clients_init_fn_list=self.clients,
                    **server_params)
        
        
        elif self.algorithm == 'fedavg':
            server_params['monitor'] = 'global_val_f1'
            server_params['mode'] = 'max'
            return ServerFactory().create(
                    'server_fedavg',
                    remote=True,
                    num_gpus=self.num_gpus,
                    clients_init_fn_list=self.clients,
                    **server_params)
        elif self.algorithm == 'fedavg_lr':
            server_params['monitor'] = f'global_val_{server_params["metrics_list"][-1]}_{server_params["groups_list"][-1]}'
            server_params['mode'] = 'min'
            return ServerFactory().create(
                    'server_fedavg',
                    remote=True,
                    num_gpus=self.num_gpus,
                    clients_init_fn_list=self.clients,
                    **server_params)
        elif self.algorithm == 'fedfb':
            server_params['monitor'] = f'global_val_{server_params["metrics_list"][-1]}_{server_params["groups_list"][-1]}'
            server_params['mode'] = 'min'
            return ServerFactory().create(
                    'server_fedfb',
                    remote=True,
                    num_gpus=self.num_gpus,
                    clients_init_fn_list=self.clients,
                    fair_metric = server_params["metrics_list"][-1],
                    **server_params)
        else:
            raise ValueError(f'Unknown algorithm: {self.algorithm}. Supported algorithms are fedfairlab and fedavg.')
    
    def run(self):
        if self.eval_mode:
            print('Entering EVAL MODE.')
            self.evaluate(checkpoint_dir=self.eval_dir,
                          prefix=self.eval_prefix)
        else:
            print('Number of CPUs:',self.num_cpus)
            print('Number of GPUs:',self.num_gpus)
            print('Number of GPUs per client:',self.num_gpus_per_client)
            ray.init(num_cpus=20,num_gpus=1)
            self.server.setup()
            self.server.execute()
            self.server.shutdown()
            ray.shutdown()
        
    

    def evaluate(self,checkpoint_dir,prefix='fedavg'):
        ray.init(num_cpus=20,num_gpus=1)
        self.server.setup()
        checkpoint_path = f'{checkpoint_dir}/{prefix}_server_global.h5'
        global_results = self.server.evaluate_model_from_ckpt(checkpoint_path=checkpoint_path,
                                             client_id=None
                                             )
        local_results = []
        for c in range(len(self.clients)):
            try:
                checkpoint_path = f'{checkpoint_dir}/{prefix}_client_{1+c}_local.h5'
                results = self.server.evaluate_model_from_ckpt(checkpoint_path=checkpoint_path,
                                                    client_id=c)
            except FileNotFoundError as e:
                checkpoint_path = f'{checkpoint_dir}/{prefix}_client_{1+c}_local_final.h5'
                results = self.server.evaluate_model_from_ckpt(checkpoint_path=checkpoint_path,
                                                    client_id=c)
            local_results.append(results)

        print('Results')
        print()
        print(50 * '-')
        print('Global Results:')
        pprint.pprint(global_results)
        print(50 * '-')
        print()
        for c in range(len(self.clients)):
            print()
            print(50 * '-')
            print(f'Local Results for Client {1+c}:')
            pprint.pprint(local_results[c])
            print(50 * '-')
            print()
        self.server.shutdown(log_results=False)
        ray.shutdown()
        

    def evaluate_old(self,checkpoint_path,run_id,client_id=None,init_fl =True,shutdown=False):
        project='Folk_Employment_New2'
        run = wandb.init(project=project, 
                         id=run_id, 
                         resume='must')
        
        #print("ID inizializzato:", run.id)
        #print("Stato della run:", run.)
        
        if init_fl:
            ray.init(num_cpus=self.num_cpus,num_gpus=self.num_gpus)
            self.server.setup()
        attributes = ['GenderRace', 'GenderMarital', 'RaceMarital', 'Gender', 'Race', 'Marital']
        attributes = [ 'RaceMarital', 'Race', 'Marital']
        metric_name = 'final_val_demographic_parity_'
       
        
        results = self.server.evaluate_model_from_ckpt(checkpoint_path=checkpoint_path,
                                                       log_results=False,
                                                       client_id=client_id)

        
        results_to_log = {}
        for attribute in attributes:
            results_to_log[f'{metric_name}{attribute}'] = results[f'{metric_name}{attribute}'].item()
        run.log(results_to_log)
        
    
        run.finish()
       
        return results_to_log
    
    def shutdown(self):
        self.server.shutdown(log_results=False)
        ray.shutdown()