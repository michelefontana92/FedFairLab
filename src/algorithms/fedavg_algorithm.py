from .base_algorithm import BaseAlgorithm
from .algorithm_factory import register_algorithm
from server.server_factory import ServerFactory
from client.client_factory import ClientFactory
from client.client_fedavg import ClientFedAvg
import ray
from functools import partial

@register_algorithm("FedAvg")
class FedAvgAlgorithm(BaseAlgorithm):
    def __init__(self,**kwargs):
     
        self.project = kwargs.get('project','FedAvg')
        self.server_config = kwargs.get('server_config')
        self.clients_config = kwargs.get('clients_config')
        self.algorithm_config = kwargs.get('algorithm_config')
        
        assert isinstance(self.server_config,dict), "server_config must be a dictionary"
        assert isinstance(self.clients_config,list), "clients_config must be a list"
        assert isinstance(self.algorithm_config,dict), "algorithm_config must be a dictionary"

       
        self.model = self.algorithm_config.get('model')
        self.loss = self.algorithm_config.get('loss')
        self.metrics = self.algorithm_config.get('metrics')
        self.optimizer_name = self.algorithm_config.get('optimizer_name','Adam')
        self.lr = self.algorithm_config.get('lr',1e-3)
        
        self.num_local_epochs = self.algorithm_config.get('num_local_epochs',3)
        self.num_federated_epochs = self.algorithm_config.get('num_federated_epochs',2)
        self.verbose = self.algorithm_config.get('verbose',False)
        
        self.clients = []
        
        for i,client_config in enumerate(self.clients_config):
            c_id = client_config.get('id',f'fedavg_client_{i+1}')
            c_data = client_config.get('data')
            c_ckpt_dir = client_config.get('checkpoint_dir',f'checkpoints/{self.project}')
            c_ckpt_name = client_config.get('checkpoint_name',f'fedavg_model_{c_id}.h5')
            c_config = {
                'early_stopping_patience': client_config.get('early_stopping_patience',5),
                'monitor': client_config.get('monitor','val_f1'),
                'mode': client_config.get('mode','max'),
                'optimizer_name': self.optimizer_name,
                'lr': self.lr,
                'num_epochs':self.num_local_epochs
            }


            self.clients.append(partial(ClientFactory().create,
                'client_fedavg',
                remote=True,
                config=c_config,
                id = c_id,
                project = self.project,
                model = self.model,
                data = c_data,
                loss = self.loss,
                metrics = self.metrics,
                log_model = client_config.get('log_model',False),
                checkpoint_dir = c_ckpt_dir,
                checkpoint_name = c_ckpt_name
                )
                )
            
            
        s_config = {
            'early_stopping_patience': self.server_config.get('early_stopping_patience',5),
            'monitor': self.server_config.get('monitor','global_val_f1'),
            'mode': self.server_config.get('mode','max'),
            'num_rounds':self.num_federated_epochs
        }

        s_id = self.server_config.get('id','fedavg_server')
        s_ckpt_dir = self.server_config.get('checkpoint_dir',f'checkpoints/{self.project}')
        s_ckpt_name = self.server_config.get('checkpoint_name','fedavg_global_model.h5')
        
        self.server = ServerFactory().create('server_fedavg',
        config=s_config,
        id = s_id,
        project = self.project,
        model = self.model,
        metrics = self.metrics,
        loss = self.loss,
        clients_init_fn_list = self.clients,
        log_model = self.server_config.get('log_model',False),
        checkpoint_dir = s_ckpt_dir,
        checkpoint_name = s_ckpt_name,
        verbose = self.verbose,
        )

    def __call__(self,**kwargs):
        ray.init(num_cpus=len(self.clients_config)+1)
        self.server.setup()
        self.server.execute()
        self.server.shutdown()
        ray.shutdown()