import shutil
from .education_run import EducationRun
from ..run_factory import register_run
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
from builder import FedFairLabBuilder
import wandb

@register_run('education_fedfairlab')
class EducationHierALMCentralized(EducationRun):
    def __init__(self, **kwargs) -> None:
        super(EducationHierALMCentralized, self).__init__(**kwargs)
        kwargs['run_dict'] = self.to_dict()
        self.builder = FedFairLabBuilder(**kwargs)
    
    def setUp(self):
        #print(self.builder.clients)
        pass
    
    

    def run(self):
        self.builder.run()
        
        """
        run_ids = ['3u7ijuuz','bgw23ovg','cfwxwtqu','x8jzfpf3','b19oh8t0','cegswopz','93ct7fhl','01t6xqp1','i8z6jdqb']
        run_ids = ['01t6xqp1']
        for i,run_id in zip(range(9,10),run_ids):
            init_fl = i==9
            print(f'Running client {i} with run_id {run_id}')
            self.builder.evaluate(f'checkpoints/FedFairLab_Folk_New/fedfairlab_performance_client_{i}_local.h5',
                              run_id,
                              client_id=i-1,
                              init_fl=init_fl,
                              shutdown=False,
                              )
           
        #self.builder.shutdown()
        """
    def tearDown(self) -> None:
        # Pulizia finale dei file di checkpoint, se necessario
        pass
        #shutil.rmtree(f'checkpoints/{self.project_name}')
