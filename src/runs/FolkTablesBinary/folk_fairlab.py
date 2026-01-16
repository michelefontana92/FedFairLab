import shutil
from .folk_run import FolkTablesBinaryRun
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
@register_run('folk_fedfairlab')
class FolkTablesBinaryHierALMCentralized(FolkTablesBinaryRun):
    def __init__(self, **kwargs) -> None:
        super(FolkTablesBinaryHierALMCentralized, self).__init__(**kwargs)
        kwargs['run_dict'] = self.to_dict()
        self.builder = FedFairLabBuilder(**kwargs)
    
    def setUp(self):
        #print(self.builder.clients)
        pass


    def run(self):
        self.builder.run()

    def tearDown(self) -> None:
        # Pulizia finale dei file di checkpoint, se necessario
        pass
        #shutil.rmtree(f'checkpoints/{self.project_name}')
