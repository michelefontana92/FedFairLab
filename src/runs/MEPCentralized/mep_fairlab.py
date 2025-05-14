import shutil
from .mep_run import CentralizedMEPRun
from ..run_factory import register_run
from builder import FedFairLabBuilder

@register_run('mep_fedfairlab')
class MEPHierALMCentralized(CentralizedMEPRun):
    def __init__(self, **kwargs) -> None:
        super(MEPHierALMCentralized, self).__init__(**kwargs)
        kwargs['run_dict'] = self.to_dict()
        self.builder = FedFairLabBuilder(**kwargs)
    
    def setUp(self):
        #print(self.builder.clients)
        pass
    
    def run(self):
        self.builder.run()
          
    def tearDown(self) -> None:
        # Pulizia finale dei file di checkpoint, se necessario
        #pass
        shutil.rmtree(f'checkpoints/{self.project_name}')
