import shutil
from .compas_run import CentralizedCompasRun
from ..run_factory import register_run
from builder import FedFairLabBuilder


@register_run('compas_fairlab')
class CompasHierALMCentralized(CentralizedCompasRun):
    def __init__(self, **kwargs) -> None:
        super(CompasHierALMCentralized, self).__init__(**kwargs)
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
