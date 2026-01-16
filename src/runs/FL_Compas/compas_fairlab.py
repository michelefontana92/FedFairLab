import shutil
from .compas_run import CompasRun
from ..run_factory import register_run
from builder import FedFairLabBuilder


@register_run('compas_fedfairlab')
class FLCompasHierALMCentralized(CompasRun):
    def __init__(self, **kwargs) -> None:
        super(FLCompasHierALMCentralized, self).__init__(**kwargs)
        kwargs['run_dict'] = self.to_dict()
        self.builder = FedFairLabBuilder(**kwargs)
    
    def setUp(self):
        #print(self.builder.clients)
        pass
    
    def run(self):
        self.builder.run()

    def eval(self):
        pass      
    def tearDown(self) -> None:
        # Pulizia finale dei file di checkpoint, se necessario
        #pass
        shutil.rmtree(f'checkpoints/{self.project_name}')
