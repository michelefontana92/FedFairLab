from experiments import FedAvgExperiment
import shutil
from .compas_run import CentralizedCompasRun
from ..run_factory import register_run

@register_run('compas_fedavg')
class CompasFedAvgRun(CentralizedCompasRun):
    def __init__(self,**kwargs) -> None:
        super(CompasFedAvgRun, self).__init__(**kwargs)
        self.project_name = 'CompasFedAvgFairnessInjected'
        self.num_clients = 1
        self.lr=1e-4
        self.num_federated_rounds = 100
        self.project_name =  kwargs.get('project_name')
        self.start_index = kwargs.get('start_index',1)
    def setUp(self):
       
        self.experiment = FedAvgExperiment( sensitive_attributes=self.sensitive_attributes,
                                            dataset=self.dataset,
                                            data_root=self.data_root,
                                            model=self.model,
                                            num_clients=self.num_clients,
                                            num_federated_rounds=self.num_federated_rounds,
                                            lr=self.lr,
                                            
                                            project=self.project_name,
                                            start_index=self.start_index
                                            )

    def run(self):
        self.experiment.setup()
        self.experiment.run()

    def tearDown(self) -> None:
        shutil.rmtree(f'checkpoints/{self.project_name}')