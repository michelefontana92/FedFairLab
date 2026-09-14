from ..base_run import BaseRun
from architectures import ArchitectureFactory

class MEPSRun(BaseRun):
    
    """Dataset and model configuration for MEPS."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super().__init__(**kwargs)
        self.input = 132
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.num_classes=2
        self.output = self.num_classes
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': self.input,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'dropout': self.dropout,
                                                'output': self.output})
        self.dataset = 'meps'
        self.data_file_prefix = 'meps'
        self.data_root = self.resolve_experiment_data_root(kwargs, 'MEPS')
        self.clean_data_path = kwargs.get('clean_data_path') or f'{self.data_root}/meps_clean.csv'
        race_values = ['Hispanic','Black','White','Other']
        marry_values = ['Married','Never Married','Other']
        gender_values = ['Male','Female']
        race_var = 'RACE'
        gender_var = 'SEX'
        marry_var = 'MARRY'
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {race_var:race_values}
                                                ),
                                                ('Gender',
                                                 {gender_var:gender_values}
                                                 ),
                                                 ('Marital',
                                                  {marry_var:marry_values}),
                                                ('GenderRace',
                                                 {gender_var:gender_values,
                                                  race_var:race_values}),
                                                ('GenderMarital',
                                                 {gender_var:gender_values,
                                                  marry_var:marry_values}),
                                                ('RaceMarital',
                                                 {marry_var:marry_values,
                                                  race_var:race_values}),
                                                ('GenderRaceMarital',
                                                 {gender_var:gender_values,
                                                  marry_var:marry_values,
                                                  race_var:race_values})
                                                ])
        self.configure_validation_splits(
            kwargs, 'HIGH_EXPENSES', ('RACE', 'MARRY'))
        
        

    def setUp(self):
        """Handle setUp."""
        pass
    def run(self):
        """Handle run."""
        pass
    def tearDown(self):
        """Handle tearDown."""
        pass
