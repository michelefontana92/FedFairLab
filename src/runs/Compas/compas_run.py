from ..base_run import BaseRun
from architectures import ArchitectureFactory

class CompasRun(BaseRun):
    
    """Implementation of CompasRun."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super(CompasRun, self).__init__(**kwargs)
        self.input = 34
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
        self.dataset = 'compas'
        self.data_file_prefix = 'compas'
        self.data_root = self.resolve_experiment_data_root(kwargs, 'Compas')
        self.clean_data_path = kwargs.get('clean_data_path') or f'{self.data_root}/compas_clean.csv'
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {'race':['Other', 'African-American', 
                                                             'Caucasian'
                                                             ]}
                                                ),
                                                ('Gender',{'sex':['Male','Female']}),
                                                ('Age',{'age_cat':['Greater than 45','25 - 45','Less than 25']}),
                                                ('GenderRace',{
                                                    'race':['Other', 'African-American', 
                                                             'Caucasian'  
                                                             ],
                                                    'sex':['Male','Female']
                                                }),
                                                ('GenderAge',{
                                                    'age_cat':['Greater than 45','25 - 45','Less than 25'],
                                                    'sex':['Male','Female']
                                                }),
                                                ('RaceAge',{
                                                    'age_cat':['Greater than 45','25 - 45','Less than 25'],
                                                    'race':['Other', 'African-American', 
                                                             'Caucasian' ]
                                                }),
                                                ('GenderRaceAge',{
                                                    'age_cat':['Greater than 45','25 - 45','Less than 25'],
                                                    'race':['Other', 'African-American', 
                                                             'Caucasian'  
                                                             ],
                                                    'sex':['Male','Female']
                                                }),
                                                ])
        self.configure_validation_splits(
            kwargs, 'two_year_recid', ('race', 'age_cat'))
        
        
    def setUp(self):
        """Handle setUp."""
        pass
    def run(self):
        """Handle run."""
        pass
    def tearDown(self):
        """Handle tearDown."""
        pass
