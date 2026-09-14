from ..base_run import BaseRun
from architectures import ArchitectureFactory

class IncomeRun(BaseRun):
    
    """Dataset and model configuration for binary Income prediction."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super().__init__(**kwargs)
        self.num_classes=2
        self.input = 20
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.output = self.num_classes
    
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': self.input,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'dropout': self.dropout,
                                                'output': self.output})
        self.dataset = 'income'
        self.data_file_prefix = 'income'
        self.data_root = self.resolve_experiment_data_root(kwargs, 'Income')
        self.clean_data_path = kwargs.get('clean_data_path') or f'{self.data_root}/income_clean.csv'
        
        self.sensitive_attributes = kwargs.get('sensitive_attributes',[
                                                
                                                ('Marital',{
                                                     'Marital':['Married','Never Married','Divorced','Other']}),   
                                                
                                                ('Job',{'Job':['Public Employee','Self Employed','Private Employee']}),
                                                 ('Race',{'Race':['White','Black','Asian','Other','Indigenous']}),
                                                 ('Gender',{'Gender':['Male','Female']}),
                                                 
                                                ('JobMarital',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                
                                                
                                                ('JobRace',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Race':['White','Black','Asian','Other','Indigenous']
                                                    }),
                                                
                                                ('GenderRace',{
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Gender':['Male','Female'],
                                                    }),
                                                ('RaceMarital',{
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                
                                                ('GenderMarital',{
                                                    'Gender':['Male','Female'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                
                                                ('GenderRaceMarital',{
                                                    'Gender':['Male','Female'],
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),

                                                ('JobRaceMarital',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                
                                                
                                                
                                                ('JobMaritalGender',{
                                                     'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    'Gender':['Male','Female']
                                                     }),
                                                ('JobRaceMaritalGender',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    'Gender':['Male','Female']
                                                    }),
                                                    
                                                    ]
                                                )
        self.configure_validation_splits(
            kwargs, 'PINCP', ('Race', 'Marital'))
                                             
    def setUp(self):
        """Handle setUp."""
        pass
    def run(self):
        """Handle run."""
        pass
    def tearDown(self):
        """Handle tearDown."""
        pass
