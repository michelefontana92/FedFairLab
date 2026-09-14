from ..base_run import BaseRun
from architectures import ArchitectureFactory

class EducationRun(BaseRun):
    
    """Implementation of EducationRun."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super(EducationRun, self).__init__(**kwargs)
        self.num_classes=3
        self.input = 85
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
        self.dataset = 'education'
        self.data_file_prefix = 'education'
        self.data_root = self.resolve_experiment_data_root(
            kwargs, 'Education')
        self.clean_data_path = kwargs.get('clean_data_path') or f'{self.data_root}/education_clean.csv'
        
        self.sensitive_attributes = kwargs.get('sensitive_attributes',[
                                                
                                                ('Marital',{
                                                     'Marital':['Married','Never Married','Divorced','Other']}),   
                                                
                                                 ('Race',{'Race':['White','Black','Asian','Other','Indigenous']}),
                                                 ('Gender',{'Gender':['Male','Female']}),
                                                 
                                               
                                            
                                                
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

                                               
                                                    
                                                    ]
                                                )
        self.configure_validation_splits(
            kwargs, 'SCHL', ('Race', 'Marital'))
                                             
    def setUp(self):
        """Handle setUp."""
        pass
    def run(self):
        """Handle run."""
        pass
    def tearDown(self):
        """Handle tearDown."""
        pass
