import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset

@register_dataset('compas')
class CompasDataset(BaseDataset):

    """Implementation of CompasDataset."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super(CompasDataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/10_Clients/Compas')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'compas_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        self.clean_data_path = kwargs.get('clean_data_path', 
                                          os.path.join(self.root, 'fake_compas.csv'))
        self.target = 'two_year_recid'
        self.cat_cols = [
            'c_charge_degree',
            'age_cat',
            'score_text',
            'decile_score',
            'sex',
            'race'
        ]
        # Keep the federated feature schema stable even when a client's
        # training fold does not contain every possible category. These are
        # dataset-domain categories, not values learned from validation/test.
        self.categorical_categories = {
            'c_charge_degree': ('F', 'M'),
            'age_cat': ('25 - 45', 'Greater than 45', 'Less than 25'),
            'score_text': ('High', 'Low', 'Medium'),
            'decile_score': tuple(range(1, 11)),
            'sex': ('Female', 'Male'),
            'race': (
                'African-American', 'Asian', 'Caucasian', 'Hispanic',
                'Native American', 'Other'),
        }
        self.num_cols = [
            'age',
            'priors_count',
            'days_b_screening_arrest',
            'length_of_stay',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
            'c_days_from_compas'
        ]
        self.labels = [0,1]
        
        self.setup()
        
        
        
        
    

    
