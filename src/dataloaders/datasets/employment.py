import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset


@register_dataset('employment')
class EmploymentDataset(BaseDataset):

    def __init__(self,**kwargs):
        super(EmploymentDataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/Employment')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'employment_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'ESR'
        self.cat_cols = ['Gender','Race','Marital',
                         'SCHL', 'RELP', 'DIS', 'ESP', 
                         'CIT', 'MIG', 'MIL', 'ANC', 
                         'NATIVITY', 'DEAR', 'DEYE', 
                         'DREM']
        self.num_cols = ['AGEP']
        self.labels = [0,1]
        self.clean_data_path = os.path.join(self.root,'employment_clean.csv')
        self.setup()
        