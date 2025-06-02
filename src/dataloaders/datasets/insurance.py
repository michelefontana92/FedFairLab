import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset


@register_dataset('insurance')
class InsuranceDataset(BaseDataset):

    def __init__(self,**kwargs):
        super(InsuranceDataset, self).__init__(**kwargs)
        self.root = kwargs.get('root', 'data/Insurance')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'insurance_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'HINS2'
        self.cat_cols = ['Gender','Race','Marital',
                         'SCHL', 'DIS', 'ESP', 'CIT', 
                         'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 
                         'DEYE', 'DREM', 'ESR', 'FER']
        self.num_cols = ['AGEP','PINCP']
        self.labels = [0,1]
        self.clean_data_path = os.path.join(self.root,'insurance_clean.csv')
        self.setup()
        