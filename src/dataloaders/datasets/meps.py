import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset

@register_dataset('meps')
class MEPSDataset(BaseDataset):

    """Medical Expenditure Panel Survey dataset."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super().__init__(**kwargs)
        self.root = kwargs.get('root', 'data/10_Clients/MEPS')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'meps_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        self.clean_data_path = kwargs.get(
            'clean_data_path', os.path.join(self.root, 'meps_clean.csv'))
        self.target = 'HIGH_EXPENSES'
        self.cat_cols = [
            'RACE',
            'SEX',
            'MARRY',
            'AGE_CAT',
            'REGION',
            'FTSTU',
            'ACTDTY',
            'HONRDC',
            'RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','MIDX',
            'OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX',
            'DIABDX','JTPAIN','ARTHDX','ARTHTYPE','ASTHDX',
            'ADHDADDX','PREGNT','WLKLIM','ACTLIM','SOCLIM',
            'COGLIM','DFHEAR42','DFSEE42','ADSMOK42','EMPST',
            'POVCAT','INSCOV'
        ]
        self.num_cols = [
            'AGE','PCS42','MCS42','K6SUM42','PHQ242','PERWT15F'
        ]
        self.labels = [0,1]
        
        self.setup()
        
        
        
        
    

    
