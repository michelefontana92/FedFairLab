import os
from .dataset_factory import register_dataset
from .base_dataset import BaseDataset


@register_dataset('income')
class IncomeDataset(BaseDataset):

    """Binary Income dataset."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super().__init__(**kwargs)
        self.root = kwargs.get('root', 'data/10_Clients/Income')
        data_name = kwargs['filename']

        self.data_path = os.path.join(self.root, data_name)
        
        self.scaler_name = kwargs.get('scaler_name', 
                                      'income_scalers.p')
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                                [{}])
       
        self.scaler_path = f'{self.root}/{self.scaler_name}'
        
        self.target = 'PINCP'
        self.cat_cols = ['Gender','Race','Job','Marital']
        self.num_cols = []
        self.labels = [0,1]
        self.clean_data_path = kwargs.get(
            'clean_data_path', os.path.join(self.root, 'income_clean.csv'))
        self.setup()
        
