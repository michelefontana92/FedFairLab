from abc import ABC, abstractmethod
class BaseRun(ABC):
    def __init__(self,**kwargs):
        
        self.model = kwargs.get('model')
        self.dataset = kwargs.get('dataset')
        self.sensitive_attributes = kwargs.get('sensitive_attributes')
        self.project_name = kwargs.get('project_name')
        self.data_root = kwargs.get('data_root')

    def compute_group_cardinality(self,group_name):
        for name,group_dict in self.sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 
    
    @abstractmethod
    def setUp(self):
        pass 
    
    @abstractmethod
    def tearDown(self):
        pass

    @abstractmethod
    def run(self,**kwargs):
        pass
    
    @abstractmethod
    def eval(self):
        pass
    def __call__(self, **kwargs):
        self.setUp()
        self.run(**kwargs)
        self.tearDown()

    def build_server_config(self,**kwargs):
        server_config = {
            'early_stopping_patience':5,
            'monitor':'global_val_requirements',
            'mode':'min',
        }
        return server_config
    

    def to_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'sensitive_attributes': self.sensitive_attributes,
            'project_name': self.project_name,
            'data_root': self.data_root,
            'hidden1': self.hidden1,
            'hidden2': self.hidden2,
            'input': self.input,
            'dropout': self.dropout,
            'num_classes': self.num_classes,
            'output': self.output,
            'server_config': self.build_server_config(),
        }