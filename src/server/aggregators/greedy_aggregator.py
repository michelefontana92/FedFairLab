from .aggregator_factory import register_aggregator
from .base_aggregator import BaseAggregator
import torch.nn as nn
from ..utils import compute_global_score

@register_aggregator("GreedyAggregator")
class GreedyAggregator(BaseAggregator):
    def __init__(self,**kwargs):
        super(GreedyAggregator,self).__init__(**kwargs)
    
    def setup(self,**kwargs):
        pass
    
    def _compute_total_weight(self,params_list):
        return sum([params['weight'] for params in params_list])
        
    def __call__(self,**kwargs):
        pass