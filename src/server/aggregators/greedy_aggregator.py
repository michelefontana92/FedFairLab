from .aggregator_factory import register_aggregator
from .base_aggregator import BaseAggregator
import torch.nn as nn
from ..utils import compute_global_score

@register_aggregator("GreedyAggregator")
class GreedyAggregator(BaseAggregator):
    """Implementation of GreedyAggregator."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super(GreedyAggregator,self).__init__(**kwargs)
    
    def setup(self,**kwargs):
        """Prepare.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass
    
    def _compute_total_weight(self,params_list):
        """Handle compute total weight.
        
        Args:
            params_list: Client parameter payloads to aggregate.
        """
        return sum([params['weight'] for params in params_list])
        
    def __call__(self,**kwargs):
        """Evaluate the callable object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass