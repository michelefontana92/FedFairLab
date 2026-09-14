import torch.nn as nn
from .metrics_factory import register_metric

def _cross_entropy_loss(**kwargs):
    """Handle cross entropy loss.
    
    Args:
        **kwargs: Additional options forwarded to the implementation.
    """
    return nn.CrossEntropyLoss()

def _mse_loss(**kwargs):
    """Handle mse loss.
    
    Args:
        **kwargs: Additional options forwarded to the implementation.
    """
    return nn.MSELoss()

def _binary_crossentropy_loss(**kwargs):
    """Handle binary crossentropy loss.
    
    Args:
        **kwargs: Additional options forwarded to the implementation.
    """
    return nn.BCELoss()

def _weighted_cross_entropy_loss(**kwargs):
    """Handle weighted cross entropy loss.
    
    Args:
        **kwargs: Additional options forwarded to the implementation.
    """
    return nn.CrossEntropyLoss(weight=kwargs.get('weight'))

@register_metric('loss')
class Loss:
    """Implementation of Loss."""
    def __init__(self, name):
        """Initialize the object.
        
        Args:
            name: Registered object name.
        """
        self.name = name
        self.loss_fn = None
        self._losses = {
            'cross_entropy': _cross_entropy_loss,
            'mse': _mse_loss,
            'binary_crossentropy': _binary_crossentropy_loss,
            'weighted_cross_entropy': _weighted_cross_entropy_loss
        }
        self._set_loss_fn()

    def _set_loss_fn(self):
        """Handle set loss fn."""
        self.loss_fn = self._losses.get(self.name)

    def __call__(self, **kwargs):
        """Evaluate the callable object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        return self.loss_fn(**kwargs)