from abc import ABC, abstractmethod
class BaseMetric(ABC):
    """Abstract interface for metrics used by training and evaluation code."""
    @abstractmethod
    def calculate(self, y_pred, y_true):
        """Calculate the metric value.
        
        Args:
            y_pred: Predicted labels or model outputs.
            y_true: Ground-truth labels.
        """
        pass
    
    @abstractmethod
    def get(self):
        """Return the current value."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset."""
        pass
