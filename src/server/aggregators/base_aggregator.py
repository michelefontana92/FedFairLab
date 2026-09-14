from abc import ABC, abstractmethod
class BaseAggregator(ABC):
    """Base interface for server-side aggregation strategies."""
    @abstractmethod
    def __call__(self,**kwargs):
        """Evaluate the callable object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass
