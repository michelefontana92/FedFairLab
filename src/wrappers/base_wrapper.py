# Define the BaseWrapper abstract class, which is a wrapper for a black box ML model
from abc import ABC, abstractmethod
class BaseWrapper(ABC):
    """Abstract interface for model wrappers used by clients and orchestrators."""
    
    @abstractmethod
    def predict(self, data_loader):
        """Predict labels.
        
        Args:
            data_loader: Data loader used for inference or evaluation.
        
        Returns:
            Requested result.
        """
        pass

    @abstractmethod
    def predict_proba(self, data_loader):
        """Predict class probabilities.
        
        Args:
            data_loader: Data loader used for inference or evaluation.
        
        Returns:
            Requested result.
        """
        pass
    
    @abstractmethod
    def score(self, data_loader,metrics):
        """Compute a score.
        
        Args:
            data_loader: Data loader used for inference or evaluation.
            metrics: Metric objects or metric dictionary.
        
        Returns:
            Requested result.
        """
        pass

    @abstractmethod    
    def save(self, path):
        """Save.
        
        Args:
            path: Filesystem path to read from or write to.
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """Load.
        
        Args:
            path: Filesystem path to read from or write to.
        """
        pass

   

    
