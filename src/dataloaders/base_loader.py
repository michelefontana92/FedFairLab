from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    """Abstract data-module interface exposing train, validation and test loaders."""
    @abstractmethod
    def train_loader(self):
        """Return the training data loader.
        
        Returns:
            Requested result.
        """
        pass 

    @abstractmethod
    def val_loader(self):
        """Return the validation data loader.
        
        Returns:
            Requested result.
        """
        pass 
    
    @abstractmethod
    def test_loader(self):
        """Return the test data loader.
        
        Returns:
            Requested result.
        """
        pass 
    
