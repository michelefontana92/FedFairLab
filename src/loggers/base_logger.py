from abc import ABC, abstractmethod

class BaseLogger(ABC):
    """Abstract logger interface used by experiment components."""
    @abstractmethod
    def log(self, message):
        """Log.
        
        Args:
            message: Message to log.
        """
        pass

    @abstractmethod
    def error(self, message):
        """Log an error.
        
        Args:
            message: Message to log.
        """
        pass

    @abstractmethod
    def info(self, message):
        """Log an informational message.
        
        Args:
            message: Message to log.
        """
        pass

    @abstractmethod
    def debug(self, message):
        """Log debug information.
        
        Args:
            message: Message to log.
        """
        pass
    
    @abstractmethod
    def close(self): 
        """Close."""
        pass
