from abc import ABC,abstractmethod
from .server_factory import register_server
from client import BaseClient

@register_server("BaseServer")
class BaseServer(ABC):
    """Abstract server interface for federated training algorithms."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        self.config:dict = kwargs['config']
        self.children_list:list = kwargs['children_list']
        assert isinstance(self.config,dict), "config must be a dictionary"
        assert isinstance(self.children_list,list), "children_list must be a list"
    
    @abstractmethod
    def setup(self,**kwargs):
        """Prepare.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass 
    
    @abstractmethod
    def step(self,**kwargs):
        """Execute one step.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass
    
    @abstractmethod
    def execute(self,**kwargs):
        """Execute.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass
    
    @abstractmethod
    def evaluate(self,**kwargs):
        """Evaluate.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        
        Returns:
            Requested result.
        """
        pass
    
    @abstractmethod
    def fine_tune(self,**kwargs):
        """Handle fine tune.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass
    
    @abstractmethod
    def shutdown(self,**kwargs):
        """Shut down.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        pass
