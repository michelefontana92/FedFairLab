from abc import ABC,abstractmethod



class BaseClient(ABC):
    """
    Abstract interface implemented by all federated clients.

    Concrete clients may run locally or as Ray actors, but they expose the same
    lifecycle: setup, local update, evaluation, optional fine tuning, and shutdown.
    The server relies on this interface when broadcasting work across the federation.
    """
    def __init__(self,**kwargs):
        """
        Store the client configuration.

        Args:
            **kwargs: Must contain ``config``, a dictionary with experiment,
                model, data, optimization, logging, and checkpoint settings.
        """
        self.config:dict = kwargs['config']
        assert isinstance(self.config,dict), "config must be a dictionary"
        
    @abstractmethod
    def update(self,**kwargs):
        """
        Update the client state or model.

        Args:
            **kwargs: Implementation-specific update payload.
        """
        pass
    
    @abstractmethod
    def setup(self,**kwargs):
        """
        Prepare the client before federated execution starts.

        Args:
            **kwargs: Optional runtime setup values supplied by the server.
        """
        pass 
    
    @abstractmethod
    def evaluate(self,**kwargs):
        """
        Evaluate a model on the client's local data.

        Args:
            **kwargs: Evaluation payload, typically including model parameters
                and the problem definition.
        """
        pass
    
    @abstractmethod
    def fine_tune(self,**kwargs):
        """
        Optionally adapt a global model to the local client distribution.

        Args:
            **kwargs: Fine-tuning options supplied by the server.
        """
        pass
    
    @abstractmethod
    def shutdown(self,**kwargs):
        """
        Release resources and flush logs/artifacts.

        Args:
            **kwargs: Shutdown options such as whether final metrics should be logged.
        """
        pass
