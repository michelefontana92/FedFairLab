import torch
_ARCHITECTURES ={}

def register_architecture(architecture):
    """
    A decorator to register a new architecture class in the global _ARCHITECTURES dictionary.

    Args:
        architecture (str): The name of the architecture to register.

    Returns:
        function: A decorator function that registers the given class.

    Raises:
        ValueError: If the architecture name is already registered or if the class does not extend torch.nn.Module.

    Example:
        @register_architecture('my_architecture')
        class MyArchitecture(torch.nn.Module):
            ...
    """
    def decorator(cls):
        """Register the decorated class or function and return it unchanged.
        
        Args:
            cls: Class being registered.
        """
        if architecture in _ARCHITECTURES:
            raise ValueError(f"Cannot register duplicate architecture ({architecture})")
        if not issubclass(cls, torch.nn.Module):
            raise ValueError(f"architecture ({architecture}: {cls.__name__}) must extend torch.nn.Module")
        _ARCHITECTURES[architecture] = cls
        return cls
    return decorator

class ArchitectureFactory:
    """Factory for constructing registered neural network architectures."""
    @staticmethod
    def create_architecture(architecture, **kwargs):
        """Create architecture.
        
        Args:
            architecture: Registered architecture identifier.
            **kwargs: Additional options forwarded to the implementation.
        
        Returns:
            Requested result.
        """
        if architecture not in _ARCHITECTURES:
            raise ValueError(f"Unknown architecture type: {architecture}")
        return _ARCHITECTURES[architecture](**kwargs)
