_AGGREGATORS = {}

def register_aggregator(name:str):
    """Register aggregator.
    
    Args:
        name: Registered object name.
    """
    def decorator(cls):
        """Register the decorated class or function and return it unchanged.
        
        Args:
            cls: Class being registered.
        """
        if name in _AGGREGATORS:
            raise ValueError(f"Cannot register {name} as "
                             f"it is already registered")
        _AGGREGATORS[name] = cls
        return cls
    return decorator

class AggregatorFactory:
    """Factory for constructing registered aggregation strategies."""
    @staticmethod
    def create(name:str,**kwargs):
        """Create a registered instance.
        
        Args:
            name: Registered object name.
            **kwargs: Additional options forwarded to the implementation.
        """
        if name not in _AGGREGATORS:
            raise ValueError(f"{name} not found")
        return _AGGREGATORS[name](**kwargs)
