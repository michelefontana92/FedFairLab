from .base_run import BaseRun
_RUNS ={}

def register_run(run):
    """Register run.
    
    Args:
        run: Registered run identifier.
    """
    def decorator(cls):
        """Register the decorated class or function and return it unchanged.
        
        Args:
            cls: Class being registered.
        """
        if run in _RUNS:
            raise ValueError(f"Cannot register duplicate run ({run})")
        if not issubclass(cls, BaseRun):
            raise ValueError(f"run ({run}: {cls.__name__}) must extend BaseRun")
        _RUNS[run] = cls
        return cls
    return decorator

class RunFactory:
    """Factory for constructing registered experiment runs."""
    @staticmethod
    def create_run(run, **kwargs):
        """Create run.
        
        Args:
            run: Registered run identifier.
            **kwargs: Additional options forwarded to the implementation.
        
        Returns:
            Requested result.
        """
        if run not in _RUNS:
            raise ValueError(f"Unknown run type: {run}")
        return _RUNS[run](**kwargs)
