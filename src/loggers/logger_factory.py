_LOGGERS = {}

def register_logger(logger_type):
    """Register logger.
    
    Args:
        logger_type: Registered logger identifier.
    """
    def decorator(fn):
        """Register the decorated class or function and return it unchanged.
        
        Args:
            fn: Callable or class being registered.
        """
        _LOGGERS[logger_type] = fn
        return fn
    return decorator

class LoggerFactory:
    """Factory for constructing registered logger implementations."""
    @staticmethod
    def create_logger(logger_type, **kwargs):
        """Create logger.
        
        Args:
            logger_type: Registered logger identifier.
            **kwargs: Additional options forwarded to the implementation.
        
        Returns:
            Requested result.
        """
        if logger_type not in _LOGGERS:
            raise ValueError(f"Unknown logger type: {logger_type}")
        return _LOGGERS[logger_type](**kwargs)
