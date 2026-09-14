_SERVERS = {}

def register_server(server_type):
    """Register server.
    
    Args:
        server_type: Registered server identifier.
    """
    def decorator(fn):
        """Register the decorated class or function and return it unchanged.
        
        Args:
            fn: Callable or class being registered.
        """
        _SERVERS[server_type] = fn
        return fn
    return decorator

class ServerFactory:
    """Factory for constructing registered federated server implementations."""
    @staticmethod
    def create(server_type, **kwargs):
        """Create a registered instance.
        
        Args:
            server_type: Registered server identifier.
            **kwargs: Additional options forwarded to the implementation.
        """
        if server_type not in _SERVERS:
            raise ValueError(f"Unknown server type: {server_type}")
        return _SERVERS[server_type](**kwargs)
