_CLIENTS = {}

def register_client(client_type):
    """
    Register a client implementation under a factory key.

    Args:
        client_type: String identifier used by ``ClientFactory.create``.

    Returns:
        A decorator that stores the decorated client class in the registry.
    """
    def decorator(fn):
        """Store the decorated client class in the global registry."""
        _CLIENTS[client_type] = fn
        return fn
    return decorator

class ClientFactory:
    """
    Factory for local or Ray-remote client instances.

    Client classes register themselves through ``register_client``. The factory is
    the single construction point used by builders, which keeps Ray resource
    assignment separate from client implementation details.
    """
    @staticmethod
    def create(client_type, remote=False,num_gpus=0,num_cpus=1,**kwargs):
        """
        Create a client instance.

        Args:
            client_type: Registry key of the requested client implementation.
            remote: If true, create a Ray actor instead of a local object.
            num_gpus: Ray GPU resources assigned to the actor.
            num_cpus: Ray CPU resources assigned to the actor. Fractional values
                allow multiple client actors to share one logical CPU.
            **kwargs: Constructor arguments forwarded to the client class.

        Returns:
            A local client object or a Ray actor handle.

        Raises:
            ValueError: If ``client_type`` has not been registered.
        """
        if client_type not in _CLIENTS:
            raise ValueError(f"Unknown client type: {client_type}")
        if remote:
            if num_gpus > 0:
                return _CLIENTS[client_type].options(num_cpus=num_cpus,num_gpus=num_gpus).remote(**kwargs)
            else:
                return _CLIENTS[client_type].options(num_cpus=num_cpus).remote(**kwargs)
        return _CLIENTS[client_type](**kwargs)
