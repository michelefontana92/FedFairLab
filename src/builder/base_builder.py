from debug_utils import debug_print
import ray


class Base_Builder:

    """Implementation of Base Builder."""
    def _assign_resources(self):
        """Handle assign resources."""
        num_clients = self.num_clients
        self.num_cpus = num_clients + 1
        self.num_gpus = len(self.gpu_devices)

        self.num_gpus_per_client = self.num_gpus//num_clients if self.num_gpus > 0 else 0

    def __init__(self, **kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        self.num_clients = kwargs.get('num_clients')
        self.gpu_devices = kwargs.get('gpu_devices', [])
        self._assign_resources()

    def run(self):
        """Handle run."""
        debug_print('Number of CPUs:', self.num_cpus)
        debug_print('Number of GPUs:', self.num_gpus)
        debug_print('Number of GPUs per client:', self.num_gpus_per_client)
        ray.init(num_cpus=self.num_cpus, num_gpus=self.num_gpus)
        self.server.setup()
        self.server.execute()
        self.server.shutdown()
        ray.shutdown()
