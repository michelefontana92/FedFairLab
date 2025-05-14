import ray 

class Base_Builder:
    
    def _assign_resources(self):
        num_clients = self.num_clients
        self.num_cpus = num_clients + 1
        self.num_gpus = len(self.gpu_devices)
     
        self.num_gpus_per_client = self.num_gpus//num_clients if self.num_gpus > 0 else 0
        

    def __init__(self,**kwargs):
        self.num_clients = kwargs.get('num_clients')
        self.gpu_devices = kwargs.get('gpu_devices',[]) 
        self._assign_resources()
        
    
    def run(self):
        print('Number of CPUs:',self.num_cpus)
        print('Number of GPUs:',self.num_gpus)
        print('Number of GPUs per client:',self.num_gpus_per_client)
        ray.init(num_cpus=self.num_cpus,num_gpus=self.num_gpus)
        self.server.setup()
        #self.server.execute()
        self.server.shutdown()
        ray.shutdown()
     