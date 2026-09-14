from debug_utils import debug_print
from .base_logger import BaseLogger
import wandb
from icecream import ic
from .logger_factory import register_logger
from .file_logger import FileLogger
import os
import pickle as pkl
from dataloaders import DataModule

@register_logger("wandb")
class WandbLogger(BaseLogger):
    """Implementation of WandbLogger."""
    def __init__(self, **kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        self.project=kwargs['project']
        self.config = kwargs['config']
        self.id = kwargs.get('id', 'client-1')
        self.file_path = kwargs.get('file_path', f'{self.id}.log')
        self.file_dir = kwargs.get('file_dir', './logs')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', './checkpoints')
        self.checkpoint_path = kwargs.get('checkpoint_path', 'model.h5')
        self.prefix = kwargs.get('prefix', '')
        self.data_module = kwargs.get('data_module',None)
        self.log_model = kwargs.get('log_model', False)
        self.include_context = kwargs.get('include_context', False)
        self.file_logger = FileLogger(file_path=self.file_path,
                                      file_dir=self.file_dir,
                                      prefix=self.prefix,
                                      include_context=self.include_context)
                                
        self.run = wandb.init(
            project=self.project,
            config=self.config,
            job_type='client',
            reinit="return_previous",
            name=self.id,
        )
                   
        self.artifact = wandb.Artifact(name=self.file_path, type='log')
        self.data_artifact = None
        if self.data_module is not None:
            self.data_artifact = wandb.Artifact(name='data', type='data')
            
    def serialize(self):
        """
        Serializza gli attributi essenziali del logger.
        """
        return {
            'project': self.project,
            'config': self.config,
            'id': self.id,
            'file_path': self.file_path,
            'file_dir': self.file_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'checkpoint_path': self.checkpoint_path,
            'prefix': self.prefix,
            'log_model': self.log_model,
            'include_context': self.include_context,
            'data_module': self.data_module.serialize() if self.data_module else None  # Presuppone che data_module abbia un metodo serialize
        }

    @staticmethod
    def deserialize(data):
        """
        Ricostruisce un'istanza di WandbLogger dai dati serializzati.
        """
        # Inizializza il logger con i parametri serializzati
        logger = WandbLogger(
            project=data['project'],
            config=data['config'],
            id=data['id'],
            file_path=data['file_path'],
            file_dir=data['file_dir'],
            checkpoint_dir=data['checkpoint_dir'],
            checkpoint_path=data['checkpoint_path'],
            prefix=data['prefix'],
            log_model=data.get('log_model', False),
            include_context=data['include_context'],
            data_module=DataModule.deserialize(data['data_module']) if data['data_module'] else None
        )
        return logger
    
    def log(self, message):
        """Log.
        
        Args:
            message: Message to log.
        """
        self.file_logger.log(message)
        wandb.log(message)

    def error(self, message):
        """Log an error.
        
        Args:
            message: Message to log.
        """
        self.file_logger.log(f'[ERROR] {message}')

    def info(self, message):
        """Log an informational message.
        
        Args:
            message: Message to log.
        """
        self.file_logger.log(f'[INFO] {message}')

    def debug(self, message):
        """Log debug information.
        
        Args:
            message: Message to log.
        """
        self.file_logger.log(f'[DEBUG] {message}')
    
    def log_artifact(self, name,path):
        """Log an artifact.
        
        Args:
            name: Registered object name.
            path: Filesystem path to read from or write to.
        """
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)
        
    def close(self):
        # Log artifact: data module
        """Close."""
        if self.data_module is not None:
            with open('data.pkl', 'wb') as f:
                pkl.dump(self.data_module, f)
            self.data_artifact.add_file('data.pkl')
            try:
                self.run.log_artifact(self.data_artifact)
            except wandb.errors.UsageError as e:
                debug_print("Cannot log data_artifact: run already finished.")
            os.remove('data.pkl')

        # Log artifact: generic file
        file_path = os.path.join(self.file_dir, self.file_path)
        if os.path.exists(file_path):
            self.artifact.add_file(file_path)
            try:
                self.run.log_artifact(self.artifact)
            except wandb.errors.UsageError as e:
                debug_print("Cannot log artifact: run already finished.")

        # Close local file logger
        self.file_logger.close()

        # Only then finish the wandb run (and be sure to use self.run)
        if self.run is not None:
            self.run.finish()

        
        
        
