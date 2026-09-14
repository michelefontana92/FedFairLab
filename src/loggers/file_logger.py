from .base_logger import BaseLogger
import os
from icecream import ic
from .logger_factory import register_logger

@register_logger("file")
class FileLogger(BaseLogger):
    """Implementation of FileLogger."""
    def __init__(self, **kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        file_path = kwargs['file_path']
        file_dir = kwargs.get('file_dir', './logs')
        include_context = kwargs.get('include_context', False)
        prefix = kwargs.get('prefix', '')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        self.file_path = os.path.join(file_dir, file_path)
        self.include_context = include_context
        self.prefix = prefix
        self.reset()

    def reset(self):
        """Reset."""
        with open(self.file_path, 'w') as file:
            file.write('')
    
    def log(self, message):
        """Log.
        
        Args:
            message: Message to log.
        """
        def output_fn(message):
            """Handle output fn.
            
            Args:
                message: Message to log.
            """
            with open(self.file_path, 'a') as file:
                file.write(f'{message}\n')
        
        ic.configureOutput(prefix=self.prefix,
                           includeContext=self.include_context,
                           outputFunction=output_fn)
        
        ic(message)

    def error(self, message):
        """Log an error.
        
        Args:
            message: Message to log.
        """
        self.log(f'[ERROR] {message}')

    def info(self, message):
        """Log an informational message.
        
        Args:
            message: Message to log.
        """
        self.log(f'[INFO] {message}')

    def debug(self, message):
        """Log debug information.
        
        Args:
            message: Message to log.
        """
        self.log(f'[DEBUG] {message}')
    
    def close(self):
        """Close."""
        pass