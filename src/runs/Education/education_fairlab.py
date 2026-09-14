from .education_run import EducationRun
from ..run_factory import register_run
from builder import FedFairLabBuilder


@register_run('education_fedfairlab')
class EducationFedFairLabRun(EducationRun):
    """FedFairLab experiment configuration for Education."""
    def __init__(self, **kwargs) -> None:
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        super().__init__(**kwargs)
        kwargs['run_dict'] = self.to_dict()
        self.builder = FedFairLabBuilder(**kwargs)
    
    def setUp(self):
        """Handle setUp."""
        pass

    def run(self):
        """Handle run."""
        self.builder.run()

    def tearDown(self) -> None:
        """Handle tearDown."""
        super().tearDown()
