from .income_3_run import Income3Run
from ..run_factory import register_run
from builder import FedFairLabBuilder


@register_run('income3_fedfairlab')
class Income3FedFairLabRun(Income3Run):
    """FedFairLab experiment configuration for three-class Income prediction."""

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
