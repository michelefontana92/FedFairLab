class EarlyStopping:
    """
    Early stopping utility to stop training when a monitored metric has stopped improving.
    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
        monitor (str): Metric to be monitored.
        mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing.
        counter (int): Counts the number of epochs with no improvement.
        best_score (float or None): Best score observed so far.
        early_stop (bool): Whether training should be stopped.
    Methods:
        __call__(**kwargs): Checks if training should be stopped based on the monitored metric.
        reset(): Resets the early stopping state.
    """
    def __init__(self, patience=5, delta=0.0,
                 monitor='val_loss', mode='min'):
        """Initialize the object.
        
        Args:
            patience: Number of non-improving checks before stopping.
            delta: Minimum improvement or distance margin.
            monitor: Metric name monitored by the callback.
            mode: Optimization direction for the monitored metric.
        """
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, **kwargs):
        """Evaluate the callable object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        metrics = kwargs.get('metrics')
        assert metrics is not None, "Metrics are required for EarlyStopping"
        assert isinstance(metrics, dict), "Logs must be a dictionary"
        score = metrics[self.monitor]
        if (self.best_score is not None) and ((self.mode == 'min' and score >= self.best_score) or (self.mode == 'max' and score <= self.best_score)):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop,self.counter

    def reset(self,keep_best=False):
        """Reset.
        
        Args:
            keep_best: Whether to restore the best checkpoint after training.
        """
        self.counter = 0
        self.early_stop = False
        if keep_best:
            self.best_score = self.best_score
        else: 
            self.best_score = None
    
   