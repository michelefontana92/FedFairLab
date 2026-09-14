from debug_utils import debug_print
from torchmetrics import Accuracy, Precision, Recall,F1Score
from .metrics_factory import register_metric
from .base_metric import BaseMetric
import torch

@register_metric('performance')
class Performance(BaseMetric):
    """Implementation of Performance."""
    def __init__(self,**kwargs):
        """Initialize the object.
        
        Args:
            **kwargs: Additional options forwarded to the implementation.
        """
        task = kwargs.get('task','multiclass')
        num_classes = kwargs.get('num_classes',2)
        average = kwargs.get('average','weighted')
        debug_print(f"Initializing Performance metric with task: {task}, num_classes: {num_classes}, average: {average}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.accuracy = Accuracy(task=task,
                                 num_classes=num_classes).to(self.device)
        self.precision = Precision(task='multiclass',
                                   num_classes=num_classes,
                                   average='weighted').to(self.device)
        self.recall = Recall(task='multiclass',
                             num_classes=num_classes,
                             average='weighted').to(self.device)
        self.f1 = F1Score(task='multiclass',
                          num_classes=num_classes,
                          average='weighted').to(self.device)
        
    def calculate(self, y_pred, y_true):
        #print('Y_pred: ',y_pred[:10])
        #print('Y_true: ',y_true[:10])
        """Calculate the metric value.
        
        Args:
            y_pred: Predicted labels or model outputs.
            y_true: Ground-truth labels.
        """
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)
        #print(f'Labels: {y_true.unique()}')
        #print(f'Predictions: {y_pred.unique()}')
        self.accuracy.update(y_pred, y_true)
        self.precision.update(y_pred, y_true)
        self.recall.update(y_pred, y_true)
        self.f1.update(y_pred, y_true)

    def get(self):
        """Return the current value."""
        return {
            "accuracy": self.accuracy.compute().cpu(),
            "precision": self.precision.compute().cpu(),
            "recall": self.recall.compute().cpu(),
            "f1": self.f1.compute().cpu()
        }

    def reset(self):
        """Reset."""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

