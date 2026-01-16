import torch
import numpy as np
from sklearn.metrics import f1_score
from soft_confusion_matrix.performance import multiclass_f1_score

torch.manual_seed(0)

# Simuliamo 10 esempi, 4 classi
labels = torch.randint(0, 4, (10,))
logits = torch.randn(10, 4)

# Applichiamo entmax (o softmax per test di base)
from entmax import entmax15
probabilities = entmax15(logits, dim=1)  # [10, 4]
# Conversione alle classi predette
preds = torch.argmax(probabilities, dim=1).numpy()
true = labels.numpy()

# F1 con sklearn
f1_macro_sklearn = f1_score(true, preds, average='macro')
f1_weighted_sklearn = f1_score(true, preds, average='weighted')

# F1 con il tuo codice
f1_macro_custom = multiclass_f1_score(probabilities, labels=labels, average='macro')
f1_weighted_custom = multiclass_f1_score(probabilities, labels=labels, average='weighted')

print("SKLEARN  macro:", f1_macro_sklearn)
print("CUSTOM   macro:", f1_macro_custom.item())
print("SKLEARN  weighted:", f1_weighted_sklearn)
print("CUSTOM   weighted:", f1_weighted_custom.item())

preds = torch.argmax(probabilities, dim=1)
one_hot_preds = torch.nn.functional.one_hot(preds, num_classes=probabilities.shape[1]).float()

f1_custom_hard = multiclass_f1_score(one_hot_preds, labels=labels, average='weighted')
print("CUSTOM weighted (hard preds):", f1_custom_hard.item())
from torchmetrics.classification import MulticlassF1Score

metric = MulticlassF1Score(num_classes=4, average='macro')
preds = torch.argmax(probabilities, dim=1)
f1_tm = metric(preds, labels)
print("TorchMetrics macro:", f1_tm.item())
metric = MulticlassF1Score(num_classes=4, average='weighted')
preds = torch.argmax(probabilities, dim=1)
f1_tm = metric(preds, labels)
print("TorchMetrics weighted:", f1_tm.item())