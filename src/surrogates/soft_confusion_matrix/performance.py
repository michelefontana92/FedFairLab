import torch

def binary_accuracy(y_hat,**kwargs):
    #print('Binary Accuracy')
    tp = true_positive(y_hat,**kwargs)
    tn = true_negative(y_hat,**kwargs)
    fp = false_positive(y_hat,**kwargs)
    fn = false_negative(y_hat,**kwargs)
    if tp + tn + fp + fn < 1e-3:
        return torch.tensor(0.0)
    return (tp + tn) / (tp + tn + fp + fn)


def true_positive(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    get_probability = kwargs.get('get_probability', False)
    assert positive_mask is not None
    
    # Controlla se la maschera è vuota
    if torch.sum(positive_mask).item() == 0:
        return torch.tensor(0.0)
    
    positive_proba = torch.mean(y_hat[positive_mask])
    
    if not get_probability:
        positive_proba *= torch.sum(positive_mask)
        
    return positive_proba

def true_negative(y_hat, **kwargs):
    negative_mask = ~kwargs.get('positive_mask')
    get_probability = kwargs.get('get_probability', False)
    assert negative_mask is not None
    
    # Controlla se la maschera è vuota
    if torch.sum(negative_mask).item() == 0:
        return torch.tensor(0.0)
    
    negative_proba = 1 - torch.mean(y_hat[negative_mask])
    
    if not get_probability:
        negative_proba *= torch.sum(negative_mask)
        
    return negative_proba

def false_positive(y_hat, **kwargs):
    negative_mask = ~kwargs.get('positive_mask')
    get_probability = kwargs.get('get_probability', False)
    assert negative_mask is not None
    
    # Controlla se la maschera è vuota
    if torch.sum(negative_mask).item() == 0:
        return torch.tensor(0.0)
    
    negative_proba = torch.mean(y_hat[negative_mask])
    
    if not get_probability:
        negative_proba *= torch.sum(negative_mask)
        
    return negative_proba

def false_negative(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    get_probability = kwargs.get('get_probability', False)
    assert positive_mask is not None
    
    # Controlla se la maschera è vuota
    if torch.sum(positive_mask).item() == 0:
        return torch.tensor(0.0)
    
    positive_proba = 1 - torch.mean(y_hat[positive_mask])
    
    if not get_probability:
        positive_proba *= torch.sum(positive_mask)
        
    return positive_proba

def _precision(y_hat, **kwargs):
    tp = true_positive(y_hat, **kwargs)
    fp = false_positive(y_hat, **kwargs)
    
    # Controllo per NaN e numeri molto piccoli
    if torch.isnan(tp) or torch.isnan(fp) or (tp + fp < 1e-6):
        return torch.tensor(0.0)
    
    return tp / (tp + fp)

def _recall(y_hat, **kwargs):
    tp = true_positive(y_hat, **kwargs)
    fn = false_negative(y_hat, **kwargs)
    
    # Controllo per NaN e numeri molto piccoli
    if torch.isnan(tp) or torch.isnan(fn) or (tp + fn < 1e-6):
        return torch.tensor(0.0)
    
    return tp / (tp + fn)

def binary_precision(y_hat, **kwargs):
    average = kwargs.get('average')
    if average is None:
        return _precision(y_hat, **kwargs)
    elif average == 'weighted':
        return _weighted_precision(y_hat, **kwargs)
    else: 
        raise ValueError(f'{average} method is unknown')
    
def binary_recall(y_hat, **kwargs):
    average = kwargs.get('average')
    if average is None:
        return _recall(y_hat, **kwargs)
    elif average == 'weighted':
        return _weighted_recall(y_hat, **kwargs)
    else: 
        raise ValueError(f'{average} method is unknown')

def _f1_score(y_hat, **kwargs):
    kwargs['average'] = None
    precision = binary_precision(y_hat, **kwargs)
    recall = binary_recall(y_hat, **kwargs)
    
    # Prevenzione di divisioni per zero
    if precision + recall < 1e-6:
        return torch.tensor(0.0)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Controllo per NaN
    if torch.isnan(f1):
        print('F1 is NaN!')
        return torch.tensor(0.0)
    
    return f1

def binary_f1_score(y_hat, **kwargs):
    average = kwargs.get('average')
    if average is None:
        return _f1_score(y_hat, **kwargs)
    elif average == 'weighted':
        return _weighted_f1_score(y_hat, **kwargs)
    else: 
        raise ValueError(f'{average} method is unknown')

def _weighted_f1_score(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    assert positive_mask is not None 
    negative_mask = ~positive_mask
    f1_score_class_0 = _f1_score(1 - y_hat, positive_mask=negative_mask)
    f1_score_class_1 = _f1_score(y_hat, positive_mask=positive_mask)
    n_positive = torch.sum(positive_mask)
    n_negative = torch.sum(negative_mask)
    n_records = n_positive + n_negative
    
    return (n_positive * f1_score_class_1 + n_negative * f1_score_class_0) / n_records

def _weighted_precision(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    assert positive_mask is not None 
    negative_mask = ~positive_mask
    precision_class_0 = _precision(1 - y_hat, positive_mask=negative_mask)
    precision_class_1 = _precision(y_hat, positive_mask=positive_mask)
    n_positive = torch.sum(positive_mask)
    n_negative = torch.sum(negative_mask)
    n_records = n_positive + n_negative
    
    return (n_positive * precision_class_1 + n_negative * precision_class_0) / n_records

def _weighted_recall(y_hat, **kwargs):
    positive_mask = kwargs.get('positive_mask')
    assert positive_mask is not None 
    negative_mask = ~positive_mask
    recall_class_0 = _recall(1 - y_hat, positive_mask=negative_mask)
    recall_class_1 = _recall(y_hat, positive_mask=positive_mask)
    n_positive = torch.sum(positive_mask)
    n_negative = torch.sum(negative_mask)
    n_records = n_positive + n_negative
    
    return (n_positive * recall_class_1 + n_negative * recall_class_0) / n_records