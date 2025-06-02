import torch

def _binary_class_demographic_parity(probabilities, group_masks, group_ids):
    available_groups = torch.unique(group_masks)
    
    # Escludi il calcolo se uno dei gruppi target non è presente
    if not all(gid in available_groups for gid in group_ids):
        return None  # oppure torch.tensor(float('nan'))

    mask_0 = group_masks == group_ids[0]
    mask_1 = group_masks == group_ids[1]

    dp = torch.mean(probabilities[mask_0]) - torch.mean(probabilities[mask_1])
    return torch.abs(dp)

   
def demographic_parity(probabilities, **kwargs):
    group_masks = kwargs.get('group_masks')
    group_ids = kwargs.get('target_groups')
   
    
    probabilities = probabilities[:, 1]
    return _binary_class_demographic_parity(probabilities, group_masks, group_ids)

def _binary_class_equal_opportunity(probabilities,labels_mask,group_masks,group_ids):
    
    if len(torch.unique(group_masks)) < 2:
       
        return torch.tensor(0.0, device=probabilities.device)
  
    else:
        positive_mask = (group_masks == group_ids[0]) & (labels_mask)
        negative_mask = (group_masks == group_ids[1]) & (labels_mask)
        if torch.sum(positive_mask) == 0 or torch.sum(negative_mask) == 0:
            return torch.tensor(0.0, device=probabilities.device)
        eo = torch.mean(probabilities[positive_mask]) - torch.mean(probabilities[negative_mask])
        return torch.abs(eo)
    
def equal_opportunity(probabilities,**kwargs):
    group_masks = kwargs.get('group_masks')
    group_ids = kwargs.get('group_ids')
    labels = kwargs.get('labels')
   
    assert group_masks is not None, 'group_masks must be provided'
    assert group_ids is not None, 'group_ids must be provided'
    assert labels is not None, 'labels must be provided'
   
    probabilities = probabilities[:,1]
    labels_mask = labels == 1
        
    return _binary_class_equal_opportunity(probabilities,labels_mask,
                                               group_masks,group_ids)

    

def predictive_equality(probabilities,**kwargs):
    group_masks = kwargs.get('group_masks')
    group_ids = kwargs.get('group_ids')
    labels = kwargs.get('labels')
    
    assert group_masks is not None, 'group_masks must be provided'
    assert group_ids is not None, 'group_ids must be provided'
    assert labels is not None, 'labels must be provided'
    
    probabilities = probabilities[:,1]
    labels_mask = labels == 0
    return _binary_class_equal_opportunity(probabilities,labels_mask,
                                               group_masks,group_ids)
    

       
def equalized_odds(probabilities,**kwargs):
    
    eo = equal_opportunity(probabilities,**kwargs)
    pe = predictive_equality(probabilities,**kwargs)
    return torch.max(eo,pe)