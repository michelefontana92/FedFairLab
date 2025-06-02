import numpy as np
import torch
import copy
import torch.nn.functional as F

def compute_group_cardinality(group_name,sensitive_attributes):
        for name,group_dict in sensitive_attributes:
            if name == group_name:
                total = 1
                for key in group_dict.keys():
                    total *= len(group_dict[key])
                return total 
        raise KeyError(f'Group {group_name} not found in sensitive attributes') 

def average_dictionary_list(dictionary_list):
    result = {k:0 for k in dictionary_list[0].keys()}
    for d in dictionary_list:
        for k,v in d.items():
            result[k] += v
    for k in result.keys():
        result[k] /= len(dictionary_list)
    return result

def scoring_function(results,use_training=False,weight_constraint=10):
    prefix = 'train' if use_training else 'val'
    score = 1-results[f'{prefix}_objective_fn']
   # print(f'Val constraints:',results['val_constraints'])
    for constraint in results[f'{prefix}_constraints']:
        score -= constraint*weight_constraint
    return score

def collect_local_results(**kwargs):
    results = kwargs.get('eval_results')
    params = kwargs.get('model_params')
    assert len(results) == len(params), "Results and parameters must have the same length"
    performance_constraint = kwargs.get('performance_constraint')
    original_threshold_list = kwargs.get('original_threshold_list')
    thresholds = [performance_constraint]+list(original_threshold_list) if performance_constraint<1.0 else original_threshold_list
    
    local_results = {k : {} for k in range(len(results))}
    for i in range(len(results)):
        local_results[i] = {k : {} for k in range(len(results))}
    
    
    for i in range(len(results)):
        for j,res in enumerate(results[i]):
            res_constraints = res['val_constraints']
            new_res_list = []
            for k,constraint in enumerate(res_constraints):   
                tau = thresholds[k]
                if performance_constraint<1.0 and k==0:
                    new_res_list.append(max(0,tau-constraint))
                else:
                    new_res_list.append(max(0,constraint-tau))
            
            l_res = copy.deepcopy(res)
            l_res['val_constraints'] = new_res_list
            l_res['metrics']['val_constraints_score'] = scoring_function(l_res,use_training=False)
            #print(f'[CLIENT {i}] LOCAL Evaluation results: {l_res["metrics"]}\n{l_res["metrics"]["val_constraints_score"]}')
            #print()
            local_results[i][j] = { 
                'model_params':params[i],
                'score':l_res['metrics']['val_constraints_score']
                } 
                           
    return local_results


def select_from_scores(scores, tau=1):
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    scores*=10
    #print('Scores:',scores)
    # Normalizzazione min-max per evitare problemi di scala
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    # Applica softmax sui punteggi negativi (per preferire i più piccoli)
    probabilities = F.softmax(-scores / tau, dim=0)
    #print('Probabilities:',probabilities)
    # Estrae un indice in base alle probabilità
    selected = torch.multinomial(probabilities, num_samples=1).item()

    # Per confronto/debug: l'indice del punteggio minimo
    argmin = torch.argmin(scores).item()

    return selected, argmin

def compute_global_score(**kwargs):
        performance_constraint = kwargs.get('performance_constraint')
        original_threshold_list = kwargs.get('original_threshold_list')
        results = kwargs.get('eval_results')
        assert results is not None, "Evaluation results are required"
        
       
        #print('Results:',results)
        global_results = {k:[] for k in results[0].keys()}
        
        for result in results:
            for kind in result.keys():
                global_results[kind].append(result[kind])
        
        global_scores = {}
        for k,v in global_results.items():
            if isinstance(v[0],dict):
                global_scores[k] = average_dictionary_list(v)
            else: 
                global_scores[k] = np.mean(np.array(v),axis=0)
             
        #print("Global results:",global_scores)
        if len(original_threshold_list) > 0:
            for kind,res_list in global_scores.items():
                #if kind in ['train_constraints','val_constraints']:
                if kind in ['val_constraints']:
                    new_res_list = []
                    thresholds = [performance_constraint]+list(original_threshold_list) if performance_constraint<1.0 else original_threshold_list
                    #print('Thresholds:',thresholds)
                    for i,res in enumerate(res_list):   
                        tau = thresholds[i]
                        if performance_constraint<1.0 and i==0:
                            new_res_list.append(max(0,tau-res))
                        else:
                            new_res_list.append(max(0,res-tau))
                    global_scores[kind] = new_res_list
            
        del global_scores['metrics']['val_constraints_score']
        #print("Global results after threshold:",global_scores)
        #train_global_score = scoring_function(global_scores,use_training=True)
        val_global_score = scoring_function(global_scores,use_training=False)
        #print("Global scores:",train_global_score,val_global_score)
        #global_scores['metrics']['train_global_score'] = train_global_score
        global_scores['metrics']['val_global_score'] = val_global_score
        #print('Global scores:',global_scores)
        return global_scores