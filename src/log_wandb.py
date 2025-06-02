import wandb 

results = {
    'final_val_demographic_parity_JobRace':0.4593,
    'final_val_demographic_parity_JobMarital':0.5271,
    'final_val_demographic_parity_RaceMarital':0.5148,
    'final_val_demographic_parity_Job':0.1941,
    'final_val_demographic_parity_Race':0.3046,
    'final_val_demographic_parity_Marital':0.3547,
}

run_id = '3u7ijuuz'
project='FedFairLab_Folk_New'
run = wandb.init(project=project, id=run_id, 
           resume='allow')

run.log(results)
run.finish()