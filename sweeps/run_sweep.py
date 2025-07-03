import wandb
import yaml
from src.utils.utils import load_config

def run_sweep():
    # Initialize sweep
    with open('sweeps/sweep_config.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(sweep_config, project="document-classifier")
    
    def train_with_sweep():
        # Initialize wandb run
        wandb.init()
        
        # Get sweep parameters
        config = load_config('configs/config.yaml')
        
        # Update config with sweep parameters
        for key, value in wandb.config.items():
            keys = key.split('_')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
        
        # Run training
        from train import main
        main(config)
    
    # Start sweep
    wandb.agent(sweep_id, train_with_sweep, count=20)

if __name__ == "__main__":
    run_sweep()