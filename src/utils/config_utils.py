"""
Configuration utilities to handle both legacy YAML configs and new Hydra configs
"""
import yaml
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from either legacy YAML or return Hydra config as dict.
    
    Args:
        config_path: Path to config file or Hydra DictConfig
        
    Returns:
        Dictionary configuration
    """
    if isinstance(config_path, (DictConfig, dict)):
        # If it's already a config object, convert to dict
        if isinstance(config_path, DictConfig):
            return OmegaConf.to_container(config_path, resolve=True)
        return config_path
    
    # Legacy YAML loading
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def normalize_config_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config structure to ensure compatibility between old and new formats.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Normalized configuration dictionary
    """
    normalized = config.copy()
    
    # Handle different optimizer config formats
    if 'optimizer' in normalized:
        opt_config = normalized['optimizer']
        if 'learning_rate' in opt_config and 'lr' not in opt_config:
            opt_config['lr'] = opt_config['learning_rate']
        elif 'lr' in opt_config and 'learning_rate' not in opt_config:
            opt_config['learning_rate'] = opt_config['lr']
    
    # Handle scheduler config normalization
    if 'scheduler' in normalized:
        sched_config = normalized['scheduler']
        if 'name' in sched_config:
            # Convert scheduler name to params structure if needed
            if sched_config['name'] == 'CosineAnnealingWarmRestarts':
                if 'scheduler_params' not in normalized.get('train', {}):
                    if 'train' not in normalized:
                        normalized['train'] = {}
                    normalized['train']['scheduler_params'] = {
                        'T_0': sched_config.get('T_0', 10),
                        'T_mult': sched_config.get('T_mult', 2),
                        'eta_min': sched_config.get('eta_min', 0.0)
                    }
                    normalized['train']['scheduler'] = sched_config['name']
    
    # Ensure paths structure exists
    if 'paths' not in normalized:
        normalized['paths'] = {
            'output_dir': 'outputs',
            'prediction_dir': 'predictions', 
            'model_dir': 'models',
            'batch_dir': 'batch',
            'batch_summary_filename': 'batch_summary.csv'
        }
    
    # Ensure logging structure exists for early stopping
    if 'logging' not in normalized:
        normalized['logging'] = {
            'checkpoint_dir': str(Path(normalized['paths']['output_dir']) / normalized['paths']['model_dir']),
            'log_dir': str(Path(normalized['paths']['output_dir']) / 'logs')
        }
    
    return normalized


def get_experiment_name(config: Dict[str, Any]) -> str:
    """
    Get experiment name from config, handling both old and new formats.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment name string
    """
    # Try new Hydra format first
    if 'experiment' in config and isinstance(config['experiment'], dict):
        return config['experiment'].get('name', 'default_experiment')
    
    # Try WandB run name
    if 'wandb' in config and 'name' in config['wandb']:
        return config['wandb']['name']
    
    # Default fallback
    return 'default_experiment'


def update_wandb_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update WandB configuration with experiment info from Hydra config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    config = config.copy()
    
    if 'experiment' in config and 'wandb' in config:
        experiment = config['experiment']
        wandb_config = config['wandb']
        
        # Update WandB name if not set
        if 'name' not in wandb_config or not wandb_config['name']:
            wandb_config['name'] = experiment.get('name', 'hydra_experiment')
        
        # Update tags
        if 'tags' in experiment:
            existing_tags = wandb_config.get('tags', [])
            experiment_tags = experiment['tags']
            # Merge tags, avoiding duplicates
            all_tags = list(set(existing_tags + experiment_tags))
            wandb_config['tags'] = all_tags
        
        # Update notes
        if 'description' in experiment:
            wandb_config['notes'] = experiment['description']
    
    return config


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the current configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*50)
    print("📋 CONFIGURATION SUMMARY")
    print("="*50)
    
    # Experiment info
    exp_name = get_experiment_name(config)
    print(f"🏷️  Experiment: {exp_name}")
    
    if 'experiment' in config:
        exp = config['experiment']
        if 'description' in exp:
            print(f"📝 Description: {exp['description']}")
        if 'tags' in exp:
            print(f"🏷️  Tags: {', '.join(exp['tags'])}")
    
    # Model info
    if 'model' in config:
        model = config['model']
        print(f"🧠 Model: {model.get('name', 'unknown')}")
        print(f"📏 Pretrained: {model.get('pretrained', False)}")
    
    # Training info
    if 'train' in config:
        train = config['train']
        print(f"🏋️  Epochs: {train.get('epochs', 'unknown')}")
        print(f"📦 Batch Size: {train.get('batch_size', 'unknown')}")
    
    # Optimizer info
    if 'optimizer' in config:
        opt = config['optimizer']
        lr = opt.get('learning_rate', opt.get('lr', 'unknown'))
        print(f"⚙️  Optimizer: {opt.get('name', 'unknown')} (lr={lr})")
    
    # Scheduler info
    if 'scheduler' in config:
        sched = config['scheduler']
        print(f"📈 Scheduler: {sched.get('name', 'unknown')}")
    
    # Data info
    if 'data' in config:
        data = config['data']
        print(f"🖼️  Image Size: {data.get('image_size', 'unknown')}")
        print(f"🔄 Augmentation: {data.get('use_document_augmentation', False)}")
    
    print("="*50 + "\n")


# Backward compatibility function
def load_config_legacy(config_path: str) -> Dict[str, Any]:
    """Legacy config loading function for backward compatibility."""
    return load_config(config_path)