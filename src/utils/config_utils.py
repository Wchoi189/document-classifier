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
            result = OmegaConf.to_container(config_path, resolve=True)
            if not isinstance(result, dict):
                raise TypeError("Config must be a dictionary after conversion, got: {}".format(type(result)))
            # Ensure keys are str for type safety
            if not all(isinstance(k, str) for k in result.keys()):
                raise TypeError("All config keys must be strings after conversion.")
            return dict(result)  # type: ignore
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

#Reusable configuration utilities to handle Hydra/OmegaConf safely
def safe_get(config, key_path, default=None):
    """
    Safely get nested config values without Pylance warnings
    
    Args:
        config: Configuration dict (might be None)
        key_path: String like 'data.augmentation.strategy' or list ['data', 'augmentation', 'strategy']
        default: Default value if not found
    
    Returns:
        Config value or default
    """
    if config is None or not isinstance(config, dict):
        return default
    
    # Convert string path to list
    if isinstance(key_path, str):
        keys = key_path.split('.')
    else:
        keys = key_path
    
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current

def safe_config_get(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from config dictionary."""
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dictionary, got {type(config)}")
    return config.get(key, default)


def convert_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string config values to appropriate types."""
    # Convert seed to int
    if 'seed' in config and isinstance(config['seed'], str):
        config['seed'] = int(config['seed'])
    
    # Convert data config
    if 'data' in config and isinstance(config['data'], dict):
        data_config = config['data']
        
        # Convert numeric values
        if 'image_size' in data_config and isinstance(data_config['image_size'], str):
            data_config['image_size'] = int(data_config['image_size'])
        
        if 'val_size' in data_config and isinstance(data_config['val_size'], str):
            data_config['val_size'] = float(data_config['val_size'])
            
        if 'num_workers' in data_config and isinstance(data_config['num_workers'], str):
            data_config['num_workers'] = int(data_config['num_workers'])
        
        # Convert lists (mean, std)
        if 'mean' in data_config and isinstance(data_config['mean'], str):
            data_config['mean'] = [float(x.strip()) for x in data_config['mean'].split(',')]
        elif 'mean' in data_config and isinstance(data_config['mean'], list):
            data_config['mean'] = [float(x) for x in data_config['mean']]
            
        if 'std' in data_config and isinstance(data_config['std'], str):
            data_config['std'] = [float(x.strip()) for x in data_config['std'].split(',')]
        elif 'std' in data_config and isinstance(data_config['std'], list):
            data_config['std'] = [float(x) for x in data_config['std']]
    
    # Convert train config
    if 'train' in config and isinstance(config['train'], dict):
        train_config = config['train']
        if 'batch_size' in train_config and isinstance(train_config['batch_size'], str):
            train_config['batch_size'] = int(train_config['batch_size'])
    
    # Convert model config
    if 'model' in config and isinstance(config['model'], dict):
        model_config = config['model']
        if 'pretrained' in model_config and isinstance(model_config['pretrained'], str):
            model_config['pretrained'] = model_config['pretrained'].lower() in ('true', '1', 'yes')
    
    # Convert optimizer config
    if 'optimizer' in config and isinstance(config['optimizer'], dict):
        opt_config = config['optimizer']
        if 'learning_rate' in opt_config and isinstance(opt_config['learning_rate'], str):
            opt_config['learning_rate'] = float(opt_config['learning_rate'])
        if 'weight_decay' in opt_config and isinstance(opt_config['weight_decay'], str):
            opt_config['weight_decay'] = float(opt_config['weight_decay'])
        if 'momentum' in opt_config and isinstance(opt_config['momentum'], str):
            opt_config['momentum'] = float(opt_config['momentum'])
    
    # Convert scheduler config
    if 'scheduler' in config and isinstance(config['scheduler'], dict):
        sched_config = config['scheduler']
        for key in ['T_0', 'T_mult', 'T_max', 'step_size']:
            if key in sched_config and isinstance(sched_config[key], str):
                sched_config[key] = int(sched_config[key])
        for key in ['eta_min', 'gamma']:
            if key in sched_config and isinstance(sched_config[key], str):
                sched_config[key] = float(sched_config[key])
    
    return config