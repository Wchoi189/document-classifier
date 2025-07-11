"""
Configuration utilities to handle both legacy YAML configs and new Hydra configs
"""
import yaml
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any, Union
from icecream import ic


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Enhanced config loading with Hydra defaults support
    """
    if isinstance(config_path, (DictConfig, dict)):
        # 이미 로드된 config인 경우
        if isinstance(config_path, DictConfig):
            result = OmegaConf.to_container(config_path, resolve=True)
            if not isinstance(result, dict):
                raise TypeError("Config must be a dictionary after conversion")
            return {str(k): v for k, v in result.items()}
        return config_path
    
    # 파일에서 로드하는 경우
    config_path = Path(config_path)
    
    # Hydra defaults가 있는 경우 수동 병합
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            temp_config = yaml.safe_load(f)
        
        if 'defaults' in temp_config:
            ic("🔧 Hydra defaults 감지, 수동 병합 실행")
            config = load_and_merge_hydra_defaults(config_path)
        else:
            ic("📄 일반 YAML 로드")
            config = temp_config
            
    except Exception as e:
        ic(f"❌ Config 로드 실패: {e}")
        raise
    
    return config


def ensure_required_config_sections(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    필수 config 섹션들이 없는 경우 기본값으로 추가
    """
    ic("🔧 필수 config 섹션 확인 및 추가")
    
    # 기본 model 설정
    if 'model' not in config:
        ic("➕ model 섹션 추가")
        config['model'] = {
            'name': 'resnet50',
            'pretrained': True,
            'dropout_rate': 0.5
        }
    
    # 기본 optimizer 설정
    if 'optimizer' not in config:
        ic("➕ optimizer 섹션 추가")
        config['optimizer'] = {
            'name': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.01
        }
    
    # 기본 scheduler 설정
    if 'scheduler' not in config:
        ic("➕ scheduler 섹션 추가")
        config['scheduler'] = {
            'name': 'CosineAnnealingWarmRestarts',
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 0.00001
        }
    
    # train 섹션 보완
    if 'train' in config:
        train_config = config['train']
        if 'early_stopping' not in train_config:
            ic("➕ early_stopping 섹션 추가")
            train_config['early_stopping'] = {
                'patience': 8,
                'metric': 'val_f1',
                'mode': 'max'
            }
    
    # paths 섹션 보완
    if 'paths' not in config:
        ic("➕ paths 섹션 추가")
        config['paths'] = {
            'output_dir': 'outputs',
            'prediction_dir': 'predictions',
            'model_dir': 'models',
            'batch_dir': 'batch',
            'batch_summary_filename': 'batch_summary.csv'
        }
    
    # logging 섹션 보완
    if 'logging' not in config:
        ic("➕ logging 섹션 추가")
        config['logging'] = {
            'checkpoint_dir': str(Path(config['paths']['output_dir']) / config['paths']['model_dir']),
            'log_dir': str(Path(config['paths']['output_dir']) / 'logs')
        }
    
    ic(f"보완 완료, 최종 키들: {list(config.keys())}")
    return config


def normalize_config_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced config normalization with required sections
    """
    # 기존 normalize 로직
    normalized = config.copy()
    
    # Handle different optimizer config formats
    if 'optimizer' in normalized:
        opt_config = normalized['optimizer']
        if 'learning_rate' in opt_config and 'lr' not in opt_config:
            opt_config['lr'] = opt_config['learning_rate']
        elif 'lr' in opt_config and 'learning_rate' not in opt_config:
            opt_config['learning_rate'] = opt_config['lr']
    
    # 필수 섹션 보장
    normalized = ensure_required_config_sections(normalized)
    
    return normalized


# 기존 함수들 유지 (backward compatibility)
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
    
    return config

# Backward compatibility function
def load_config_legacy(config_path: str) -> Dict[str, Any]:
    """Legacy config loading function for backward compatibility."""
    return load_config(config_path)

# def normalize_config_structure(config: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Normalize config structure to ensure compatibility between old and new formats.
    
#     Args:
#         config: Configuration dictionary
        
#     Returns:
#         Normalized configuration dictionary
#     """
#     normalized = config.copy()
    
#     # Handle different optimizer config formats
#     if 'optimizer' in normalized:
#         opt_config = normalized['optimizer']
#         if 'learning_rate' in opt_config and 'lr' not in opt_config:
#             opt_config['lr'] = opt_config['learning_rate']
#         elif 'lr' in opt_config and 'learning_rate' not in opt_config:
#             opt_config['learning_rate'] = opt_config['lr']
    
#     # Handle scheduler config normalization
#     if 'scheduler' in normalized:
#         sched_config = normalized['scheduler']
#         if 'name' in sched_config:
#             # Convert scheduler name to params structure if needed
#             if sched_config['name'] == 'CosineAnnealingWarmRestarts':
#                 if 'scheduler_params' not in normalized.get('train', {}):
#                     if 'train' not in normalized:
#                         normalized['train'] = {}
#                     normalized['train']['scheduler_params'] = {
#                         'T_0': sched_config.get('T_0', 10),
#                         'T_mult': sched_config.get('T_mult', 2),
#                         'eta_min': sched_config.get('eta_min', 0.0)
#                     }
#                     normalized['train']['scheduler'] = sched_config['name']
    
#     # Ensure paths structure exists
#     if 'paths' not in normalized:
#         normalized['paths'] = {
#             'output_dir': 'outputs',
#             'prediction_dir': 'predictions', 
#             'model_dir': 'models',
#             'batch_dir': 'batch',
#             'batch_summary_filename': 'batch_summary.csv'
#         }
    
#     # Ensure logging structure exists for early stopping
#     if 'logging' not in normalized:
#         normalized['logging'] = {
#             'checkpoint_dir': str(Path(normalized['paths']['output_dir']) / normalized['paths']['model_dir']),
#             'log_dir': str(Path(normalized['paths']['output_dir']) / 'logs')
#         }
    
#     return normalized


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

# def safe_config_get(config: Dict[str, Any], key: str, default: Any = None) -> Any:
#     """Safely get a value from config dictionary."""
#     if not isinstance(config, dict):
#         raise TypeError(f"Config must be a dictionary, got {type(config)}")
#     return config.get(key, default)


# def convert_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
#     """Convert string config values to appropriate types."""
#     # Convert seed to int
#     if 'seed' in config and isinstance(config['seed'], str):
#         config['seed'] = int(config['seed'])
    
#     # Convert data config
#     if 'data' in config and isinstance(config['data'], dict):
#         data_config = config['data']
        
#         # Convert numeric values
#         if 'image_size' in data_config and isinstance(data_config['image_size'], str):
#             data_config['image_size'] = int(data_config['image_size'])
        
#         if 'val_size' in data_config and isinstance(data_config['val_size'], str):
#             data_config['val_size'] = float(data_config['val_size'])
            
#         if 'num_workers' in data_config and isinstance(data_config['num_workers'], str):
#             data_config['num_workers'] = int(data_config['num_workers'])
        
#         # Convert lists (mean, std)
#         if 'mean' in data_config and isinstance(data_config['mean'], str):
#             data_config['mean'] = [float(x.strip()) for x in data_config['mean'].split(',')]
#         elif 'mean' in data_config and isinstance(data_config['mean'], list):
#             data_config['mean'] = [float(x) for x in data_config['mean']]
            
#         if 'std' in data_config and isinstance(data_config['std'], str):
#             data_config['std'] = [float(x.strip()) for x in data_config['std'].split(',')]
#         elif 'std' in data_config and isinstance(data_config['std'], list):
#             data_config['std'] = [float(x) for x in data_config['std']]
    
#     # Convert train config
#     if 'train' in config and isinstance(config['train'], dict):
#         train_config = config['train']
#         if 'batch_size' in train_config and isinstance(train_config['batch_size'], str):
#             train_config['batch_size'] = int(train_config['batch_size'])
    
#     # Convert model config
#     if 'model' in config and isinstance(config['model'], dict):
#         model_config = config['model']
#         if 'pretrained' in model_config and isinstance(model_config['pretrained'], str):
#             model_config['pretrained'] = model_config['pretrained'].lower() in ('true', '1', 'yes')
    
#     # Convert optimizer config
#     if 'optimizer' in config and isinstance(config['optimizer'], dict):
#         opt_config = config['optimizer']
#         if 'learning_rate' in opt_config and isinstance(opt_config['learning_rate'], str):
#             opt_config['learning_rate'] = float(opt_config['learning_rate'])
#         if 'weight_decay' in opt_config and isinstance(opt_config['weight_decay'], str):
#             opt_config['weight_decay'] = float(opt_config['weight_decay'])
#         if 'momentum' in opt_config and isinstance(opt_config['momentum'], str):
#             opt_config['momentum'] = float(opt_config['momentum'])
    
#     # Convert scheduler config
#     if 'scheduler' in config and isinstance(config['scheduler'], dict):
#         sched_config = config['scheduler']
#         for key in ['T_0', 'T_mult', 'T_max', 'step_size']:
#             if key in sched_config and isinstance(sched_config[key], str):
#                 sched_config[key] = int(sched_config[key])
#         for key in ['eta_min', 'gamma']:
#             if key in sched_config and isinstance(sched_config[key], str):
#                 sched_config[key] = float(sched_config[key])
    
#     return config

def safe_dict_get(dictionary: Dict[Union[str, int], Any], key: str) -> Any:
    """
    Safely get a value from a dictionary that might have string or int keys.
    
    This handles the case where classification_report returns a dict with
    mixed key types (str for class names, int for numeric labels).
    
    Args:
        dictionary: Dictionary with potentially mixed key types
        key: String key to look up
        
    Returns:
        Value if found, None otherwise
    """
    # First try direct string key access
    if key in dictionary:
        return dictionary[key]
    
    # If not found, try converting key to int and back
    try:
        int_key = int(key)
        if int_key in dictionary:
            return dictionary[int_key]
    except (ValueError, TypeError):
        pass
    
    return None


def safe_classification_report_access(report: Dict[Any, Any], class_key: str) -> Dict[str, Any]:
    """
    Safely access classification report metrics for a specific class.
    
    Args:
        report: Classification report dictionary from sklearn
        class_key: Class name/label as string
        
    Returns:
        Class metrics dictionary or empty dict if not found
    """
    # Use .get() method which is type-safe
    class_metrics = report.get(class_key)
    
    if class_metrics is not None and isinstance(class_metrics, dict):
        return class_metrics
    
    # Try with integer conversion if string key doesn't work
    try:
        int_key = int(class_key)
        class_metrics = report.get(int_key)
        if class_metrics is not None and isinstance(class_metrics, dict):
            return class_metrics
    except (ValueError, TypeError):
        pass
    
    return {}


def get_classification_metrics(report: Dict[Any, Any], class_name: Union[str, int]) -> Dict[str, float]:
    """
    Type-safe getter for classification report metrics with default values.
    
    Args:
        report: Classification report from sklearn
        class_name: Class name or label
        
    Returns:
        Dictionary with precision, recall, f1-score, support
    """
    key = str(class_name)
    metrics = report.get(key, {})
    
    if isinstance(metrics, dict):
        return {
            'precision': float(metrics.get('precision', 0.0)),
            'recall': float(metrics.get('recall', 0.0)),
            'f1-score': float(metrics.get('f1-score', 0.0)),
            'support': int(metrics.get('support', 0))
        }
    
    return {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}


def load_and_merge_hydra_defaults(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Hydra defaults를 수동으로 로드하고 병합하는 함수
    Conservative tester에서 Hydra context 없이 사용할 때 필요
    """
    config_path = Path(config_path)
    ic(f"🔧 Hydra defaults 수동 병합 시작: {config_path}")
    
    # 1. 메인 config 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    
    ic("메인 config 로드 완료")
    
    # 2. defaults 섹션 확인
    defaults = main_config.get('defaults', [])
    if not defaults:
        ic("⚠️ defaults 섹션 없음, 메인 config만 반환")
        return main_config
    
    ic(f"발견된 defaults: {defaults}")
    
    # 3. 병합할 config 수집
    merged_config = {}
    configs_dir = config_path.parent
    
    for default_item in defaults:
        if default_item == '_self_':
            # _self_는 나중에 처리
            continue
        elif isinstance(default_item, dict):
            # {category: name} 형태
            for category, name in default_item.items():
                config_file = configs_dir / category / f"{name}.yaml"
                if config_file.exists():
                    ic(f"✅ {category}/{name}.yaml 로드 중")
                    with open(config_file, 'r', encoding='utf-8') as f:
                        category_config = yaml.safe_load(f)
                    merged_config[category] = category_config
                else:
                    ic(f"⚠️ {config_file} 파일 없음")
        elif isinstance(default_item, str):
            # 단일 파일명
            config_file = configs_dir / f"{default_item}.yaml"
            if config_file.exists():
                ic(f"✅ {default_item}.yaml 로드 중")
                with open(config_file, 'r', encoding='utf-8') as f:
                    item_config = yaml.safe_load(f)
                merged_config.update(item_config)
    
    # 4. 메인 config와 병합 (_self_ 처리)
    for key, value in main_config.items():
        if key != 'defaults':  # defaults는 제외
            merged_config[key] = value
    
    ic(f"병합 완료, 최종 키들: {list(merged_config.keys())}")
    return merged_config

def ensure_required_config_sections(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    필수 config 섹션들이 없는 경우 기본값으로 추가
    """
    ic("🔧 필수 config 섹션 확인 및 추가")
    
    # 🔧 FIXED: Add missing global settings
    if 'seed' not in config:
        ic("➕ seed 추가")
        config['seed'] = 42
    
    if 'device' not in config:
        ic("➕ device 추가")
        config['device'] = 'cuda'
    
    # 🔧 FIXED: Ensure experiment section exists
    if 'experiment' not in config:
        ic("➕ experiment 섹션 추가")
        config['experiment'] = {
            'name': 'default_experiment',
            'description': 'Default experiment configuration',
            'tags': ['default']
        }
    
    # 기본 model 설정
    if 'model' not in config:
        ic("➕ model 섹션 추가")
        config['model'] = {
            'name': 'resnet50',
            'pretrained': True,
            'dropout_rate': 0.5
        }
    
    # 기본 optimizer 설정
    if 'optimizer' not in config:
        ic("➕ optimizer 섹션 추가")
        config['optimizer'] = {
            'name': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.01
        }
    
    # 기본 scheduler 설정
    if 'scheduler' not in config:
        ic("➕ scheduler 섹션 추가")
        config['scheduler'] = {
            'name': 'CosineAnnealingWarmRestarts',
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 0.00001
        }
    
    # 🔧 FIXED: Ensure augmentation section exists
    if 'augmentation' not in config:
        ic("➕ augmentation 섹션 추가")
        config['augmentation'] = {
            'enabled': True,
            'strategy': 'basic',
            'intensity': 0.3
        }
    
    # train 섹션 보완
    if 'train' in config:
        train_config = config['train']
        if 'early_stopping' not in train_config:
            ic("➕ early_stopping 섹션 추가")
            train_config['early_stopping'] = {
                'patience': 8,
                'metric': 'val_f1',
                'mode': 'max'
            }
    
    # paths 섹션 보완
    if 'paths' not in config:
        ic("➕ paths 섹션 추가")
        config['paths'] = {
            'output_dir': 'outputs',
            'prediction_dir': 'predictions',
            'model_dir': 'models',
            'batch_dir': 'batch',
            'batch_summary_filename': 'batch_summary.csv'
        }
    
    # logging 섹션 보완
    if 'logging' not in config:
        ic("➕ logging 섹션 추가")
        config['logging'] = {
            'checkpoint_dir': str(Path(config['paths']['output_dir']) / config['paths']['model_dir']),
            'log_dir': str(Path(config['paths']['output_dir']) / 'logs')
        }
    
    ic(f"보완 완료, 최종 키들: {list(config.keys())}")
    return config