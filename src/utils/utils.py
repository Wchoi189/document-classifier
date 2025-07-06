import yaml
import torch
import random
import numpy as np
import os
from src.utils.config_utils import load_config as load_config_new, normalize_config_structure

def load_config(config_path):
    """YAML 설정 파일을 로드합니다. (Now supports both legacy and Hydra)"""
    config = load_config_new(config_path)  # Use the new function from config_utils
    return normalize_config_structure(config)

def set_seed(seed):
    """재현성을 위해 랜덤 시드를 설정합니다."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cudnn을 사용하면 결과가 달라질 수 있지만, 속도 향상을 위해 아래 두 옵션은 주석 처리할 수 있습니다.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def get_project_root():
    """Returns the root directory of the project (two levels up from this file)."""
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
       
    )


def get_model_dir(model_name):
    return os.path.join(
        get_project_root(),  
        "models",            
        model_name           
    )


def auto_increment_run_suffix(name: str, pad=3):
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)

def convert_numpy_types(obj):
    """Recursively convert numpy types in a structure to native Python types. It is useful before serializing to JSON."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class EarlyStopping:
    """조기 종료를 위한 클래스"""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, mode='min', metric='val_loss'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_best = np.inf if mode == 'min' else -np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode
        self.metric = metric

    def __call__(self, val_metric, model):
        score = -val_metric if self.mode == 'max' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''최상의 모델을 저장합니다.'''
        if self.verbose:
            self.trace_func(f'{self.metric} ({self.val_metric_best:.6f} --> {val_metric:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_best = val_metric
    