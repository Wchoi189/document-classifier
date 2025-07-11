import yaml
import torch
import random
import numpy as np
import os

def load_config(config_path):
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

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
