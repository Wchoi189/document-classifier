import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pickle
import json


class EnsembleModel:
    """여러 모델의 예측을 결합하는 앙상블 클래스"""
    
    def __init__(self, models: List[nn.Module], ensemble_method: str = 'voting'):
        """
        Args:
            models: 앙상블할 모델들의 리스트
            ensemble_method: 앙상블 방법 ('voting', 'averaging', 'weighted', 'stacking')
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = None
        
    def set_weights(self, weights: List[float]):
        """가중 앙상블을 위한 가중치 설정"""
        if len(weights) != len(self.models):
            raise ValueError("가중치 수가 모델 수와 일치하지 않습니다.")
        self.weights = weights
        
    def predict(self, x: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """앙상블 예측 수행"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x.to(device))
                predictions.append(pred.cpu())
        
        predictions = torch.stack(predictions, dim=0)
        
        if self.ensemble_method == 'voting':
            return self._voting_ensemble(predictions)
        elif self.ensemble_method == 'averaging':
            return self._averaging_ensemble(predictions)
        elif self.ensemble_method == 'weighted':
            return self._weighted_ensemble(predictions)
        elif self.ensemble_method == 'stacking':
            return self._stacking_ensemble(predictions)
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {self.ensemble_method}")
    
    def _voting_ensemble(self, predictions: torch.Tensor) -> torch.Tensor:
        """투표 기반 앙상블 (하드 보팅)"""
        # 각 모델의 예측 클래스
        predicted_classes = torch.argmax(predictions, dim=-1)
        
        # 다수결 투표
        ensemble_pred = torch.mode(predicted_classes, dim=0)[0]
        
        # 원-핫 인코딩으로 변환
        batch_size = predictions.shape[1]
        num_classes = predictions.shape[2]
        result = torch.zeros(batch_size, num_classes)
        result.scatter_(1, ensemble_pred.unsqueeze(1), 1)
        
        return result
    
    def _averaging_ensemble(self, predictions: torch.Tensor) -> torch.Tensor:
        """평균 기반 앙상블 (소프트 보팅)"""
        return torch.mean(predictions, dim=0)
    
    def _weighted_ensemble(self, predictions: torch.Tensor) -> torch.Tensor:
        """가중 평균 기반 앙상블"""
        if self.weights is None:
            # 가중치가 설정되지 않은 경우 균등 가중치 사용
            weights = torch.ones(len(self.models)) / len(self.models)
        else:
            weights = torch.tensor(self.weights)
        
        # 가중 평균 계산
        weighted_pred = torch.sum(predictions * weights.unsqueeze(1).unsqueeze(2), dim=0)
        return weighted_pred
    
    def _stacking_ensemble(self, predictions: torch.Tensor) -> torch.Tensor:
        """스태킹 앙상블 (메타 모델 사용)"""
        # 간단한 구현: 로지스틱 회귀를 시뮬레이션
        # 실제로는 별도의 메타 모델을 훈련해야 함
        return torch.mean(predictions, dim=0)


class TestTimeAugmentation:
    """테스트 타임 증강 기반 앙상블"""
    
    def __init__(self, model: nn.Module, augmentations: List[Any]):
        """
        Args:
            model: 예측할 모델
            augmentations: 적용할 증강 기법들의 리스트
        """
        self.model = model
        self.augmentations = augmentations
        
    def predict(self, x: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """TTA 기반 예측"""
        predictions = []
        
        # 원본 이미지로 예측
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x.to(device))
            predictions.append(pred.cpu())
        
        # 증강된 이미지들로 예측
        for aug in self.augmentations:
            x_aug = aug(x)
            with torch.no_grad():
                pred_aug = self.model(x_aug.to(device))
                predictions.append(pred_aug.cpu())
        
        # 평균 계산
        predictions = torch.stack(predictions, dim=0)
        return torch.mean(predictions, dim=0)


class CrossValidationEnsemble:
    """교차 검증 기반 앙상블"""
    
    def __init__(self, model_class, model_params: Dict[str, Any], n_folds: int = 5):
        """
        Args:
            model_class: 모델 클래스
            model_params: 모델 파라미터
            n_folds: 교차 검증 폴드 수
        """
        self.model_class = model_class
        self.model_params = model_params
        self.n_folds = n_folds
        self.models = []
        
    def train_models(self, train_loader, val_loaders, trainer_config):
        """각 폴드별로 모델 훈련"""
        for fold in range(self.n_folds):
            print(f"폴드 {fold + 1}/{self.n_folds} 훈련 중...")
            
            # 모델 생성
            model = self.model_class(**self.model_params)
            
            # Lightning 모듈 생성 및 훈련
            # (실제 구현에서는 trainer를 사용하여 훈련)
            
            self.models.append(model)
    
    def predict(self, x: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """교차 검증 모델들의 앙상블 예측"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x.to(device))
                predictions.append(pred.cpu())
        
        predictions = torch.stack(predictions, dim=0)
        return torch.mean(predictions, dim=0)


class SnapshotEnsemble:
    """스냅샷 앙상블 (동일 모델의 다른 체크포인트들)"""
    
    def __init__(self, model: nn.Module, checkpoint_paths: List[str]):
        """
        Args:
            model: 기본 모델
            checkpoint_paths: 체크포인트 파일 경로들
        """
        self.model = model
        self.checkpoint_paths = checkpoint_paths
        self.models = []
        
        # 각 체크포인트에서 모델 로드
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_copy = type(model)()
            model_copy.load_state_dict(checkpoint['state_dict'])
            self.models.append(model_copy)
    
    def predict(self, x: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        """스냅샷 앙상블 예측"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x.to(device))
                predictions.append(pred.cpu())
        
        predictions = torch.stack(predictions, dim=0)
        return torch.mean(predictions, dim=0)


def create_ensemble_from_checkpoints(
    model_class, 
    model_params: Dict[str, Any], 
    checkpoint_paths: List[str],
    ensemble_method: str = 'averaging'
) -> EnsembleModel:
    """체크포인트들로부터 앙상블 모델 생성"""
    models = []
    
    for checkpoint_path in checkpoint_paths:
        # 모델 생성
        model = model_class(**model_params)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        models.append(model)
    
    return EnsembleModel(models, ensemble_method)


def save_ensemble_config(ensemble_model: EnsembleModel, save_path: str):
    """앙상블 설정 저장"""
    config = {
        'ensemble_method': ensemble_model.ensemble_method,
        'num_models': len(ensemble_model.models),
        'weights': ensemble_model.weights
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_ensemble_config(load_path: str) -> Dict[str, Any]:
    """앙상블 설정 로드"""
    with open(load_path, 'r') as f:
        return json.load(f) 