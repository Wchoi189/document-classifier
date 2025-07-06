"""
src/training/conservative_augmentation_tester.py

보수적 증강 테스터 - 기존 성능 유지하며 점진적 개선
Conservative augmentation tester for safe progressive enhancement
"""

import sys
import os
from pathlib import Path

# 🔧 프로젝트 루트 경로 설정 (항상 첫 번째로)
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)
os.chdir(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import fire
from icecream import ic
import copy
from datetime import datetime

from src.utils.config_utils import load_config, normalize_config_structure
from src.data.csv_dataset import CSVDocumentDataset
from src.data.augmentation import get_configurable_transforms, get_valid_transforms
from src.models.model import create_model
from src.trainer.wandb_trainer import WandBTrainer
from src.trainer.trainer import Trainer
from src.inference.predictor import predict_from_checkpoint, save_predictions


class ConservativeAugmentationTester:
    """기존 성능 유지하며 안전하게 증강 테스트하는 시스템"""
    
    def __init__(self, config_path: str = "configs/config.yaml", 
                 baseline_checkpoint: str = None):
        """
        초기화
        Args:
            config_path: 설정 파일 경로
            baseline_checkpoint: 기준점이 되는 모델 체크포인트
        """
        self.config = load_config(config_path)
        self.config = normalize_config_structure(self.config)
        self.baseline_checkpoint = baseline_checkpoint
        self.setup_paths()
        ic("보수적 증강 테스터 초기화 완료")
        
    def setup_paths(self):
        """경로 설정"""
        self.output_dir = Path('outputs/conservative_augmentation_test')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 결과 저장용 디렉토리
        self.experiment_dir = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        ic(f"실험 결과 저장: {self.experiment_dir}")
    
    def create_progressive_augmentation_configs(self) -> List[Dict]:
        """점진적 증강 설정들 생성 (Gemini 전략 기반)"""
        ic("점진적 증강 설정 생성")
        
        # 기본 설정 복사
        base_config = copy.deepcopy(self.config)
        
        # Gemini의 권장사항 기반 점진적 설정
        progressive_configs = []
        
        # Phase 0: 기준점 (기존 설정)
        baseline_config = copy.deepcopy(base_config)
        baseline_config['experiment_name'] = 'phase_0_baseline'
        baseline_config['description'] = '기준점 - 기존 설정'
        baseline_config['data']['augmentation'] = {
            'strategy': 'basic',
            'intensity': 0.3
        }
        progressive_configs.append(baseline_config)
        
        # Phase 1: 회전 도입 (Critical Priority)
        phase1_config = copy.deepcopy(base_config)
        phase1_config['experiment_name'] = 'phase_1_rotation_mild'
        phase1_config['description'] = 'Phase 1: 회전 증강 도입 (±10°)'
        phase1_config['data']['augmentation'] = {
            'strategy': 'robust',
            'intensity': 0.5,
            'geometric': {
                'enabled': True,
                'intensity': 0.3,
                'rotation_limit': 10,  # ±10° (Gemini 권장)
                'perspective_scale': [0.02, 0.08]
            },
            'lighting': {
                'enabled': True,
                'intensity': 0.4,
                'brightness_contrast_limit': 0.2,
                'gamma_range': [80, 120]
            },
            'quality': {
                'enabled': False  # 아직 블러는 추가하지 않음
            }
        }
        progressive_configs.append(phase1_config)
        
        # Phase 2: 회전 + 조명 강화
        phase2_config = copy.deepcopy(base_config)
        phase2_config['experiment_name'] = 'phase_2_rotation_lighting'
        phase2_config['description'] = 'Phase 2: 회전 + 조명 증강 (±15°, 과노출 시뮬레이션)'
        phase2_config['data']['augmentation'] = {
            'strategy': 'robust',
            'intensity': 0.6,
            'geometric': {
                'enabled': True,
                'intensity': 0.5,
                'rotation_limit': 15,  # ±15° 확장
                'perspective_scale': [0.03, 0.10]
            },
            'lighting': {
                'enabled': True,
                'intensity': 0.6,
                'brightness_contrast_limit': 0.3,  # 과노출 시뮬레이션 강화
                'gamma_range': [70, 130],
                'overexposure_probability': 0.3  # 과노출 빈도 증가
            },
            'quality': {
                'enabled': True,
                'intensity': 0.3,  # 약한 블러만
                'blur_probability': 0.2
            }
        }
        progressive_configs.append(phase2_config)
        
        # Phase 3: 타겟 강도 (±25°)
        phase3_config = copy.deepcopy(base_config)
        phase3_config['experiment_name'] = 'phase_3_target_intensity'
        phase3_config['description'] = 'Phase 3: 타겟 강도 (±25°, 풀 조명 증강)'
        phase3_config['data']['augmentation'] = {
            'strategy': 'robust',
            'intensity': 0.7,
            'geometric': {
                'enabled': True,
                'intensity': 0.7,
                'rotation_limit': 25,  # Gemini Phase 2 수준
                'perspective_scale': [0.05, 0.15]
            },
            'lighting': {
                'enabled': True,
                'intensity': 0.8,
                'brightness_contrast_limit': 0.4,  # [0.8, 1.8] 범위
                'gamma_range': [70, 130],
                'overexposure_probability': 0.5  # 테스트 데이터 수준
            },
            'quality': {
                'enabled': True,
                'intensity': 0.5,
                'blur_probability': 0.4,
                'noise_probability': 0.3
            }
        }
        progressive_configs.append(phase3_config)
        
        # Phase 4: 최대 강도 (±45° - Gemini 최종 권장)
        phase4_config = copy.deepcopy(base_config)
        phase4_config['experiment_name'] = 'phase_4_maximum_robustness'
        phase4_config['description'] = 'Phase 4: 최대 robustness (±45°)'
        phase4_config['data']['augmentation'] = {
            'strategy': 'robust',
            'intensity': 0.8,
            'geometric': {
                'enabled': True,
                'intensity': 0.9,
                'rotation_limit': 45,  # Gemini 최종 권장
                'perspective_scale': [0.05, 0.15]
            },
            'lighting': {
                'enabled': True,
                'intensity': 0.8,
                'brightness_contrast_limit': 0.5,  # 최대 강도
                'gamma_range': [60, 140],
                'overexposure_probability': 0.6
            },
            'quality': {
                'enabled': True,
                'intensity': 0.6,
                'blur_probability': 0.5,
                'noise_probability': 0.4
            }
        }
        progressive_configs.append(phase4_config)
        
        ic(f"생성된 증강 설정: {len(progressive_configs)}개 단계")
        return progressive_configs
    
    def create_augmentation_transforms(self, aug_config: Dict):
        """증강 설정으로부터 transform 생성"""
        data_config = self.config['data']
        img_size = data_config['image_size']
        mean = data_config['mean']
        std = data_config['std']
        
        # 설정된 증강 전략에 따라 transform 생성
        if aug_config.get('strategy') == 'robust':
            return get_configurable_transforms(img_size, img_size, mean, std, aug_config)
        else:
            # 기본 증강으로 폴백
            from src.data.augmentation import get_train_transforms
            return get_train_transforms(img_size, img_size, mean, std)
    
    def run_quick_validation(self, config: Dict, epochs: int = 3) -> Dict:
        """빠른 검증 훈련 (성능 변화 확인용)"""
        ic(f"빠른 검증 시작: {config['experiment_name']}")
        
        # 디바이스 설정
        device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 데이터 준비
        augmentation_config = config['data'].get('augmentation', {})
        
        # Transform 생성
        train_transforms = self.create_augmentation_transforms(augmentation_config)
        valid_transforms = get_valid_transforms(
            height=self.config['data']['image_size'],
            width=self.config['data']['image_size'],
            mean=self.config['data']['mean'],
            std=self.config['data']['std']
        )
        
        # 데이터셋 생성
        train_dataset = CSVDocumentDataset(
            root_dir=self.config['data']['root_dir'],
            csv_file=self.config['data']['csv_file'],
            meta_file=self.config['data']['meta_file'],
            split='train',
            transform=train_transforms,
            val_size=self.config['data']['val_size'],
            seed=self.config['seed']
        )
        
        val_dataset = CSVDocumentDataset(
            root_dir=self.config['data']['root_dir'],
            csv_file=self.config['data']['csv_file'],
            meta_file=self.config['data']['meta_file'],
            split='val',
            transform=valid_transforms,
            val_size=self.config['data']['val_size'],
            seed=self.config['seed']
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # 모델 생성 (안전한 config 접근)
        num_classes = len(train_dataset.classes)
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'resnet50')
        pretrained = model_config.get('pretrained', True)
        
        model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained
        ).to(device)
        
        # 기준 모델 가중치 로드 (있는 경우)
        if self.baseline_checkpoint and Path(self.baseline_checkpoint).exists():
            ic(f"기준 모델 로드: {self.baseline_checkpoint}")
            checkpoint = torch.load(self.baseline_checkpoint, map_location=device)
            model.load_state_dict(checkpoint)
        
        # 옵티마이저 및 스케줄러
        optimizer_config = self.config['optimizer']
        if optimizer_config['name'] == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['name'] == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        
        # 손실 함수
        loss_fn = nn.CrossEntropyLoss()
        
        # 빠른 훈련을 위한 설정 수정
        quick_config = copy.deepcopy(config)
        quick_config['train']['epochs'] = epochs
        quick_config['wandb']['enabled'] = False  # WandB 로깅 비활성화
        
        # 트레이너 생성 (WandB 없는 기본 트레이너)
        trainer = Trainer(model, optimizer, None, loss_fn, train_loader, val_loader, device, quick_config)
        
        # 훈련 실행
        trainer.train()
        
        # 결과 수집
        if trainer.history:
            final_results = trainer.history[-1]
            results = {
                'experiment_name': config['experiment_name'],
                'final_val_accuracy': final_results.get('val_acc', 0),
                'final_val_f1': final_results.get('val_f1', 0),
                'final_train_loss': final_results.get('train_loss', 0),
                'final_val_loss': final_results.get('val_loss', 0),
                'epochs_completed': len(trainer.history),
                'augmentation_config': augmentation_config
            }
        else:
            results = {
                'experiment_name': config['experiment_name'],
                'error': 'No training history available',
                'augmentation_config': augmentation_config
            }
        
        # 모델 저장
        model_path = self.experiment_dir / f"{config['experiment_name']}_model.pth"
        torch.save(model.state_dict(), model_path)
        results['model_path'] = str(model_path)
        
        ic(f"빠른 검증 완료: {config['experiment_name']}")
        ic(f"Val F1: {results.get('final_val_f1', 0):.4f}")
        
        return results
    
    def compare_with_baseline(self, results: List[Dict]) -> Dict:
        """기준점 대비 성능 비교"""
        ic("기준점 대비 성능 비교")
        
        baseline_result = None
        for result in results:
            if 'baseline' in result['experiment_name']:
                baseline_result = result
                break
        
        if not baseline_result:
            ic("기준점 결과 없음")
            return {}
        
        baseline_f1 = baseline_result.get('final_val_f1', 0)
        baseline_acc = baseline_result.get('final_val_accuracy', 0)
        
        comparison = {
            'baseline_performance': {
                'f1': baseline_f1,
                'accuracy': baseline_acc
            },
            'phase_comparisons': []
        }
        
        for result in results:
            if result['experiment_name'] == baseline_result['experiment_name']:
                continue
            
            current_f1 = result.get('final_val_f1', 0)
            current_acc = result.get('final_val_accuracy', 0)
            
            f1_improvement = current_f1 - baseline_f1
            acc_improvement = current_acc - baseline_acc
            
            phase_comparison = {
                'experiment_name': result['experiment_name'],
                'f1_score': current_f1,
                'accuracy': current_acc,
                'f1_improvement': f1_improvement,
                'f1_improvement_percent': (f1_improvement / baseline_f1 * 100) if baseline_f1 > 0 else 0,
                'accuracy_improvement': acc_improvement,
                'performance_safe': f1_improvement >= -0.05,  # 5% 이하 성능 저하만 허용
                'recommended_for_full_training': f1_improvement > 0.02 and f1_improvement >= -0.02
            }
            
            comparison['phase_comparisons'].append(phase_comparison)
        
        # 최고 성능 단계 식별
        best_phase = max(comparison['phase_comparisons'], 
                        key=lambda x: x['f1_score']) if comparison['phase_comparisons'] else None
        
        if best_phase:
            comparison['best_phase'] = best_phase['experiment_name']
            comparison['best_f1_improvement'] = best_phase['f1_improvement']
        
        return comparison
    
    def generate_recommendations(self, comparison: Dict) -> Dict:
        """비교 결과 기반 권장사항 생성"""
        ic("권장사항 생성")
        
        recommendations = {
            'safe_phases': [],
            'promising_phases': [],
            'risky_phases': [],
            'next_actions': [],
            'full_training_candidate': None
        }
        
        if 'phase_comparisons' not in comparison:
            return recommendations
        
        for phase in comparison['phase_comparisons']:
            if phase['performance_safe']:
                recommendations['safe_phases'].append(phase['experiment_name'])
                
                if phase['recommended_for_full_training']:
                    recommendations['promising_phases'].append(phase['experiment_name'])
            else:
                recommendations['risky_phases'].append(phase['experiment_name'])
        
        # 풀 훈련 후보 선정
        promising_phases = [p for p in comparison['phase_comparisons'] 
                          if p['recommended_for_full_training']]
        
        if promising_phases:
            best_promising = max(promising_phases, key=lambda x: x['f1_improvement'])
            recommendations['full_training_candidate'] = best_promising['experiment_name']
            
            recommendations['next_actions'].extend([
                f"'{best_promising['experiment_name']}' 설정으로 풀 훈련 실행 권장",
                f"예상 F1 개선: {best_promising['f1_improvement']:.4f} ({best_promising['f1_improvement_percent']:.1f}%)",
                "기존 79% 성능 대비 안전한 개선 예상"
            ])
        else:
            recommendations['next_actions'].extend([
                "현재 단계에서는 큰 개선이 어려움",
                "더 세밀한 증강 튜닝 또는 다른 접근법 필요",
                "기존 설정 유지를 권장"
            ])
        
        return recommendations
    
    def save_experiment_results(self, results: List[Dict], comparison: Dict, 
                              recommendations: Dict):
        """실험 결과 저장"""
        ic("실험 결과 저장")
        
        # 전체 결과 저장
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_checkpoint': self.baseline_checkpoint,
            'phase_results': results,
            'performance_comparison': comparison,
            'recommendations': recommendations
        }
        
        with open(self.experiment_dir / 'experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 요약 보고서 생성
        report_path = self.experiment_dir / 'experiment_summary.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 보수적 증강 테스트 결과 보고서\n")
            f.write("=" * 50 + "\n\n")
            
            # 기준점 성능
            if 'baseline_performance' in comparison:
                baseline = comparison['baseline_performance']
                f.write(f"## 기준점 성능\n")
                f.write(f"- F1 Score: {baseline['f1']:.4f}\n")
                f.write(f"- Accuracy: {baseline['accuracy']:.4f}\n\n")
            
            # 단계별 결과
            f.write("## 단계별 결과\n")
            for phase in comparison.get('phase_comparisons', []):
                f.write(f"### {phase['experiment_name']}\n")
                f.write(f"- F1 Score: {phase['f1_score']:.4f} (개선: {phase['f1_improvement']:+.4f})\n")
                f.write(f"- Accuracy: {phase['accuracy']:.4f} (개선: {phase['accuracy_improvement']:+.4f})\n")
                f.write(f"- 안전성: {'✅' if phase['performance_safe'] else '❌'}\n")
                f.write(f"- 풀 훈련 권장: {'✅' if phase['recommended_for_full_training'] else '❌'}\n\n")
            
            # 권장사항
            f.write("## 권장사항\n")
            for action in recommendations.get('next_actions', []):
                f.write(f"- {action}\n")
            
            if recommendations.get('full_training_candidate'):
                f.write(f"\n**추천 설정**: {recommendations['full_training_candidate']}\n")
        
        ic(f"실험 결과 저장 완료: {self.experiment_dir}")
        return str(report_path)
    
    def run_conservative_augmentation_test(self, baseline_checkpoint: str = None, 
                                         quick_epochs: int = 3):
        """보수적 증강 테스트 실행"""
        ic("보수적 증강 테스트 시작")
        
        if baseline_checkpoint:
            self.baseline_checkpoint = baseline_checkpoint
        
        # 1. 점진적 증강 설정 생성
        progressive_configs = self.create_progressive_augmentation_configs()
        
        # 2. 각 단계별 빠른 검증
        results = []
        for config in progressive_configs:
            try:
                result = self.run_quick_validation(config, epochs=quick_epochs)
                results.append(result)
                
                # 중간 결과 출력
                ic(f"Phase: {config['experiment_name']}")
                ic(f"Val F1: {result.get('final_val_f1', 0):.4f}")
                
            except Exception as e:
                ic(f"오류 발생: {config['experiment_name']} - {e}")
                results.append({
                    'experiment_name': config['experiment_name'],
                    'error': str(e)
                })
        
        # 3. 성능 비교
        comparison = self.compare_with_baseline(results)
        
        # 4. 권장사항 생성
        recommendations = self.generate_recommendations(comparison)
        
        # 5. 결과 저장
        summary_path = self.save_experiment_results(results, comparison, recommendations)
        
        ic("보수적 증강 테스트 완료")
        ic(f"결과 보고서: {summary_path}")
        
        return {
            'results': results,
            'comparison': comparison,
            'recommendations': recommendations,
            'summary_report': summary_path,
            'experiment_dir': str(self.experiment_dir)
        }


def main():
    """Fire CLI 인터페이스"""
    fire.Fire(ConservativeAugmentationTester)


if __name__ == "__main__":
    main()