from pathlib import Path
import torch
import wandb
import numpy as np
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
import time
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.utils.metrics import calculate_metrics
from src.utils.utils import EarlyStopping


def get_gpu_memory_info():
    """GPU 메모리 사용량 정보 가져오기"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB  
        return {
            'allocated': memory_allocated,
            'reserved': memory_reserved, 
            'total': memory_total,
            'free': memory_total - memory_reserved
        }
    return None

class WandBTrainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.batch_step = 0

        # 🔧 FIXED: Hydra config 호환성 개선
        # WandB 활성화 체크 - 환경 변수와 config 모두 확인
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        config_wandb_enabled = config.get('wandb', {}).get('enabled', True)  # 🔧 기본값 True로 변경
        
        # WandB 활성화 조건 개선
        self.wandb_enabled = (wandb_mode != "disabled") and config_wandb_enabled
        
        if self.wandb_enabled:
            self._init_wandb()
        else:
            print("🚫 WandB logging disabled - Check config.yaml wandb.enabled setting")

        # 훈련 설정
        self.use_autocast = config.get('train', {}).get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_autocast else None
        self.history = []
        self.memory_log = []
        self.class_names = self._get_class_names()
        
    def _init_wandb(self):
        """🔧 ENHANCED: WandB 초기화 - Hydra 실험 정보 포함"""
        wandb_config = self.config['wandb']
        wandb_username = self.config.get('wandb', {}).get('username', 'default-user')
        
        # ✅ FIXED: Use correct config structure
        experiment_info = self.config.get('experiment', {})
        experiment_name = experiment_info.get('name', 'hydra_experiment')
        experiment_description = experiment_info.get('description', 'Hydra managed experiment')
        experiment_tags = experiment_info.get('tags', [])
        
        # ✅ FIXED: Get augmentation from correct location
        model_name = self.config.get('model', {}).get('name', 'unknown')
        batch_size = self.config.get('train', {}).get('batch_size', 32)
        image_size = self.config.get('data', {}).get('image_size', 224)
        
        # ✅ FIXED: Get augmentation config from top level, not nested in data
        augmentation_config = self.config.get('augmentation', {})
        augmentation_enabled = augmentation_config.get('enabled', False)
        
        # ✅ FIXED: Only use strategy if augmentation is enabled
        if augmentation_enabled:
            augmentation_strategy = augmentation_config.get('strategy', 'basic')
            augmentation_intensity = augmentation_config.get('intensity', 0.7)
        else:
            augmentation_strategy = 'none'
            augmentation_intensity = 0.0
        
        # 성능 점수 플레이스홀더가 포함된 실행 이름
        run_name = f"{wandb_username}--{experiment_name}-{model_name}-{augmentation_strategy}-b{batch_size}-s{image_size}-(f1_pending)"
        
        # 🔧 Config 평면화 - WandB용 설정 준비
        flat_config = self._flatten_config(self.config)
        
        # ✅ FIXED: Use actual config values, not defaults
        flat_config.update({
            'augmentation_strategy': augmentation_strategy,
            'augmentation_intensity': augmentation_intensity,
            'augmentation_enabled': augmentation_enabled,
            'experiment_name': experiment_name,  # Add this for tracking
            'experiment_description': experiment_description,
        })
        
        # WandB 모드 확인
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        allowed_modes = ['online', 'offline', 'disabled']
        wandb_mode_literal = wandb_mode if wandb_mode in allowed_modes else 'online'
        
        # 🔧 WandB 초기화 - 향상된 메타데이터 포함
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            name=run_name,
            tags=list(set(wandb_config.get('tags', []) + experiment_tags)),  # 태그 병합
            notes=f"{experiment_description} | Model: {model_name} | Strategy: {augmentation_strategy}",
            config=flat_config,
            mode=wandb_mode_literal,
        )

        # 모델 감시 설정
        if wandb_config.get('watch_model', True):
            wandb.watch(self.model, log='all', log_freq=wandb_config.get('log_frequency', 10))
    
        print(f"🚀 WandB initialized: {run_name}")
        print(f"📊 Mode: {wandb_mode}, Project: {wandb_config['project']}")
        print(f"🎯 Experiment: {experiment_name}")
        print(f"🎨 Augmentation: {augmentation_strategy} (intensity: {augmentation_intensity})") 

    def _flatten_config(self, config, parent_key='', sep='_'):
        """중첩된 config를 WandB용으로 평면화"""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _get_class_names(self):
        """클래스 이름 가져오기"""
        if hasattr(self.train_loader.dataset, 'classes'):
            return self.train_loader.dataset.classes
        else:
            # 메타 파일에서 클래스 이름 가져오기 시도
            try:
                import pandas as pd
                meta_file = self.config.get('data', {}).get('meta_file', 'data/raw/metadata/meta.csv')
                if os.path.exists(meta_file):
                    meta_df = pd.read_csv(meta_file)
                    return meta_df['class_name'].tolist()
            except:
                pass
            # 폴백: 숫자 클래스명
            return [f"class_{i}" for i in range(17)]
    
    def _log_sample_predictions(self, epoch, num_samples=8):
        """샘플 예측 이미지를 WandB에 로깅"""
        if not self.wandb_enabled or not self.config['wandb'].get('log_images', False):
            return
            
        self.model.eval()
        samples_logged = 0
        wandb_images = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                if samples_logged >= num_samples:
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # 텐서를 numpy로 변환하여 로깅
                for i in range(min(data.size(0), num_samples - samples_logged)):
                    # 이미지 텐서를 numpy로 변환 (C, H, W) -> (H, W, C)
                    img_np = data[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # 이미지 역정규화
                    mean = np.array(self.config['data']['mean'])
                    std = np.array(self.config['data']['std'])
                    img_np = img_np * std + mean
                    img_np = np.clip(img_np, 0, 1)
                    
                    true_class = self.class_names[target[i].item()]
                    pred_class = self.class_names[int(predicted[i].item())]
                    conf = confidence[i].item()
                    
                    # 캡션 생성
                    caption = f"True: {true_class}\nPred: {pred_class}\nConf: {conf:.3f}"
                    
                    wandb_images.append(wandb.Image(
                        img_np,
                        caption=caption
                    ))
                    
                    samples_logged += 1
                    if samples_logged >= num_samples:
                        break
        
        if wandb_images:
            wandb.log({"val/sample_predictions": wandb_images}, step=self.batch_step)     

    def _log_confusion_matrix(self, y_true, y_pred, epoch):
        """혼동 매트릭스를 WandB에 로깅"""
        if not self.wandb_enabled or not self.config['wandb'].get('log_confusion_matrix', False):
            return
            
        cm = confusion_matrix(y_true, y_pred)
        
        # matplotlib 그림 생성
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        wandb.log({
            "val/confusion_matrix_plot": wandb.Image(plt),
            "val/confusion_matrix_data": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=self.class_names
            )
        }, step=self.batch_step)

        plt.close()
    
    def _train_epoch(self, epoch):
        """🔧 ENHANCED: 훈련 에포크 - 향상된 WandB 로깅"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        log_freq = self.config['wandb'].get('log_frequency', 10) if self.wandb_enabled else 100
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch}")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_autocast:
                with autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target)
                    
                assert isinstance(loss, torch.Tensor), "Loss function must return a torch.Tensor"
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
            else:
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                
                # 그래디언트 클리핑
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            self.batch_step += 1
            
            # 🔧 통합 배치 로깅
            if self.wandb_enabled and batch_idx % log_freq == 0:
                batch_log_data = {
                    "batch/step": self.batch_step,
                    "batch/loss": loss.item(),
                    "batch/learning_rate": self.optimizer.param_groups[0]['lr']
                }
                
                # GPU 메모리 정보 추가
                mem_info = get_gpu_memory_info()
                if mem_info:
                    batch_log_data["batch/gpu_mem_alloc_gb"] = mem_info['allocated']
                    batch_log_data["batch/gpu_mem_reserved_gb"] = mem_info['reserved']
                    
                # WandB에 로깅
                wandb.log(batch_log_data, step=self.batch_step)

            # 메모리 정리
            del data, target, output, loss
            if batch_count % 20 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self, epoch):
        """검증 에포크 - 상세 메트릭 포함"""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss = total_loss / len(self.val_loader)
        metrics = calculate_metrics(all_targets, all_preds)
        
        return val_loss, metrics['accuracy'], metrics['f1'], all_targets, all_preds
    
    def train(self):
        """🔧 ENHANCED: 향상된 훈련 루프 - Hydra + WandB 통합"""
        
        # 경로 설정 - Hydra config에서 가져오기
        paths_config = self.config.get('paths', {})
        output_dir = Path(paths_config.get('output_dir', 'outputs'))
        log_dir = output_dir / paths_config.get('log_dir', 'logs')
        model_dir = output_dir / paths_config.get('model_dir', 'models')   
         
        # 디렉토리 생성
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)        

        # Early Stopping 설정
        es_config = self.config['train']['early_stopping']
        best_model_path = model_dir / 'best_model.pth'
    
        early_stopping = EarlyStopping(
            patience=es_config['patience'],
            verbose=True,
            path=best_model_path,
            mode=es_config['mode'],
            metric=es_config['metric']
        )

        start_time = time.time()
        print("🚀 Training has started!")
        
        # 🔧 모델 메트릭 로깅
        if self.wandb_enabled:
            model_metrics = {
                "model/total_parameters": sum(p.numel() for p in self.model.parameters()),
                "model/trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "model/size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
            }
            wandb.log(model_metrics)

        best_f1 = 0
        best_epoch = 0
        epoch = 0

        for epoch in range(1, self.config['train']['epochs'] + 1):
            epoch_start_time = time.time()
            
            # 훈련 및 검증
            train_loss = self._train_epoch(epoch)
            val_loss, val_acc, val_f1, y_true, y_pred = self._validate_epoch(epoch)
            
            # 스케줄러 스텝
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # 콘솔 로깅
            print(f"Epoch {epoch}/{self.config['train']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                  f"Time: {epoch_time:.1f}s")
                    
            # 에포크 체크포인트 저장
            epoch_checkpoint_path = model_dir / f"model_epoch_{epoch}.pth"
            torch.save(self.model.state_dict(), epoch_checkpoint_path)        

            # 최신 모델 상태 저장
            last_model_path = model_dir / 'last_model.pth'
            torch.save(self.model.state_dict(), last_model_path)
            
            # 🔧 WandB 에포크 로깅 - 향상된 메트릭
            if self.wandb_enabled:
                epoch_log_data = {
                    "train/epoch": epoch,
                    "train/loss": train_loss,
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/f1": val_f1,
                    "train/epoch_time": epoch_time
                }

                wandb.log(epoch_log_data, step=self.batch_step)

                # 5 에포크마다 추가 시각화
                if epoch % 5 == 0 or epoch == self.config['train']['epochs']:
                    self._log_sample_predictions(epoch)
                    self._log_confusion_matrix(y_true, y_pred, epoch)

            # 최고 모델 추적
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                
                # WandB에 최고 모델 저장
                if self.wandb_enabled and self.config['wandb'].get('log_model', False):
                    model_artifact = wandb.Artifact(
                        name=f"model_epoch_{epoch}",
                        type="model",
                        description=f"Best model at epoch {epoch} with F1: {val_f1:.4f}"
                    )
                    
                    torch.save(self.model.state_dict(), "temp_best_model.pth")
                    model_artifact.add_file("temp_best_model.pth")
                    wandb.log_artifact(model_artifact)
                    os.remove("temp_best_model.pth")
            
            # 히스토리 저장
            self.history.append({
                'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                'val_acc': val_acc, 'val_f1': val_f1, 'epoch_time': epoch_time
            })
            
            # 메모리 정보 로깅
            mem_info = get_gpu_memory_info()
            if mem_info:
                self.memory_log.append({
                    'epoch': epoch,
                    'memory_allocated': mem_info['allocated'],
                    'memory_free': mem_info['free']
                })
            
            # Early stopping
            early_stopping(val_f1 if es_config['metric'] == 'val_f1' else val_loss, self.model)
            
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break       

        total_time = time.time() - start_time
        print(f"✨ Training finished in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"🏆 Best F1: {best_f1:.4f} at epoch {best_epoch}")
        
        # 최종 모델 저장
        last_model_path = model_dir / 'last_model.pth'
        torch.save(self.model.state_dict(), last_model_path)
        print(f"💾 Final model state saved to: {last_model_path}")
        
        # 🔧 WandB 최종 요약 및 실행 이름 업데이트
        if self.wandb_enabled:
            if wandb.run is not None and getattr(wandb.run, "summary", None) is not None:
                
                # 실행 이름을 F1 점수로 업데이트
                original_name = wandb.run.name
                if original_name is not None:
                    new_name = original_name.replace("(f1_pending)", f"f1-{best_f1:.4f}")
                    wandb.run.name = new_name
                
                # 요약 메트릭 설정
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary["best_val_f1"] = best_f1
                wandb.run.summary["total_training_time_sec"] = total_time
                wandb.run.summary["total_epochs_run"] = epoch

                if early_stopping.early_stop:
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["early_stop_epoch"] = epoch

                # 최고 모델 아티팩트 업로드
                if self.config['wandb'].get('log_model', False):
                    model_artifact = wandb.Artifact(
                        name=f"best-model-{wandb.run.id}",
                        type="model",
                        description=f"Best model with F1: {best_f1:.4f} at epoch {best_epoch}"
                    )
                    model_artifact.add_file(early_stopping.path)
                    wandb.log_artifact(model_artifact)

                # 훈련 히스토리 테이블 로깅
                history_df = pd.DataFrame(self.history)
                wandb.log({"training_history_table": wandb.Table(dataframe=history_df)})

                wandb.finish() 

        # 로컬 로그 저장
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(log_dir / 'training_log.csv', index=False)
        
        if self.memory_log:
            memory_df = pd.DataFrame(self.memory_log)
            memory_df.to_csv(log_dir / 'memory_log.csv', index=False)
            print(f"💾 Memory log saved to {log_dir / 'memory_log.csv'}")
        

    def _log_training_efficiency(self, epoch, epoch_time, train_loss, val_loss):
        """훈련 효율성 메트릭 로깅"""
        
        # 효율성 메트릭 계산
        samples_per_second = len(self.train_loader.dataset) / epoch_time
        loss_improvement = self.history[-2]['train_loss'] - train_loss if len(self.history) > 1 else 0
        
        # 메모리 효율성
        if torch.cuda.is_available():
            memory_efficiency = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            wandb.log({
                "training_samples_per_second": samples_per_second,
                "loss_improvement_rate": loss_improvement,
                "memory_efficiency": memory_efficiency,
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            })