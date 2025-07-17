#!/usr/bin/env python3
"""
Training script with FIXED path handling and dynamic config support
"""
import sys
import os
import icecream as ic
from pathlib import Path

# 🔧 FIX: Change directory BEFORE Hydra decorator
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # ← This must happen BEFORE @hydra.main



from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.augmentation import get_configurable_transforms, get_train_transforms, get_valid_transforms

from src.utils.config_utils import print_config_summary, normalize_config_structure
from src.utils.utils import set_seed, load_config
from src.data.csv_dataset import CSVDocumentDataset
from src.models.model import create_model
from src.trainer.trainer import Trainer
from src.trainer.wandb_trainer import WandBTrainer
import pandas as pd
from src.inference.predictor import predict_from_checkpoint
from src.data.csv_dataset import is_cross_phase_config, create_cross_phase_datasets

# config_path=str(project_root / "configs")
@hydra.main(version_base="1.2", config_path="../configs")
def main(cfg: DictConfig) -> None:
  
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")


    """
    Main training function with Hydra configuration management.
    
    🔧 FIXED: Now properly handles experiment configs
    
    Examples:
        # Use default config.yaml
        python scripts/train.py
        
        # Use experiment config (will merge with config.yaml)
        python scripts/train.py experiment=document_classifier_0701
        
        # Override specific parameters  
        python scripts/train.py experiment=document_classifier_0701 train.batch_size=16
        
        # Quick test run
        python scripts/train.py experiment=quick_debug
    """
    
  

    print("🚀 Starting training with Hydra configuration management")
    
    # 🔧 FIXED: Better experiment info extraction
    experiment_name = cfg.get('experiment', {}).get('name', 'default_experiment')
    experiment_description = cfg.get('experiment', {}).get('description', 'No description')
    experiment_tags = cfg.get('experiment', {}).get('tags', [])
    
    print(f"📋 Experiment: {experiment_name}")
    print(f"📝 Description: {experiment_description}")
    print(f"🏷️  Tags: {experiment_tags}")
    
    # 🔧 FIXED: Debug config merging
    print("\n🔧 CONFIG DEBUG INFO:")
    print(f"   Batch Size from config: {cfg.get('train', {}).get('batch_size', 'NOT_SET')}")
    print(f"   Learning Rate from config: {cfg.get('optimizer', {}).get('learning_rate', 'NOT_SET')}")
    print(f"   Epochs from config: {cfg.get('train', {}).get('epochs', 'NOT_SET')}")
    print(f"   Augmentation enabled: {cfg.get('augmentation', {}).get('enabled', 'NOT_SET')}")
    
    # Use your config utilities to properly handle the configuration
    config_ = load_config(cfg)  # This handles the conversion safely
    config = normalize_config_structure(config_)  # Normalize structure
    # ✅ FIXED: Convert Hydra config to dict WITHOUT overriding
    # config = OmegaConf.to_container(cfg, resolve=True)
    # Print configuration summary
    print_config_summary(config)
    
    # 🔧 VERIFICATION: Print final config values
    print("\n✅ FINAL CONFIG VERIFICATION:")
    print(f"   ✅ Batch Size: {config['train']['batch_size']}")
    print(f"   ✅ Learning Rate: {config['optimizer']['learning_rate']}")
    print(f"   ✅ Epochs: {config['train']['epochs']}")
    print(f"   ✅ Augmentation: {config.get('augmentation', {}).get('enabled', 'undefined')}")
    
    # --- Rest of your existing training code stays the same ---
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # --- 2. Data Preparation ---
    # Handle augmentation choice with new configuration system
    
    # Handle augmentation choice with new configuration system
    augmentation_config = config['augmentation']
    augmentation_enabled = augmentation_config['enabled']
    augmentation_strategy = augmentation_config['strategy']

    print(f"🎨 Using augmentation strategy: {augmentation_strategy}")
    print(f"📊 Augmentation intensity: {augmentation_config.get('intensity', 0.7)}")

    if augmentation_enabled and augmentation_strategy in ['document', 'robust']:
        train_transforms = get_configurable_transforms(
            height=config['data']['image_size'], 
            width=config['data']['image_size'],
            mean=config['data']['mean'], 
            std=config['data']['std'],
            config=augmentation_config
        )
        print(f"✅ Using configurable {augmentation_strategy} augmentation")
    else:
        train_transforms = get_train_transforms(
            height=config['data']['image_size'], 
            width=config['data']['image_size'],
            mean=config['data']['mean'], 
            std=config['data']['std']
        )
        print("✅ Using basic/no augmentation")       

    valid_transforms = get_valid_transforms(
        height=config['data']['image_size'], 
        width=config['data']['image_size'],
        mean=config['data']['mean'], 
        std=config['data']['std']
    )

    # 🔧 ENHANCED: Detect and handle cross-phase validation
    if is_cross_phase_config(config):
        print("🔄 Cross-phase validation detected!")
        print(f"📊 Training on: {config['data']['root_dir']}")
        print(f"🎯 Validating on: {config['data'].get('val_root_dir', 'separate dataset')}")
        
        # Create cross-phase datasets
        train_dataset, val_dataset = create_cross_phase_datasets(
            config, train_transforms, valid_transforms
        )
        
    else:
        print("📊 Standard single-dataset validation")
        
        # Create standard datasets
        train_dataset = CSVDocumentDataset(
            root_dir=config['data']['root_dir'],
            csv_file=config['data']['csv_file'],
            meta_file=config['data']['meta_file'],
            split='train', 
            transform=train_transforms,
            val_size=config['data']['val_size'],
            seed=config['seed']
        )

        val_dataset = CSVDocumentDataset(
            root_dir=config['data']['root_dir'], 
            csv_file=config['data']['csv_file'],
            meta_file=config['data']['meta_file'],
            split='val', 
            transform=valid_transforms,
            val_size=config['data']['val_size'],
            seed=config['seed']
        )


    # 🔧 Dataset verification and info
    print(f"\n📊 Dataset Information:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes[:5]}..." if len(train_dataset.classes) > 5 else f"Classes: {train_dataset.classes}")

    # Verify cross-phase validation is working
    if hasattr(train_dataset, 'is_cross_phase') and train_dataset.is_cross_phase:
        print("✅ Cross-phase validation 활성화됨")
        train_info = train_dataset.get_info()
        val_info = val_dataset.get_info()
        print(f"   훈련 데이터: {train_info['dataset_size']}개 샘플")
        print(f"   검증 데이터: {val_info['dataset_size']}개 샘플 (다른 단계)")
    else:
        print("📊 Standard validation 사용 중")

    # Create data loaders (rest remains the same)
    num_workers = config['data']['num_workers'] if config['data']['num_workers'] > 0 else 0

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        multiprocessing_context='spawn' if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        multiprocessing_context='spawn' if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    # Debug: Check sample batch for cross-phase validation
    print("\n--- Enhanced Sample Batch Debug ---")
    try:
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch
        print(f"✅ Training batch loaded successfully")
        print(f"Image tensor shape: {images.shape}")
        print(f"Labels range: [{labels.min()}, {labels.max()}]")
        
        # Test validation batch
        val_sample_batch = next(iter(val_loader))
        val_images, val_labels = val_sample_batch
        print(f"✅ Validation batch loaded successfully")
        print(f"Val image tensor shape: {val_images.shape}")
        print(f"Val labels range: [{val_labels.min()}, {val_labels.max()}]")
        
        if hasattr(train_dataset, 'is_cross_phase') and train_dataset.is_cross_phase:
            print("🔄 Cross-phase validation confirmed working!")
        
    except Exception as e:
        print(f"❌ Error loading sample batch: {e}")
        return

    # --- 3. Model, Loss, Optimizer, Scheduler ---
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes[:5]}..." if len(train_dataset.classes) > 5 else f"Classes: {train_dataset.classes}")
    
    # Debug: Check a sample batch
    print("\n--- Sample Batch Debug ---")
    try:
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch
        print(f"✅ Batch loaded successfully")
        print(f"Image tensor shape: {images.shape}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Image mean: {images.mean():.3f}, std: {images.std():.3f}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels[:10]}")
        print(f"Label range: [{labels.min()}, {labels.max()}]")
        
        if labels.max() >= num_classes:
            print(f"⚠️  WARNING: Found label {labels.max()} but only {num_classes} classes!")
        else:
            print(f"✅ Labels are in correct range [0, {num_classes-1}]")
            
    except Exception as e:
        print(f"❌ Error loading sample batch: {e}")
        return
    
    # Create model
    model = create_model(
        model_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    ).to(device)

    print(f"✅ Model created: {config['model']['name']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 🔧 NEW: Checkpoint loading support
    if 'load_checkpoint' in config['train']:
        checkpoint_path = config['train']['load_checkpoint']
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"📥 체크포인트 로드 중: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint)
                print(f"✅ 체크포인트 로드 성공: {checkpoint_path}")
            except Exception as e:
                print(f"❌ 체크포인트 로드 실패: {e}")
                print("⚠️ 사전훈련된 가중치로 계속 진행")
        else:
            print(f"⚠️ 체크포인트 파일 없음: {checkpoint_path}")
            print("⚠️ 사전훈련된 가중치로 계속 진행")

    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer - handle both old and new config formats
    optimizer_config = config['optimizer']
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
    elif optimizer_config['name'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=optimizer_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay'],
            momentum=optimizer_config.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    # Scheduler
    scheduler = None
    scheduler_config = config['scheduler']
    if scheduler_config['name'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config['T_0'],
            T_mult=scheduler_config['T_mult'],
            eta_min=scheduler_config['eta_min']
        )
    elif scheduler_config['name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )
    elif scheduler_config['name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )

# 🔧 DEBUG: Print actual config values being used
    print("\n🔧 FINAL CONFIG DEBUG:")
    print(f"   Batch Size: {config['train']['batch_size']}")
    print(f"   Learning Rate: {config['optimizer']['learning_rate']}")
    print(f"   Epochs: {config['train']['epochs']}")
    print(f"   Augmentation: {config.get('augmentation', {}).get('enabled', 'NOT_SET')}")
    print(f"   Use Doc Aug: {config['data'].get('use_document_augmentation', 'NOT_SET')}")

    # --- 4. Training with WandB ---
    trainer = WandBTrainer(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config)
    trainer.train()

    print(f"\n🎉 Training completed for experiment: {experiment_name}")
    print(f"📊 Check your results in the outputs directory")



if __name__ == '__main__':
    main()