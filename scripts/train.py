#!/usr/bin/env python3
"""
Training script with proper path handling
"""
import sys
from pathlib import Path

# Add parent to path for setup import
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.project_setup import setup_project_environment
# Initialize at the start of your script

from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.augmentation import get_configurable_transforms, get_train_transforms, get_valid_transforms


from src.utils.config_utils import print_config_summary,normalize_config_structure
from src.utils.utils import set_seed,load_config
from src.data.csv_dataset import CSVDocumentDataset
from src.models.model import create_model
from src.trainer.trainer import Trainer
from src.trainer.wandb_trainer import WandBTrainer
import pandas as pd
from src.inference.predictor import predict_from_checkpoint

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function with Hydra configuration management.
    
    Examples:
        # Run with default config
        python scripts/train.py
        
        # Override specific parameters
        python scripts/train.py model=efficientnet train.batch_size=64
        
        # Use different experiment config
        python scripts/train.py experiment=resnet_experiment
        
        # Quick test run
        python scripts/train.py experiment=quick_test
    """
    
    setup_project_environment()

    print("🚀 Starting training with Hydra configuration management")
    print(f"📋 Experiment: {cfg.experiment.name}")
    print(f"📝 Description: {cfg.experiment.description}")
    print(f"🏷️  Tags: {cfg.experiment.tags}")
    
     # Use your config utilities to properly handle the configuration
    config = load_config(cfg)  # This handles the conversion safely
    config = normalize_config_structure(config)  # Normalize structure

    # Print configuration summary
    print_config_summary(config)
    # --- 1. Setup ---
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # --- 2. Data Preparation ---

    # Handle augmentation choice with new configuration system
    augmentation_config = config['data'].get('augmentation', {})
    augmentation_strategy = augmentation_config.get('strategy', 'basic')

    print(f"🎨 Using augmentation strategy: {augmentation_strategy}")
    print(f"📊 Augmentation intensity: {(augmentation_config, 'intensity', 0.7)}")


    if augmentation_strategy in ['document', 'robust']:
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
        print("✅ Using basic train augmentation")        
    
    valid_transforms = get_valid_transforms(
        
        height=config['data']['image_size'], 
        width=config['data']['image_size'],
        mean=config['data']['mean'], 
        std=config['data']['std']
    )

    # Create datasets using safe_get for all config accesses
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

    # Create data loaders
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

    # --- 4. Training with WandB ---
    trainer = WandBTrainer(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config)
    trainer.train()

    print(f"\n🎉 Training completed for experiment: {cfg.experiment.name}")
    print(f"📊 Check your results in the outputs directory")


if __name__ == '__main__':
    main()