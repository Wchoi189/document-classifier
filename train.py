import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from utils.utils import load_config, set_seed
from data.dataset import DocumentDataset
from data.augmentation import get_train_transforms, get_valid_transforms, get_document_transforms
from models.model import create_model
from trainer.trainer import Trainer

def main(config_path):
    # --- 1. Setup ---
    config = load_config(config_path)
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Print debug info once inside main function
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # --- 2. Data Preparation ---
    if config['data'].get('use_document_augmentation', False):
        train_transforms = get_document_transforms(**config['data'])
    else:
        train_transforms = get_train_transforms(
            height=config['data']['image_size'], width=config['data']['image_size'],
            mean=config['data']['mean'], std=config['data']['std']
        )
    valid_transforms = get_valid_transforms(
        height=config['data']['image_size'], width=config['data']['image_size'],
        mean=config['data']['mean'], std=config['data']['std']
    )

    train_dataset = DocumentDataset(root_dir=config['data']['root_dir'], split='train', transform=train_transforms)
    val_dataset = DocumentDataset(root_dir=config['data']['root_dir'], split='val', transform=valid_transforms)
    
    # Use num_workers=0 when using CUDA to avoid multiprocessing issues
    num_workers = config['data']['num_workers'] if config['data']['num_workers'] > 0 else 0
    
    # And add the multiprocessing parameters:
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
        persistent_workers=True if num_workers > 0 else False  # Optional for val_loader
    )

    # --- 3. Model, Loss, Optimizer ---
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Debug data values
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    print(f"Image tensor shape: {images.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Image mean: {images.mean():.3f}, std: {images.std():.3f}")
    print(f"Labels: {labels}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")
    
    model = create_model(
        model_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    ).to(device)

    loss_fn = getattr(nn, config['train']['loss'])()
    optimizer = getattr(optim, config['train']['optimizer'])(
        model.parameters(), 
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    scheduler = None
    if config['train'].get('scheduler'):
        scheduler = getattr(optim.lr_scheduler, config['train']['scheduler'])(
            optimizer, **config['train']['scheduler_params']
        )

    # --- 4. Training ---
    trainer = Trainer(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a document classification model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file.")
    args = parser.parse_args()
    
    try:
        main(args.config)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\n❌ CUDA Error: {e}")
            print("💡 Try these solutions:")
            print("   1. Update your GPU drivers")
            print("   2. Reinstall PyTorch with correct CUDA version")
            print("   3. Run with CPU: change 'device: cpu' in config.yaml")
            print("   4. Reduce batch_size in config.yaml")
        else:
            raise e