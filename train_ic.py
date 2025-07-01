import fire  # Add this import
from icecream import ic  # Add this import
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from data.csv_dataset import CSVDocumentDataset
from utils.utils import load_config, set_seed
# from data.dataset import DocumentDataset
from data.augmentation import get_train_transforms, get_valid_transforms, get_document_transforms
from models.model import create_model
from trainer.trainer import Trainer
# Remove the argparse section and replace main function:
def train_model(config='config/config.yaml', debug=False):
    """
    Train a document classification model.
    
    Args:
        config (str): Path to config YAML file
        debug (bool): Enable debug mode with icecream
    """
    if debug:
        ic.enable()
        ic("🔥 Debug mode enabled!")
    else:
        ic.disable()
    
    ic(f"Loading config from: {config}")
    
    # --- 1. Setup ---
    config_data = load_config(config)
    ic(config_data['model']['name'], config_data['train']['batch_size'])
    
    set_seed(config_data['seed'])
    device = torch.device(config_data['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(config_data['logging']['log_dir'], exist_ok=True)
    os.makedirs(config_data['logging']['checkpoint_dir'], exist_ok=True)
    
    # Print debug info
    ic(device, torch.cuda.is_available())
    if torch.cuda.is_available():
        ic(torch.cuda.get_device_name())
        ic(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- 2. Data Preparation ---
    ic("Setting up data transforms...")
    if config_data['data'].get('use_document_augmentation', False):
        train_transforms = get_document_transforms(**config_data['data'])
    else:
        train_transforms = get_train_transforms(
            height=config_data['data']['image_size'], width=config_data['data']['image_size'],
            mean=config_data['data']['mean'], std=config_data['data']['std']
        )
    valid_transforms = get_valid_transforms(
        height=config_data['data']['image_size'], width=config_data['data']['image_size'],
        mean=config_data['data']['mean'], std=config_data['data']['std']
    )

    ic("Creating datasets...")
    # Create datasets
    train_dataset = CSVDocumentDataset(
        root_dir=config_data['data']['root_dir'], 
        csv_file=config_data['data']['csv_file'],
        meta_file=config_data['data']['meta_file'],
        split='train', 
        transform=train_transforms,
        val_size=config_data['data']['val_size'],
        seed=config_data['seed']
    )
    
    val_dataset = CSVDocumentDataset(
        root_dir=config_data['data']['root_dir'], 
        csv_file=config_data['data']['csv_file'],
        meta_file=config_data['data']['meta_file'],
        split='val', 
        transform=valid_transforms,
        val_size=config_data['data']['val_size'],
        seed=config_data['seed']
    )
    
    ic(len(train_dataset), len(val_dataset))
    
    # Data loaders
    num_workers = config_data['data']['num_workers'] if config_data['data']['num_workers'] > 0 else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config_data['train']['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config_data['train']['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # --- 3. Model, Loss, Optimizer ---
    num_classes = len(train_dataset.classes)
    ic(num_classes, train_dataset.classes[:3])
    
    # Debug sample batch
    ic("Loading sample batch...")
    try:
        sample_batch = next(iter(train_loader))
        images, labels = sample_batch
        ic(images.shape, labels.shape)
        ic(images.min(), images.max(), images.mean())
        ic(labels.min(), labels.max())
    except Exception as e:
        ic(f"Error loading batch: {e}")
        return
    
    # Create model
    ic("Creating model...")
    model = create_model(
        model_name=config_data['model']['name'],
        num_classes=num_classes,
        pretrained=config_data['model']['pretrained']
    ).to(device)
    
    ic(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    loss_fn = getattr(nn, config_data['train']['loss'])()
    optimizer = getattr(optim, config_data['train']['optimizer'])(
        model.parameters(), 
        lr=config_data['train']['learning_rate'],
        weight_decay=config_data['train']['weight_decay']
    )
    
    # Scheduler
    scheduler = None
    if config_data['train'].get('scheduler'):
        scheduler_class = getattr(optim.lr_scheduler, config_data['train']['scheduler'])
        scheduler = scheduler_class(optimizer, **config_data['train']['scheduler_params'])
        ic("Scheduler created")

    # --- 4. Training ---
    ic("Starting training...")
    trainer = Trainer(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config_data)
    trainer.train()

if __name__ == '__main__':
    fire.Fire(train_model)