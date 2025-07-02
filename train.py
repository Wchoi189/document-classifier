import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import wandb
from data.csv_dataset import CSVDocumentDataset
from utils.utils import load_config, set_seed
from data.augmentation import get_train_transforms, get_valid_transforms, get_document_transforms
from models.model import create_model
from trainer.trainer import Trainer
# from trainer.wandb_trainer import WandBTrainer

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
    # A single call now handles all training augmentations based on your config
    # train_transforms = create_document_transforms(
    #     config=config['augmentations'],          # Pass the detailed augmentation config section
    #     height=config['data']['image_size'],
    #     width=config['data']['image_size'],
    #     mean=config['data']['mean'],
    #     std=config['data']['std']
    # )

    # # The validation transforms remain the same
    # valid_transforms = get_valid_transforms(
    #     height=config['data']['image_size'],
    #     width=config['data']['image_size'],
    #     mean=config['data']['mean'],
    #     std=config['data']['std']
    # )


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
        print(f"Labels: {labels[:10]}")  # First 10 labels
        print(f"Label range: [{labels.min()}, {labels.max()}]")
        
        # Verify labels are in correct range
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
    loss_fn = getattr(nn, config['train']['loss'])()
    
    # Optimizer
    optimizer = getattr(optim, config['train']['optimizer'])(
        model.parameters(), 
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Scheduler (optional)
    scheduler = None
    if config['train'].get('scheduler'):
        scheduler_class = getattr(optim.lr_scheduler, config['train']['scheduler'])
        scheduler = scheduler_class(optimizer, **config['train']['scheduler_params'])

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