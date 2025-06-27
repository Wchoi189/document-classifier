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

# Disable cuDNN to avoid illegal instruction errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

def main(config_path):
    # --- 1. Setup ---
    config = load_config(config_path)
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    print(f"Using device: {device}")

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
    
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    # --- 3. Model, Loss, Optimizer ---
    num_classes = len(train_dataset.classes)
    model = create_model(
        model_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    ).to(device)

    loss_fn = getattr(nn, config['train']['loss'])()
    optimizer = getattr(optim, config['train']['optimizer'])(model.parameters(), lr=config['train']['learning_rate'])
    
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
    main(args.config)
