import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_
import time
import pandas as pd
import os
from tqdm import tqdm

from src.utils.metrics import calculate_metrics
from src.utils.utils import EarlyStopping

def get_gpu_memory_info():
    """Get current GPU memory usage"""
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

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Initialize autocast for mixed precision training if enabled
        self.use_autocast = config.get('train', {}).get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_autocast else None
        self.history = []
        self.memory_log = []  # Add memory logging

    def _train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for data, target in tqdm(self.train_loader, desc="Training"):
            # Log memory before processing
            if batch_count % 10 == 0:  # Log every 10 batches
                mem_info = get_gpu_memory_info()
                if mem_info:
                    tqdm.write(f"Batch {batch_count}: GPU Memory - "
                              f"Allocated: {mem_info['allocated']:.2f}GB, "
                              f"Free: {mem_info['free']:.2f}GB")
                    
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)

            if torch.isnan(loss):
                print("Encountered NaN loss, skipping this batch.")
                continue

            loss.backward()
            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1

            # Memory cleanup every batch
            del data, target, output, loss
            if batch_count % 20 == 0:  # Clear cache every 20 batches
                torch.cuda.empty_cache()

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        """Validates the model for one epoch."""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss = total_loss / len(self.val_loader)
        metrics = calculate_metrics(all_targets, all_preds)
        
        return val_loss, metrics['accuracy'], metrics['f1']

    def train(self):
        """Runs the full training loop."""
        log_dir = self.config['logging']['log_dir']
        checkpoint_dir = self.config['logging']['checkpoint_dir']
        
        es_config = self.config['train']['early_stopping']
        early_stopping = EarlyStopping(
            patience=es_config['patience'],
            verbose=True,
            path=os.path.join(checkpoint_dir, 'best_model.pth'),
            mode=es_config['mode'],
            metric=es_config['metric']
        )

        start_time = time.time()
        print("🚀 Training has started!")
        
        # Log initial memory state
        initial_mem = get_gpu_memory_info()
        if initial_mem:
            print(f"Initial GPU Memory: {initial_mem['allocated']:.2f}GB allocated, "
                  f"{initial_mem['free']:.2f}GB free")
        
        epochs = self.config['train']['epochs']
        if isinstance(epochs, str):
            raise ValueError(f"Expected integer for epochs but got string: '{epochs}'. Check your config file.")
        for epoch in range(1, int(epochs) + 1):
            # Log memory at start of epoch
            epoch_start_mem = get_gpu_memory_info()
            
            train_loss = self._train_epoch()
            val_loss, val_acc, val_f1 = self._validate_epoch()
            
            # Log memory at end of epoch
            epoch_end_mem = get_gpu_memory_info()
            
            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch}/{self.config['train']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            if epoch_start_mem and epoch_end_mem:
                memory_change = epoch_end_mem['allocated'] - epoch_start_mem['allocated']
                print(f"Memory: {epoch_end_mem['allocated']:.2f}GB allocated "
                      f"({memory_change:+.3f}GB change)")
            
            self.history.append({
                'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                'val_acc': val_acc, 'val_f1': val_f1
            })
            
            # Log memory info
            if epoch_end_mem:
                self.memory_log.append({
                    'epoch': epoch,
                    'memory_allocated': epoch_end_mem['allocated'],
                    'memory_free': epoch_end_mem['free']
                })
            
            early_stopping(val_f1 if es_config['metric'] == 'val_f1' else val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        total_time = time.time() - start_time
        print(f"✨ Training finished in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        
        # Save logs
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(log_dir, 'training_log.csv'), index=False)
        
        if self.memory_log:
            memory_df = pd.DataFrame(self.memory_log)
            memory_df.to_csv(os.path.join(log_dir, 'memory_log.csv'), index=False)
            print(f"💾 Memory log saved to {os.path.join(log_dir, 'memory_log.csv')}")
