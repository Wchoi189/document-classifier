import torch
import time
import pandas as pd
import os
from tqdm import tqdm

from utils.metrics import calculate_metrics
from utils.utils import EarlyStopping

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
        self.history = []

    def _train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        
        for data, target in tqdm(self.train_loader, desc="Training"):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
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
        
        for epoch in range(1, self.config['train']['epochs'] + 1):
            train_loss = self._train_epoch()
            val_loss, val_acc, val_f1 = self._validate_epoch()
            
            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch}/{self.config['train']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            self.history.append({
                'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                'val_acc': val_acc, 'val_f1': val_f1
            })
            
            early_stopping(val_f1 if es_config['metric'] == 'val_f1' else val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        total_time = time.time() - start_time
        print(f"✨ Training finished in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(log_dir, 'training_log.csv'), index=False)
