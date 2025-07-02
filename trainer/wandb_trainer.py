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
from utils.metrics import calculate_metrics
from utils.utils import EarlyStopping

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
        self.batch_step = 0 # Add this counter

        # Initialize WandB
        self.wandb_enabled = config.get('wandb', {}).get('enabled', False)
        if self.wandb_enabled:
            self._init_wandb()
        
        # Training setup
        self.use_autocast = config.get('train', {}).get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_autocast else None
        self.history = []
        self.memory_log = []
        self.class_names = self._get_class_names()
        
    def _init_wandb(self):
        """Initialize WandB run"""
        wandb_config = self.config['wandb']

        # --- 1. Dynamically create the run name ---
        model_name = self.config['model']['name']
        batch_size = self.config['train']['batch_size']
        image_size = self.config['data']['image_size']
        username = wandb_config.get('username', 'user') # Get username from config or use a default

        # Create the name with a placeholder for the score

        run_name = f"{username}-(real_score)-{model_name}-b{batch_size}-s{image_size}-(f1_score)"
        
        # Flatten config for WandB
        flat_config = self._flatten_config(self.config)
        
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            name=run_name,
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config=flat_config
        )

        # --- THIS IS THE KEY FIX ---
        # Define custom x-axes. This is the most robust way.
        if self.wandb_enabled:
            # 1. For batch-level metrics
            wandb.define_metric("batch/step")
            wandb.define_metric("batch/*", step_metric="batch/step")

            # 2. For epoch-level metrics
            wandb.define_metric("epoch")
            wandb.define_metric("epoch/*", step_metric="epoch")

        # Watch model if enabled
        if wandb_config.get('watch_model', True):
            wandb.watch(self.model, log='all', log_freq=wandb_config.get('log_frequency', 10))
        
        print("✅ WandB initialized successfully")
    
    def _flatten_config(self, config, parent_key='', sep='_'):
        """Flatten nested config for WandB"""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _get_class_names(self):
        """Get class names from dataset"""
        if hasattr(self.train_loader.dataset, 'classes'):
            return self.train_loader.dataset.classes
        else:
            # Fallback to numeric classes
            return [f"class_{i}" for i in range(17)]  # From your EDA
    
    def _log_sample_predictions(self, epoch, num_samples=8):
        """Log sample predictions with images to WandB"""
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
                
                # Convert tensors to numpy for logging
                for i in range(min(data.size(0), num_samples - samples_logged)):
                    # Convert image tensor to numpy (C, H, W) -> (H, W, C)
                    img_np = data[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Denormalize image
                    mean = np.array(self.config['data']['mean'])
                    std = np.array(self.config['data']['std'])
                    img_np = img_np * std + mean
                    img_np = np.clip(img_np, 0, 1)
                    
                    true_class = self.class_names[target[i].item()]
                    pred_class = self.class_names[int(predicted[i].item())]
                    conf = confidence[i].item()
                    
                    # Create caption
                    caption = f"True: {true_class}\nPred: {pred_class}\nConf: {conf:.3f}"
                    
                    wandb_images.append(wandb.Image(
                        img_np,
                        caption=caption
                    ))
                    
                    samples_logged += 1
                    if samples_logged >= num_samples:
                        break
        
        if wandb_images:
            # commit=False tells wandb to group these logs with the next "committed" log call, which will be your main epoch log.
            wandb.log({f"predictions_epoch_{epoch}": wandb_images}, commit=False) 
        
    def _log_confusion_matrix(self, y_true, y_pred, epoch):
        """Log confusion matrix to WandB"""
        if not self.wandb_enabled or not self.config['wandb'].get('log_confusion_matrix', False):
            return
            
        cm = confusion_matrix(y_true, y_pred)
        
        # Create matplotlib figure
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
            "epoch/confusion_matrix_plot": wandb.Image(plt),
            "epoch/confusion_matrix_data": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=self.class_names
            )
        }, commit=False)
        
        plt.close()
    
    def _train_epoch(self, epoch):
        """Enhanced training epoch with WandB logging"""
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
                    
                # Add this assertion to be safe
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
                
                # Gradient clipping
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # --- CONSOLIDATED BATCH LOGGING ---
            if self.wandb_enabled and batch_idx % log_freq == 0:
                # 1. Create a single dictionary
                batch_log_data = {
                    "batch/step": self.batch_step,
                    "batch/loss": loss.item(),
                    "batch/learning_rate": self.optimizer.param_groups[0]['lr']
                }
                # 2. Add GPU info to the same dictionary
                mem_info = get_gpu_memory_info()
                if mem_info:
                    batch_log_data["batch/gpu_mem_alloc_gb"] = mem_info['allocated']

                # 3. Make a single log call
                wandb.log(batch_log_data)

            # Increment batch_step outside the if block
            self.batch_step += 1

            # Memory cleanup
            del data, target, output, loss
            if batch_count % 20 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self, epoch):
        """Enhanced validation with detailed metrics"""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        class_correct = [0] * len(self.class_names)
        class_total = [0] * len(self.class_names)
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Per-class accuracy
                correct = (preds == target)
                for i in range(target.size(0)):
                    label = target[i].item()
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
        
        val_loss = total_loss / len(self.val_loader)
        metrics = calculate_metrics(all_targets, all_preds)
        
    
        return val_loss, metrics['accuracy'], metrics['f1'], all_targets, all_preds
    
    def train(self):
        """Enhanced training loop with comprehensive WandB logging"""
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
        
        # Log model architecture to WandB
        if self.wandb_enabled:
            wandb.log({
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
            })
        
        best_f1 = 0
        best_epoch = 0
        epoch = 0  # Initialize epoch here

        for epoch in range(1, self.config['train']['epochs'] + 1):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self._train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_f1, y_true, y_pred = self._validate_epoch(epoch)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # Console logging
            print(f"Epoch {epoch}/{self.config['train']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        

            # WandB logging
            if self.wandb_enabled:
                if epoch % 5 == 0 or epoch == self.config['train']['epochs']:
                    self._log_sample_predictions(epoch)
                    self._log_confusion_matrix(y_true, y_pred, epoch)
                    

                log_data = {
                    "epoch": epoch,
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_accuracy": val_acc,
                    "epoch/val_f1": val_f1,
                    "epoch/learning_rate": self.optimizer.param_groups[0]['lr']
                }
                wandb.log(log_data)


            # Track best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                
                # Save best model to WandB
                if self.wandb_enabled and self.config['wandb'].get('log_model', False):
                    model_artifact = wandb.Artifact(
                        name=f"model_epoch_{epoch}",
                        type="model",
                        description=f"Best model at epoch {epoch} with F1: {val_f1:.4f}"
                    )
                    
                    # Save model state dict
                    torch.save(self.model.state_dict(), "temp_best_model.pth")
                    model_artifact.add_file("temp_best_model.pth")
                    wandb.log_artifact(model_artifact)
                    os.remove("temp_best_model.pth")
            
            # Store history
            self.history.append({
                'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                'val_acc': val_acc, 'val_f1': val_f1, 'epoch_time': epoch_time
            })
            
            # Log memory info
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
                if self.wandb_enabled:
                    wandb.log({"early_stopped": True, "early_stop_epoch": epoch})
                break
        

        total_time = time.time() - start_time
        print(f"✨ Training finished in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"🏆 Best F1: {best_f1:.4f} at epoch {best_epoch}")
        

        # --- AFTER THE EPOCH LOOP ---
        if self.wandb_enabled:
            if wandb.run is not None and getattr(wandb.run, "summary", None) is not None:

                # --- THIS IS THE NEW LOGIC ---
                # 1. Get the original run name
                original_name = wandb.run.name

                # 2. Create the new name with the F1 score, only if original_name is not None
                if original_name is not None:
                    new_name = original_name.replace("(f1_score)", f"{best_f1:.4f}")
                    # 3. Update the run name in WandB
                    wandb.run.name = new_name
                    # No need to call wandb.run.save() here; setting the name is sufficient
                
                # Log the final summary values to the run's summary panel, not the history
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary["best_val_f1"] = best_f1
                wandb.run.summary["total_training_time_sec"] = total_time
                wandb.run.summary["total_epochs_run"] = epoch

                # Log the history table separately
                history_df = pd.DataFrame(self.history)
                wandb.log({"training_history_table": wandb.Table(dataframe=history_df)})

                wandb.finish() 

        # Save local logs
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(log_dir, 'training_log.csv'), index=False)
        
        if self.memory_log:
            memory_df = pd.DataFrame(self.memory_log)
            memory_df.to_csv(os.path.join(log_dir, 'memory_log.csv'), index=False)
            print(f"💾 Memory log saved to {os.path.join(log_dir, 'memory_log.csv')}")
        

    def _log_training_efficiency(self, epoch, epoch_time, train_loss, val_loss):
        """Log training efficiency metrics"""
        
        # Calculate efficiency metrics
        samples_per_second = len(self.train_loader.dataset) / epoch_time
        loss_improvement = self.history[-2]['train_loss'] - train_loss if len(self.history) > 1 else 0
        
        # Memory efficiency
        if torch.cuda.is_available():
            memory_efficiency = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            wandb.log({
                "training_samples_per_second": samples_per_second,
                "loss_improvement_rate": loss_improvement,
                "memory_efficiency": memory_efficiency,
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            })

    def _log_augmentation_analysis(self, epoch):
        """Analyze augmentation effectiveness"""
        if epoch % 10 == 0:  # Log every 10 epochs
            # Sample with and without augmentation
            original_transform = self.train_loader.dataset.transform
            
            # Temporarily disable augmentation
            from data.augmentation import get_valid_transforms
            self.train_loader.dataset.transform = get_valid_transforms(
                height=self.config['data']['image_size'],
                width=self.config['data']['image_size'],
                mean=self.config['data']['mean'],
                std=self.config['data']['std']
            )        


        if self.wandb_enabled:
            wandb.finish()       