import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from typing import List, Any


def log_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str], epoch: int):
    """Log comprehensive metrics to WandB"""
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Log per-class metrics
    for class_name in class_names:
        class_key = class_name
        class_metrics = report.get(class_key) if isinstance(report, dict) else None
        if class_metrics and isinstance(class_metrics, dict):
            wandb.log({
                f"precision_{class_key}": class_metrics.get('precision'),
                f"recall_{class_key}": class_metrics.get('recall'),
                f"f1_{class_key}": class_metrics.get('f1-score'),
                f"support_{class_key}": class_metrics.get('support')
            }, step=epoch)
    
    # Macro and weighted averages
    macro_avg = report.get('macro avg') if isinstance(report, dict) else None
    weighted_avg = report.get('weighted avg') if isinstance(report, dict) else None
    wandb.log({
        "macro_avg_precision": macro_avg.get('precision') if macro_avg else None,
        "macro_avg_recall": macro_avg.get('recall') if macro_avg else None,
        "macro_avg_f1": macro_avg.get('f1-score') if macro_avg else None,
        "weighted_avg_precision": weighted_avg.get('precision') if weighted_avg else None,
        "weighted_avg_recall": weighted_avg.get('recall') if weighted_avg else None,
        "weighted_avg_f1": weighted_avg.get('f1-score') if weighted_avg else None
    }, step=epoch)


def log_class_imbalance_analysis(train_dataset, val_dataset):
    """Analyze and log class imbalance"""
    
    # Get class distributions
    train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    val_targets = [val_dataset[i][1] for i in range(len(val_dataset))]
    
    train_dist = np.bincount(train_targets)
    val_dist = np.bincount(val_targets)
    
    # Calculate imbalance metrics
    imbalance_ratio = train_dist.max() / train_dist.min()
    
    wandb.log({
        "class_imbalance_ratio": imbalance_ratio,
        "min_class_samples": train_dist.min(),
        "max_class_samples": train_dist.max(),
        "total_classes": len(train_dist)
    })
    
    # Log class distribution chart
    class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else [f"Class_{i}" for i in range(len(train_dist))]
    
    data = [[class_names[i], train_dist[i], val_dist[i]] for i in range(len(train_dist))]
    table = wandb.Table(data=data, columns=["Class", "Train_Count", "Val_Count"])
    
    wandb.log({
        "class_distribution": wandb.plot.bar(table, "Class", "Train_Count", title="Training Class Distribution"),
        "class_distribution_table": table
    })

def log_model_complexity_analysis(model):
    """Analyze and log model complexity metrics"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024**2
    
    # Layer analysis
    layer_count = len(list(model.modules()))
    conv_layers = len([m for m in model.modules() if isinstance(m, torch.nn.Conv2d)])
    linear_layers = len([m for m in model.modules() if isinstance(m, torch.nn.Linear)])
    
    wandb.log({
        "model_total_parameters": total_params,
        "model_trainable_parameters": trainable_params,
        "model_size_mb": model_size_mb,
        "model_layer_count": layer_count,
        "model_conv_layers": conv_layers,
        "model_linear_layers": linear_layers,
        "model_complexity_score": total_params / 1e6  # Millions of parameters
    })

def create_learning_curve_plot(history_df):
    """Create and return learning curve plots for WandB"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history_df['epoch']
    
    # Loss curves
    ax1.plot(epochs, history_df['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history_df['val_loss'], label='Val Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, history_df['val_acc'], label='Val Accuracy', marker='o', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 Score curve
    ax3.plot(epochs, history_df['val_f1'], label='Val F1', marker='o', color='orange')
    ax3.set_title('Validation F1 Score')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Training time per epoch
    if 'epoch_time' in history_df.columns:
        ax4.plot(epochs, history_df['epoch_time'], label='Epoch Time', marker='o', color='red')
        ax4.set_title('Training Time per Epoch')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def log_confusion_matrix_analysis(y_true, y_pred, class_names, epoch):
    """Create detailed confusion matrix analysis"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'Confusion Matrix (Raw Counts) - Epoch {epoch}')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title(f'Confusion Matrix (Normalized) - Epoch {epoch}')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    # Log to WandB
    wandb.log({
        f"confusion_matrix_detailed_epoch_{epoch}": wandb.Image(fig),
        "confusion_matrix_wandb": wandb.plot.confusion_matrix(
            probs=None, y_true=y_true, preds=y_pred, class_names=class_names
        )
    }, step=epoch)
    
    plt.close(fig)
    
    # Calculate and log per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        wandb.log({f"class_accuracy_{class_name}": class_accuracies[i]}, step=epoch)

def log_training_efficiency_metrics(epoch_times, gpu_memory_usage=None):
    """Log training efficiency and resource utilization"""
    
    if len(epoch_times) > 0:
        wandb.log({
            "avg_epoch_time": np.mean(epoch_times),
            "total_training_time": np.sum(epoch_times),
            "epoch_time_std": np.std(epoch_times),
            "fastest_epoch_time": np.min(epoch_times),
            "slowest_epoch_time": np.max(epoch_times)
        })
    
    if gpu_memory_usage:
        wandb.log({
            "avg_gpu_memory_usage": np.mean(gpu_memory_usage),
            "max_gpu_memory_usage": np.max(gpu_memory_usage),
            "gpu_memory_efficiency": np.mean(gpu_memory_usage) / np.max(gpu_memory_usage) if np.max(gpu_memory_usage) > 0 else 0
        })