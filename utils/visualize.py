import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_history(history):
    """
    학습 과정의 손실 및 성능 지표를 시각화합니다.
    
    Args:
        history (pd.DataFrame): 에포크별 'train_loss', 'val_loss', 'val_acc', 'val_f1' 등을 포함하는 데이터프레임
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Loss plot
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='o')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True)
    
    # Metrics plot (F1 score 우선, 없으면 Accuracy)
    if 'val_f1' in history.columns:
        metric_name = 'F1 Score'
        metric_key = 'val_f1'
    else:
        metric_name = 'Accuracy'
        metric_key = 'val_acc'

    ax2.plot(history['epoch'], history[metric_key], label=f'Validation {metric_name}', marker='o', color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(metric_name, fontsize=12)
    ax2.set_title(f'Validation {metric_name} per Epoch', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
