import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Literal
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def calculate_metrics(y_true, y_pred, average: Literal['binary', 'micro', 'macro', 'samples', 'weighted'] = 'weighted'):
    """
    분류 성능 지표(정확도, 정밀도, 재현율, F1 점수)를 계산합니다.
    
    Args:
        y_true (list or np.array): 실제 라벨
        y_pred (list or np.array): 예측 라벨
        average (str): 'micro', 'macro', 'weighted' 등 F1 점수 계산 방식
        
    Returns:
        dict: 성능 지표 딕셔너리
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    혼동 행렬(Confusion Matrix)을 시각화합니다.
    
    Args:
        cm (np.array): sklearn.metrics.confusion_matrix에서 계산된 혼동 행렬
        class_names (list): 클래스 이름 목록
        normalize (bool): True일 경우, 행렬을 정규화하여 백분율로 표시
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.show()
