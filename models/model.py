import timm
import torch.nn as nn

def create_model(model_name, num_classes, pretrained=True):
    """
    timm 라이브러리를 사용하여 모델을 생성합니다.
    
    Args:
        model_name (str): timm에서 지원하는 모델 이름
        num_classes (int): 분류할 클래스의 수
        pretrained (bool): 사전 학습된 가중치를 사용할지 여부
        
    Returns:
        torch.nn.Module: 생성된 모델
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model
