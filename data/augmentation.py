import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(height, width, mean, std):
    """학습용 기본 이미지 변환 함수"""
    return A.Compose([
        A.Resize(height=1024, width=1024),  # First step
        A.Resize(height=512, width=512),    # Second step  
        A.Resize(height=height, width=width),  # Final step
        A.OneOf([
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.MotionBlur(p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.ISONoise(p=1.0),
            A.GaussNoise(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ], p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

def get_valid_transforms(height, width, mean, std):
    """검증/테스트용 이미지 변환 함수"""
    return A.Compose([
        A.Resize(height=1024, width=1024),  # First step
        A.Resize(height=512, width=512),    # Second step  
        A.Resize(height=height, width=width),  # Final step
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

def get_document_transforms(height, width, mean, std):
    """문서 특화 변환 함수 (Augraphy 라이브러리 활용)"""
    try:
        from augraphy import (
            BleedThrough, LowInkRandomLines, DirtyRollers, 
            BadPhotoCopy, Jpeg, ShadowCast
        )
        
        # Augraphy 변환은 numpy 배열을 입력으로 받으므로, ToTensorV2 이전에 위치해야 합니다.
        augraphy_pipeline = A.Compose([
            A.OneOf([
                A.Lambda(image=BleedThrough(p=1), p=1.0),
                A.Lambda(image=LowInkRandomLines(p=1), p=1.0),
                A.Lambda(image=DirtyRollers(p=1), p=1.0),
            ], p=0.5),
            A.OneOf([
                A.Lambda(image=BadPhotoCopy(p=1), p=1.0),
                A.Lambda(image=Jpeg(quality_range=(70, 95), p=1), p=1.0),
                A.Lambda(image=ShadowCast(p=1), p=1.0),
            ], p=0.5),
        ], p=0.7)

        return A.Compose([
            A.Resize(height=1024, width=1024),  # First step
            A.Resize(height=512, width=512),    # Second step  
            A.Resize(height=height, width=width),  # Final step
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            augraphy_pipeline,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    except ImportError:
        print("경고: Augraphy 라이브러리를 찾을 수 없습니다. 기본 변환을 사용합니다.")
        return get_train_transforms(height, width, mean, std)
