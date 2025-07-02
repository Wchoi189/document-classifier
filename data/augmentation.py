import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(height, width, mean, std):
    """학습용 기본 이미지 변환 함수"""
    return A.Compose([
        A.Resize(height=height, width=width),
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
        A.Resize(height=height, width=width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

# Create picklable wrapper classes for Augraphy transforms
class AugraphyTransform:
    """Base class for Augraphy transforms that can be pickled"""
    def __init__(self, transform_class, **kwargs):
        self.transform_class = transform_class
        self.kwargs = kwargs
    
    def __call__(self, image, **albumentations_kwargs):
        # Only pass the image to Augraphy transforms, ignore albumentations kwargs
        transform = self.transform_class(**self.kwargs)
        return transform(image)

class BleedThroughTransform(AugraphyTransform):
    def __init__(self):
        try:
            from augraphy import BleedThrough
            super().__init__(BleedThrough, p=1)
        except ImportError:
            raise ImportError("Augraphy not available")

class LowInkRandomLinesTransform(AugraphyTransform):
    def __init__(self):
        try:
            from augraphy import LowInkRandomLines
            super().__init__(LowInkRandomLines, p=1)
        except ImportError:
            raise ImportError("Augraphy not available")

class DirtyRollersTransform(AugraphyTransform):
    def __init__(self):
        try:
            from augraphy import DirtyRollers
            super().__init__(DirtyRollers, p=1)
        except ImportError:
            raise ImportError("Augraphy not available")

class BadPhotoCopyTransform(AugraphyTransform):
    def __init__(self):
        try:
            from augraphy import BadPhotoCopy
            super().__init__(BadPhotoCopy, p=1)
        except ImportError:
            raise ImportError("Augraphy not available")

class JpegTransform(AugraphyTransform):
    def __init__(self):
        try:
            from augraphy import Jpeg
            super().__init__(Jpeg, quality_range=(70, 95), p=1)
        except ImportError:
            raise ImportError("Augraphy not available")

class ShadowCastTransform(AugraphyTransform):
    def __init__(self):
        try:
            from augraphy import ShadowCast
            super().__init__(ShadowCast, p=1)
        except ImportError:
            raise ImportError("Augraphy not available")

def get_document_transforms(height, width, mean, std):
    """문서 특화 변환 함수 (Augraphy 라이브러리 활용) - 수정된 버전"""
    try:
        # Test if Augraphy is available
        import augraphy
        
        return A.Compose([
            A.Resize(height=height, width=width),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            
            # Paper quality effects
            A.OneOf([
                A.Lambda(image=BleedThroughTransform(), p=1.0),
                A.Lambda(image=LowInkRandomLinesTransform(), p=1.0),
                A.Lambda(image=DirtyRollersTransform(), p=1.0),
            ], p=0.4),  # Reduced probability for stability
            
            # Scanning/photo effects
            A.OneOf([
                A.Lambda(image=BadPhotoCopyTransform(), p=1.0),
                A.Lambda(image=JpegTransform(), p=1.0),
                A.Lambda(image=ShadowCastTransform(), p=1.0),
            ], p=0.4),  # Reduced probability for stability
            
            # Standard augmentations
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    except ImportError:
        print("⚠️  Augraphy 라이브러리를 찾을 수 없습니다. 문서 특화 기본 변환을 사용합니다.")
        return get_enhanced_document_transforms(height, width, mean, std)
    except Exception as e:
        print(f"⚠️  Augraphy 초기화 오류: {e}. 문서 특화 기본 변환을 사용합니다.")
        return get_enhanced_document_transforms(height, width, mean, std)

def get_enhanced_document_transforms(height, width, mean, std):
    """Augraphy 없이 문서 특화 변환 (fallback)"""
    return A.Compose([
        A.Resize(height=height, width=width),
        
        # Geometric transforms (document scanning effects)
        A.OneOf([
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, 
                             border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Perspective(scale=(0.05, 0.1), p=1.0),  # Document perspective distortion
        ], p=0.7),
        
        # Quality degradation (scanning/photo artifacts)
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.4),
        
        # Lighting effects (document scanning conditions)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, 
                          num_shadows_upper=2, shadow_dimension=5, p=1.0),
        ], p=0.5),
        
        # Noise (scanning artifacts, compression)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),
        
        # Compression and quality loss
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
        
        # Color shifts (scanner color calibration issues)
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.2),
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])