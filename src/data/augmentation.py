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
            # A.MotionBlur(p=1.0),
            # A.MedianBlur(blur_limit=3, p=1.0),
            # A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.5),
        A.OneOf([
            # A.ISONoise(p=1.0),
            A.GaussNoise(p=1.0),
            # A.RandomBrightnessContrast(p=1.0),
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
# ADD THIS NEW FUNCTION
def get_configurable_transforms(height, width, mean, std, config):
    """Configurable augmentation based on config parameters"""
    transforms_list = [A.Resize(height=height, width=width)]
    
    strategy = config.get('strategy', 'basic')
    intensity = config.get('intensity', 0.5)
    
    if strategy == 'robust':
        # Perspective distortion
        transforms_list.append(
            A.Perspective(scale=(0.01, 0.05), keep_size=True, p=0.3 * intensity)
        )
        
        # Lighting variations
        # transforms_list.append(
        #     A.OneOf([
        #         A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.4) * intensity, contrast_limit=0.3 * intensity, p=1.0),
        #         # A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=1.0),
        #         # A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        #     ], p=0.7 * intensity)
        # )
        
        # Quality degradation
        transforms_list.append(
            A.OneOf([
                # A.MotionBlur(blur_limit=7, p=1.0),
                # A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.5 * intensity)
        )
        
        transforms_list.append(A.ImageCompression(p=0.3 * intensity))
    
    transforms_list.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(transforms_list)

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


# def get_document_transforms(height, width, mean, std):
#     """
#     ROBUST document transforms - WORKING VERSION
    
#     Replaces the problematic Augraphy-based function with tested,
#     working transforms that handle real-world test conditions.
    
#     Based on successful tests from single_augmentations.py
#     """
#     return A.Compose([
#         A.Resize(height=height, width=width),
        
#         # 1. Geometric distortions (like your test images)
#         A.Perspective(
#             scale=(0.05, 0.15),  # Document photographed at angle
#             keep_size=True,
#             p=0.6                # 60% chance
#         ),
        
#         # 2. Lighting and shadow issues
#         A.OneOf([
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.3,
#                 contrast_limit=0.3,
#                 p=1.0
#             ),
#             A.RandomShadow(
#                 shadow_roi=(0, 0.5, 1, 1),
#                 shadow_dimension=5,
#                 p=1.0
#             ),
#             A.RandomGamma(
#                 gamma_limit=(70, 130),
#                 p=1.0
#             ),
#         ], p=0.7),  # 70% chance
        
#         # 3. Quality degradation
#         A.OneOf([
#             A.MotionBlur(blur_limit=7, p=1.0),
#             A.GaussianBlur(blur_limit=(3, 7), p=1.0),
#             A.GaussNoise(p=1.0),  # Simplified - no problematic params
#         ], p=0.5),  # 50% chance
        
#         # 4. Compression artifacts
#         A.ImageCompression(p=0.3),  # Simplified - no problematic params
        
#         A.Normalize(mean=mean, std=std),
#         ToTensorV2(),
#     ])

# REPLACE THIS FUNCTION - SAME NAME, NEW CONTENT
def get_document_transforms(height, width, mean, std):
    """Document transforms - now uses robust strategy by default"""
    default_config = {'strategy': 'robust', 'intensity': 0.7}
    return get_configurable_transforms(height, width, mean, std, default_config)
# Remove all the problematic Augraphy wrapper classes and fallback function
# Keep only the three main functions above