import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# # --- Augraphy wrapper functions for pickling ---
# def apply_paper_factory(img):
#     from augraphy import PaperFactory
#     return PaperFactory()(img)

# def apply_color_paper(img):
#     from augraphy import ColorPaper
#     return ColorPaper()(img)

# def apply_bleed_through(img):
#     from augraphy import BleedThrough
#     return BleedThrough()(img)

# def apply_low_ink_random_lines(img):
#     from augraphy import LowInkRandomLines
#     return LowInkRandomLines()(img)

# def apply_bad_photo_copy(img):
#     from augraphy import BadPhotoCopy
#     return BadPhotoCopy()(img)

# def apply_dirty_rollers(img):
#     from augraphy import DirtyRollers
#     return DirtyRollers()(img)

# def apply_shadow_cast(img):
#     from augraphy import ShadowCast
#     return ShadowCast()(img)



# Attempt to import Augraphy and set a flag
try:
    from augraphy import (
        BleedThrough, LowInkRandomLines, DirtyRollers, BadPhotoCopy,
        Jpeg, ShadowCast, PaperFactory, ColorPaper
    )
    AUGMENTATIONS_AVAILABLE = True
except ImportError:
    print("Warning: Augraphy library not found. Document-specific augmentations will be disabled.")
    AUGMENTATIONS_AVAILABLE = False


from typing import Sequence

# def create_document_transforms(config, height, width, mean, std):
#     """
#     Dynamically creates an augmentation pipeline from a configuration dictionary.
    
#     Args:
#         config (dict): A dictionary containing augmentation settings.
#         height (int): Target image height.
#         width (int): Target image width.
#         mean (list): Mean values for normalization.
#         std (list): Standard deviation values for normalization.
        
#     Returns:
#         A.Compose: The configured albumentations pipeline.
#     """
#     transforms: list = [A.Resize(height=height, width=width)]
    
#     # --- Geometric Transformations ---
#     geom_transforms: list[A.BasicTransform] = []
#     if config.get("rotation_limit", 0) > 0:
#         geom_transforms.append(A.Rotate(limit=config["rotation_limit"], border_mode=cv2.BORDER_CONSTANT, p=1.0))
        
#     if config.get("perspective_scale", 0) > 0:
#         geom_transforms.append(A.Perspective(scale=config["perspective_scale"], p=1.0))

#     if config.get("shift_scale_rotate", False):
#         # Using small, document-friendly values
#         geom_transforms.append(A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=5, p=1.0))
            
#     if geom_transforms:
#         transforms.append(A.OneOf(list(geom_transforms), p=config.get("geom_probability", 0.5)))

#     # --- Color & Lighting Transformations ---
#     color_transforms: list[A.BasicTransform] = []
#     if config.get("brightness_contrast", 0) > 0:
#         color_transforms.append(A.RandomBrightnessContrast(
#             brightness_limit=config.get("brightness_limit", 0.15),
#             contrast_limit=config.get("contrast_limit", 0.15),
#             p=1.0
#         ))

#     if config.get("gamma_correction", 0) > 0:
#         color_transforms.append(A.RandomGamma(p=1.0))
        
#     if color_transforms:
#         transforms.append(A.OneOf(list(color_transforms), p=config.get("color_probability", 0.4)))
        
#     # --- Noise & Quality Transformations ---
#     noise_transforms: list[A.BasicTransform] = []
#     if config.get("gaussian_noise", 0) > 0:
#         noise_variance = config.get("noise_variance", [5.0, 15.0])
#         noise_transforms.append(A.GaussNoise(
#             var_limit=(noise_variance[0], noise_variance[1]),
#             p=1.0
#         ))

#     if config.get("jpeg_compression", 0) > 0:
#         compression_quality = config.get("compression_quality", [75, 95])
#         if isinstance(compression_quality, list) or isinstance(compression_quality, tuple):
#             quality_lower = compression_quality[0]
#             quality_upper = compression_quality[1]
#         else:
#             quality_lower = quality_upper = compression_quality
#         noise_transforms.append(A.ImageCompression(
#             quality=(quality_lower, quality_upper),
#             p=1.0
#         ))
        
#     if noise_transforms:
#         transforms.append(A.OneOf(list(noise_transforms), p=config.get("noise_probability", 0.3)))

#     # --- Augraphy Transformations ---
#     if config.get("use_augraphy", False) and AUGMENTATIONS_AVAILABLE:
#         # Ensure all Augraphy classes are imported
#         from augraphy import (
#             BleedThrough, LowInkRandomLines, DirtyRollers, BadPhotoCopy,
#             ShadowCast, PaperFactory, ColorPaper
#         )
#         augraphy_transforms: list = []
        
#         # Paper Effects
#         if config.get("paper_effects", 0) > 0:
#             augraphy_transforms.append(
#                 A.OneOf([
#                     A.Lambda(image=lambda img: PaperFactory()(img), p=1.0),
#                     A.Lambda(image=lambda img: ColorPaper()(img), p=1.0)
#                 ], p=config["paper_effects"])
#             )

#         # Ink Effects
#         if config.get("ink_effects", 0) > 0:
#             augraphy_transforms.append(
#                 A.OneOf([
#                     A.Lambda(image=lambda img: BleedThrough()(img), p=1.0),
#                     A.Lambda(image=lambda img: LowInkRandomLines()(img), p=1.0)
#                 ], p=config["ink_effects"])
#             )
            
#         # Scan/Copy Effects
#         if config.get("scan_effects", 0) > 0:
#             augraphy_transforms.append(
#                 A.OneOf([
#                     A.Lambda(image=lambda img: BadPhotoCopy()(img), p=1.0),
#                     A.Lambda(image=lambda img: DirtyRollers()(img), p=1.0),
#                     A.Lambda(image=lambda img: ShadowCast()(img), p=1.0)
#                 ], p=config["scan_effects"])
#             )
        
#         if augraphy_transforms:
#             # The main probability controls whether any Augraphy effect is applied
#             transforms.append(A.OneOf(augraphy_transforms, p=config.get("augraphy_probability", 0.6)))

#     # --- Final Steps ---
#     transforms.append(A.Normalize(mean=mean, std=std))
#     transforms.append(ToTensorV2())

#     return A.Compose(transforms)


# def create_document_transforms(config, height, width, mean, std):
#     """
#     Dynamically creates a pickle-safe augmentation pipeline from a configuration.
#     """
#     transforms: list = [A.Resize(height=height, width=width)]
    
#     # --- Geometric Transformations ---
#     geom_transforms = []
#     if config.get("rotation_limit", 0) > 0:
#         geom_transforms.append(A.Rotate(limit=config["rotation_limit"], border_mode=cv2.BORDER_CONSTANT, p=1.0))
#     if config.get("perspective_scale", 0) > 0:
#         geom_transforms.append(A.Perspective(scale=config["perspective_scale"], p=1.0))
#     if config.get("shift_scale_rotate", False):
#         geom_transforms.append(A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=5, p=1.0))
#     if geom_transforms:
#         transforms.append(A.OneOf(geom_transforms, p=config.get("geom_probability", 0.5)))

#     # --- Color & Lighting Transformations ---
#     color_transforms = []
#     if config.get("brightness_contrast", False):
#         color_transforms.append(A.RandomBrightnessContrast(
#             brightness_limit=config.get("brightness_limit", 0.15),
#             contrast_limit=config.get("contrast_limit", 0.15),
#         ))
    
#     # --- Final Steps ---
#     transforms.extend([
#         A.Normalize(mean=mean, std=std),
#         ToTensorV2(),
#     ])
    
#     return A.Compose(transforms)

def get_train_transforms(height, width, mean, std):
    """학습용 기본 이미지 변환 함수"""
    return A.Compose([
        # A.Resize(height=1024, width=1024),  # First step
        # A.Resize(height=512, width=512),    # Second step  
        A.Resize(height=height, width=width),  # Final step
        A.OneOf([
            # A.RandomRotate90(p=1.0), # This line causes 90-degree rotations
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
            # This corrected code removes the 90-degree option and applies a rotation of up to 15 degrees with a 70% probability.
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
        # A.Resize(height=1024, width=1024),  # First step
        # A.Resize(height=512, width=512),    # Second step  
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
            # A.Resize(height=1024, width=1024),  # First step
            # A.Resize(height=512, width=512),    # Second step  
            A.Resize(height=height, width=width),  # Final step
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            augraphy_pipeline,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    except ImportError:
        print("경고: Augraphy 라이브러리를 찾을 수 없습니다. 기본 변환을 사용합니다.")
        return get_train_transforms(height, width, mean, std)
