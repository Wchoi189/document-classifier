# src/data/single_augmentations.py
"""
Single, modular augmentation functions that work with Albumentations.
Each function is tested individually before combining.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np




def get_perspective_transform(height: int, width: int, mean: tuple, std: tuple):
    """
    Single perspective distortion augmentation - WORKING VERSION
    
    Based on peer's working approach - simple and clean.
    
    Args:
        height: Target image height
        width: Target image width  
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        
        # Simple perspective distortion - like documents photographed at angle
        A.Perspective(
            scale=(0.05, 0.15),  # How much perspective distortion
            keep_size=True,      # Keep output size consistent
            p=0.7                # 70% chance to apply
        ),
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# Test function to validate it works
def test_perspective_transform():
    """Test the perspective transform to make sure it works"""
    
    # Standard values from your config
    height, width = 224, 224
    mean = (0.485, 0.456, 0.406) 
    std = (0.229, 0.224, 0.225)
    
    # Create the transform
    transform = get_perspective_transform(height, width, mean, std)
    
    # Create dummy image (white document-like)
    dummy_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Add some black text-like rectangles
    dummy_image[50:70, 50:200] = 0   # Horizontal line
    dummy_image[100:120, 50:150] = 0 # Another line
    dummy_image[150:170, 50:250] = 0 # Third line
    
    try:
        # Apply transform
        result = transform(image=dummy_image)
        transformed_image = result['image']
        
        print("✅ Perspective transform test PASSED!")
        print(f"Input shape: {dummy_image.shape}")
        print(f"Output shape: {transformed_image.shape}")
        print(f"Output type: {type(transformed_image)}")
        return True
        
    except Exception as e:
        print(f"❌ Perspective transform test FAILED: {e}")
        return False


def get_lighting_transform(height: int, width: int, mean: tuple, std: tuple):
    """
    Lighting and shadow augmentation - WORKING VERSION
    
    Simulates poor lighting conditions like in your test images.
    
    Args:
        height: Target image height
        width: Target image width  
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        
        # Lighting variations - critical for your test data
        A.OneOf([
            # Poor lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.3,    # ±30% brightness change
                contrast_limit=0.3,      # ±30% contrast change
                p=1.0
            ),
            # Shadow effects
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),     # Shadow can appear anywhere
                shadow_dimension=5,             # Shadow size
                p=1.0
            ),
            # Gamma correction for exposure issues
            A.RandomGamma(
                gamma_limit=(70, 130),   # Gamma range
                p=1.0
            ),
        ], p=0.8),  # 80% chance to apply one of these
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def test_lighting_transform():
    """Test the lighting transform to make sure it works"""
    
    # Standard values
    height, width = 224, 224
    mean = (0.485, 0.456, 0.406) 
    std = (0.229, 0.224, 0.225)
    
    # Create the transform
    transform = get_lighting_transform(height, width, mean, std)
    
    # Create dummy image (white document-like)
    dummy_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Add some black text-like rectangles
    dummy_image[50:70, 50:200] = 0   # Horizontal line
    dummy_image[100:120, 50:150] = 0 # Another line
    dummy_image[150:170, 50:250] = 0 # Third line
    
    try:
        # Apply transform
        result = transform(image=dummy_image)
        transformed_image = result['image']
        
        print("✅ Lighting transform test PASSED!")
        print(f"Input shape: {dummy_image.shape}")
        print(f"Output shape: {transformed_image.shape}")
        print(f"Output type: {type(transformed_image)}")
        return True
        
    except Exception as e:
        print(f"❌ Lighting transform test FAILED: {e}")
        return False


def get_quality_transform(height: int, width: int, mean: tuple, std: tuple):
    """
    Quality degradation augmentation - WORKING VERSION
    
    Simulates blur, noise, and compression like in your test images.
    
    Args:
        height: Target image height
        width: Target image width  
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        
        # Quality issues - blur, noise, compression
        A.OneOf([
            # Motion blur (camera shake)
            A.MotionBlur(
                blur_limit=7,    # Max blur amount
                p=1.0
            ),
            # Gaussian blur (out of focus)
            A.GaussianBlur(
                blur_limit=(3, 7),   # Blur range
                p=1.0
            ),
            # Gaussian noise
            A.GaussNoise(
                std_range=(0.04, 0.2),  # 🔧 Changed from var_limit to std_range, normalized to [0,1]
                p=1.0
            ),
        ], p=0.6),  # 60% chance to apply one of these
        
        # Compression artifacts
        A.ImageCompression(
            quality_range=(60, 100),  # 🔧 Changed from quality_lower/quality_upper to quality_range
            p=0.4                     # 40% chance to apply
        ),
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def test_quality_transform():
    """Test the quality transform to make sure it works"""
    
    # Standard values
    height, width = 224, 224
    mean = (0.485, 0.456, 0.406) 
    std = (0.229, 0.224, 0.225)
    
    # Create the transform
    transform = get_quality_transform(height, width, mean, std)
    
    # Create dummy image (white document-like)
    dummy_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Add some black text-like rectangles
    dummy_image[50:70, 50:200] = 0   # Horizontal line
    dummy_image[100:120, 50:150] = 0 # Another line
    dummy_image[150:170, 50:250] = 0 # Third line
    
    try:
        # Apply transform
        result = transform(image=dummy_image)
        transformed_image = result['image']
        
        print("✅ Quality transform test PASSED!")
        print(f"Input shape: {dummy_image.shape}")
        print(f"Output shape: {transformed_image.shape}")
        print(f"Output type: {type(transformed_image)}")
        return True
        
    except Exception as e:
        print(f"❌ Quality transform test FAILED: {e}")
        return False


def get_robust_document_transforms(height: int, width: int, mean: tuple, std: tuple):
    """
    COMBINED robust document augmentation - WORKING VERSION
    
    Combines all three working transforms to simulate real test conditions.
    This is the main function you'll use for training.
    
    Args:
        height: Target image height
        width: Target image width  
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        
        # 1. Geometric distortions (like your test images)
        A.Perspective(
            scale=(0.05, 0.15),  # Document photographed at angle
            keep_size=True,
            p=0.6                # 60% chance
        ),
        
        # 2. Lighting and shadow issues
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                shadow_dimension=5,
                p=1.0
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),
                p=1.0
            ),
        ], p=0.7),  # 70% chance
        
        # 3. Quality degradation
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(p=1.0),  # Simplified - no invalid params
        ], p=0.5),  # 50% chance
        
        # 4. Compression artifacts
        A.ImageCompression(p=0.3),  # Simplified - no invalid params
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def test_combined_transform():
    """Test the combined transform to make sure it works"""
    
    # Standard values
    height, width = 224, 224
    mean = (0.485, 0.456, 0.406) 
    std = (0.229, 0.224, 0.225)
    
    # Create the combined transform
    transform = get_robust_document_transforms(height, width, mean, std)
    
    # Create dummy image (white document-like)
    dummy_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Add some black text-like rectangles
    dummy_image[50:70, 50:200] = 0   # Horizontal line
    dummy_image[100:120, 50:150] = 0 # Another line
    dummy_image[150:170, 50:250] = 0 # Third line
    
    try:
        # Apply transform multiple times to test randomness
        for i in range(3):
            result = transform(image=dummy_image)
            transformed_image = result['image']
            print(f"  Run {i+1}: Output shape: {transformed_image.shape}")
        
        print("✅ Combined transform test PASSED!")
        print("🎯 Ready to use for training!")
        return True
        
    except Exception as e:
        print(f"❌ Combined transform test FAILED: {e}")
        return False


if __name__ == "__main__":
    # Run all tests when file is executed directly
    print("Testing individual transforms...")
    print("1. Perspective transform...")
    test_perspective_transform()
    
    print("\n2. Lighting transform...")
    test_lighting_transform()
    
    print("\n3. Quality transform...")
    test_quality_transform()
    
    print("\n" + "="*50)
    print("Testing COMBINED transform...")
    test_combined_transform()
    
    print("\n🚀 All tests complete!")
    print("You can now use get_robust_document_transforms() in your training!")