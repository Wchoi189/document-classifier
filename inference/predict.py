import torch
import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

from models.model import create_model
from data.augmentation import get_valid_transforms

def predict_from_checkpoint(checkpoint_path, input_path, config, device):
    """
    Predicts classes for images using a trained model checkpoint.
    """
    
    # FIX: Get class information from meta.csv instead of dataset
    meta_file = config['data']['meta_file']
    if os.path.exists(meta_file):
        meta_df = pd.read_csv(meta_file)
        num_classes = len(meta_df)
        class_names = [meta_df[meta_df['target'] == i]['class_name'].iloc[0] for i in range(num_classes)]
        print(f"✅ Loaded {num_classes} classes from meta.csv")
    else:
        # Fallback: try to get from config or use the EDA findings
        print(f"⚠️  Meta file not found: {meta_file}")
        print(f"Using EDA findings: 17 classes")
        num_classes = 17  # From your EDA findings
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    print(f"Creating model with {num_classes} classes")
    
    # Create model with correct number of classes
    model = create_model(
        model_name=config['model']['name'],
        num_classes=num_classes,  # This should be 17, not 2
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_data)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise e
    
    model.eval()
    
    # Setup transforms
    transforms = get_valid_transforms(
        height=config['data']['image_size'], 
        width=config['data']['image_size'],
        mean=config['data']['mean'], 
        std=config['data']['std']
    )
    
    # Get image files
    image_files = []
    if os.path.isfile(input_path):
        image_files.append(input_path)
    elif os.path.isdir(input_path):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [
            os.path.join(input_path, f) 
            for f in os.listdir(input_path) 
            if f.lower().endswith(valid_extensions)
        ]
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    with torch.no_grad():
        for file_path in tqdm(image_files, desc="Predicting"):
            try:
                # Load and preprocess image
                image = cv2.imread(file_path)
                if image is None:
                    print(f"⚠️  Could not load image: {file_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed_image = transforms(image=image)['image'].unsqueeze(0).to(device)
                
                # Predict
                output = model(transformed_image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                results.append({
                    'filename': os.path.basename(file_path),
                    'predicted_class': class_names[int(predicted_idx.item())],
                    'predicted_target': int(predicted_idx.item()),  # Add numeric target
                    'confidence': confidence.item()
                })
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                continue
    
    return results