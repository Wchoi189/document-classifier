import torch
import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image

from models.model import create_model
from data.augmentation import get_valid_transforms
from data.dataset import DocumentDataset # To get class names

def predict_from_checkpoint(checkpoint_path, input_path, config, device):
    """
    Predicts classes for images using a trained model checkpoint.
    """
    # This is a bit of a hack; ideally, class names are saved with the checkpoint.
    temp_dataset = DocumentDataset(root_dir=config['data']['root_dir'], split='test', val_size=0)
    class_names = temp_dataset.classes
    num_classes = len(class_names)
    
    model = create_model(
        model_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=False
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    transforms = get_valid_transforms(
        height=config['data']['image_size'], width=config['data']['image_size'],
        mean=config['data']['mean'], std=config['data']['std']
    )
    
    image_files = []
    if os.path.isfile(input_path):
        image_files.append(input_path)
    elif os.path.isdir(input_path):
        image_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    results = []
    with torch.no_grad():
        for file_path in tqdm(image_files, desc="Predicting"):
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            transformed_image = transforms(image=image)['image'].unsqueeze(0).to(device)
            
            output = model(transformed_image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            results.append({
                'filename': os.path.basename(file_path),
                'predicted_class': class_names[int(predicted_idx.item())],
                'confidence': confidence.item()
            })
            
    return results
