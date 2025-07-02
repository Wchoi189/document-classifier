import torch
import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
import wandb
import numpy as np

from pathlib import Path
from models.model import create_model
from data.augmentation import get_valid_transforms

def predict_from_checkpoint(checkpoint_path, input_path, config, device, wandb_project=None):
    """
    Predicts classes for images using a trained model checkpoint.
    Enhanced with optional WandB logging.
    """
    
    # Initialize WandB if project specified
    if wandb_project:
        # --- 1. Create a dynamic name for the prediction run ---
        model_name = config.get('model', {}).get('name', 'unknown-model')
        
        # Extract a unique identifier from the checkpoint path
        checkpoint_id = Path(checkpoint_path).stem
        
        run_name = f"predict-{model_name}-{checkpoint_id}"

        # --- 2. Initialize WandB with the custom name ---
        wandb.init(
            project=wandb_project,
            name=run_name,  # Use the custom name here
            job_type="inference",
            config={
                "checkpoint_path": checkpoint_path,
                "input_path": input_path,
                "model_name": model_name
            }
        )


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
    wandb_images = []  # For logging sample predictions
    
    with torch.no_grad():
        for i, file_path in enumerate(tqdm(image_files, desc="Predicting")):
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
                
                result = {
                    'filename': os.path.basename(file_path),
                    'predicted_class': class_names[int(predicted_idx.item())],
                    'predicted_target': int(predicted_idx.item()),  # Add numeric target
                    'confidence': confidence.item()
                }
                results.append(result)
                
                # Log sample images to WandB (first 10 images)
                if wandb_project and i < 10:
                    # Denormalize image for display
                    img_display = transformed_image[0].cpu().numpy().transpose(1, 2, 0)
                    mean = np.array(config['data']['mean'])
                    std = np.array(config['data']['std'])
                    img_display = img_display * std + mean
                    img_display = np.clip(img_display, 0, 1)
                    
                    caption = f"Pred: {result['predicted_class']}\nConf: {result['confidence']:.3f}"
                    wandb_images.append(wandb.Image(img_display, caption=caption))
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                continue
    
    # Log results to WandB
    if wandb_project and results:
        df_results = pd.DataFrame(results)
        
        # Log prediction statistics
        class_distribution = df_results['predicted_class'].value_counts()
        confidence_stats = df_results['confidence'].describe()
        
        wandb.log({
            "prediction_count": len(results),
            "mean_confidence": df_results['confidence'].mean(),
            "min_confidence": df_results['confidence'].min(),
            "max_confidence": df_results['confidence'].max(),
            "std_confidence": df_results['confidence'].std(),
            "low_confidence_count": len(df_results[df_results['confidence'] < 0.5])
        })
        
        # Log class distribution
        for class_name, count in class_distribution.items():
            wandb.log({f"predicted_{class_name}": count})
        
        # Log sample predictions
        if wandb_images:
            wandb.log({"sample_predictions": wandb_images})
        
        # Create and log prediction table
        wandb.log({"predictions_table": wandb.Table(dataframe=df_results)})
        
        # Log confidence distribution histogram
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(df_results['confidence'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        wandb.log({"confidence_distribution": wandb.Image(plt)})
        plt.close()
        
        wandb.finish()
    
    return results

# ADD NEW FUNCTION for WandB-enabled prediction
def predict_with_wandb(checkpoint, input_path, config='config/config.yaml', 
                      output='outputs/predictions.csv', wandb_project=None, debug=False):
    """
    🔮 Predict with WandB logging enabled.
    
    Args:
        checkpoint (str): Path to the model checkpoint (.pth file)
        input_path (str): Path to an image file or directory of images
        config (str): Path to the config YAML file
        output (str): Path to save the prediction results CSV file
        wandb_project (str): WandB project name for logging (optional)
        debug (bool): Enable debug mode with detailed logging
    
    Example:
        python predict.py predict_with_wandb checkpoints/best_model.pth data/test/ --wandb-project document-classifier
    """
    from utils.utils import load_config
    
    # Load config
    config_data = load_config(config)
    device = torch.device(config_data['device'] if torch.cuda.is_available() else 'cpu')
    
    # Run prediction with WandB logging
    results = predict_from_checkpoint(
        checkpoint_path=checkpoint,
        input_path=input_path,
        config=config_data,
        device=device,
        wandb_project=wandb_project
    )
    
    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output, index=False)
        print(f"✅ Prediction complete! Results saved to: {output}")
        
        # Create submission format
        output_df = pd.DataFrame({
            'ID': df_results['filename'],
            'target': df_results['predicted_target']
        }).sort_values('ID').reset_index(drop=True)
        
        submission_file = output.replace('.csv', '_submission.csv')
        output_df.to_csv(submission_file, index=False)
        print(f"📄 Submission file saved to: {submission_file}")
        
        return results
    else:
        print("⚠️  No results to save!")
        return []