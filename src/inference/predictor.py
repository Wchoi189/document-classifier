import torch
import os
import cv2
import pandas as pd
from tqdm import tqdm
import wandb
import numpy as np
from datetime import datetime
from pathlib import Path
from src.models.model import create_model
from src.data.augmentation import get_valid_transforms

def save_predictions(results: list, config: dict):
    """
    Saves prediction results to timestamped files.

    Generates two files:
    1. predictions_{timestamp}.csv: Detailed results with confidence scores.
    2. submission_{timestamp}.csv: Two-column file for submission.

    Args:
        results (list): A list of prediction result dictionaries.
        config (dict): Configuration dictionary with paths section
    """
    if not results:
        print("⚠️ No results to save.")
        return

    # Ensure the output directory exists
    paths_config = config.get('paths', {})
    output_dir = Path(paths_config.get('output_dir', 'outputs'))
    prediction_dir = Path(paths_config.get('prediction_dir', 'predictions'))
    output_path = output_dir / prediction_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate a 4-digit timestamp (HHMM)
    timestamp = datetime.now().strftime("%H%M")

    # --- Save detailed and submission files ---
    df_results = pd.DataFrame(results)
    detailed_filename = output_path / f"predictions_{timestamp}.csv"
    df_results.to_csv(detailed_filename, index=False)
    print(f"💾 Detailed predictions saved to: {detailed_filename}")

    # --- 2. Save Formatted Submission File ---
    submission_df = pd.DataFrame({
        'ID': df_results['filename'],
        'target': df_results['predicted_target']
    }).sort_values('ID').reset_index(drop=True)
    submission_filename = output_path / f"submission_{timestamp}.csv"
    submission_df.to_csv(submission_filename, index=False)
    print(f"📄 Submission file ready: {submission_filename}")

def predict_from_checkpoint(checkpoint_path, input_path, config, device, wandb_project=None, model_name_override=None):
    """
    Predicts classes for images using a trained model checkpoint.
    Enhanced with optional WandB logging.
    
    FIXED: Now uses config utilities for proper structure handling
    """
    
    # 🔧 FIX: Import and apply config utilities
    from src.utils.config_utils import normalize_config_structure, convert_config_types
    
    # 🔧 FIX: Ensure config is properly normalized
    config = normalize_config_structure(config)
    config = convert_config_types(config)

     # Use the override if provided, otherwise use the config default
    if model_name_override:
        model_name = model_name_override
        print(f"🔧 Model name overridden from command line. Using: {model_name}")
    else:
        model_name = config.get('name', 'resnet50')  # Default fallback   

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

    # 🔧 FIX: Safe access to model configuration
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    # FIX: Get class information from meta.csv instead of dataset
    meta_file = data_config.get('meta_file', 'data/raw/metadata/meta.csv')
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
    
    # 🔧 FIX: Safe model creation with fallback defaults
    model_name = model_config.get('name', 'resnet50')  # Default fallback
    pretrained = model_config.get('pretrained', True)   # Default fallback
    
    # Create model with correct number of classes
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,  # This should be 17, not 2
        pretrained=False  # Don't load pretrained for inference
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint_data = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint_data)
        print(f"✅ Model loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise e
    
    model.eval()
    
    # 🔧 FIX: Safe access to data configuration with defaults
    image_size = data_config.get('image_size', 224)
    mean = data_config.get('mean', [0.485, 0.456, 0.406])
    std = data_config.get('std', [0.229, 0.224, 0.225])
    
    # Setup transforms
    transforms = get_valid_transforms(
        height=image_size, 
        width=image_size,
        mean=mean, 
        std=std
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
                    mean_np = np.array(mean)
                    std_np = np.array(std)
                    img_display = img_display * std_np + mean_np
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
def predict_with_wandb(checkpoint, input_path, config='configs/config.yaml', 
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
    from src.utils.utils import load_config
    
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