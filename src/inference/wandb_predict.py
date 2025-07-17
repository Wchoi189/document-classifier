import wandb
import torch
import os
import pandas as pd
from pathlib import Path

from src.inference.predictor import predict_from_checkpoint
from src.utils.utils import load_config

def predict_with_wandb_logging(checkpoint, input_path, config='configs/config.yaml', 
                              output='outputs/predictions.csv', wandb_project="document-classifier"):
    """
    Enhanced prediction with comprehensive WandB logging
    """
    
    # Initialize WandB
    wandb.init(project=wandb_project, job_type="inference")
    
    # Load config
    config_data = load_config(config)
    device = torch.device(config_data['device'] if torch.cuda.is_available() else 'cpu')
    
    # Log inference configuration
    wandb.log({
        "checkpoint_path": checkpoint,
        "input_path": input_path,
        "device": str(device),
        "model_name": config_data['model']['name'],
        "image_size": config_data['data']['image_size']
    })
    
    # Run prediction
    results = predict_from_checkpoint(
        checkpoint_path=checkpoint,
        input_path=input_path,
        config=config_data,
        device=device
    )
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Comprehensive WandB logging
        _log_prediction_analysis(df_results, wandb)
        
        # Save and log artifacts
        df_results.to_csv(output, index=False)
        
        # Log prediction file as artifact
        artifact = wandb.Artifact("predictions", type="predictions")
        artifact.add_file(output)
        wandb.log_artifact(artifact)
        
        print(f"✅ Prediction complete with WandB logging!")
        print(f"📊 Results saved to: {output}")
        
    wandb.finish()
    return results

def _log_prediction_analysis(df_results, wandb_instance):
    """Log comprehensive prediction analysis to WandB"""
    
    # Basic statistics
    wandb_instance.log({
        "total_predictions": len(df_results),
        "unique_classes_predicted": df_results['predicted_class'].nunique(),
        "mean_confidence": df_results['confidence'].mean(),
        "median_confidence": df_results['confidence'].median(),
        "min_confidence": df_results['confidence'].min(),
        "max_confidence": df_results['confidence'].max(),
        "std_confidence": df_results['confidence'].std()
    })
    
    # Class distribution analysis
    class_distribution = df_results['predicted_class'].value_counts()
    for class_name, count in class_distribution.items():
        wandb_instance.log({f"count_{class_name}": count})
    
    # Confidence analysis by ranges
    confidence_ranges = [
        (0.9, 1.0, "very_high"),
        (0.8, 0.9, "high"),
        (0.7, 0.8, "medium"),
        (0.6, 0.7, "low"),
        (0.0, 0.6, "very_low")
    ]
    
    for min_conf, max_conf, label in confidence_ranges:
        count = len(df_results[(df_results['confidence'] >= min_conf) & 
                              (df_results['confidence'] < max_conf)])
        wandb_instance.log({f"confidence_{label}": count})
    
    # Log tables
    wandb_instance.log({
        "predictions_table": wandb.Table(dataframe=df_results),
        "class_distribution_table": wandb.Table(
            data=[[k, v] for k, v in class_distribution.items()],
            columns=["Class", "Count"]
        )
    })