import fire
import torch
import os
import pandas as pd
from pathlib import Path
from icecream import ic

from src.utils.utils import load_config
from src.inference.predict import predict_from_checkpoint, predict_with_wandb # logging


# ADD new function to your existing predict.py
def predict_images_wandb(
    checkpoint,
    input_path,
    config='configs/config.yaml',
    output='outputs/predictions.csv',
    wandb_project=None,
    debug=False,
    batch_size=None,
    device=None
):
    """
    🔮 Predict document classes with WandB logging.
    
    Args:
        checkpoint (str): Path to the model checkpoint (.pth file)
        input_path (str): Path to an image file or directory of images
        config (str): Path to the config YAML file
        output (str): Path to save the prediction results CSV file
        wandb_project (str): WandB project name for logging (optional)
        debug (bool): Enable debug mode with detailed logging
        batch_size (int): Override batch size for prediction (optional)
        device (str): Override device ('cuda' or 'cpu', optional)
    
    Examples:
        # Predict with WandB logging
        python predict.py predict_images_wandb checkpoints/best_model.pth data/test/ --wandb-project document-classifier
        
        # Predict without WandB
        python predict.py predict_images_wandb checkpoints/best_model.pth data/test/
    """
    
    # Setup debug mode
    if debug:
        ic.enable()
        ic("🔥 Debug mode enabled!")
    else:
        ic.disable()
    
    # Load config
    try:
        config_data = load_config(config)
        ic("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Override config values if provided
    if batch_size:
        config_data['train']['batch_size'] = batch_size
    if device:
        config_data['device'] = device
    
    # Setup device
    device = torch.device(config_data['device'] if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Starting prediction with WandB logging...")
    print(f"📁 Checkpoint: {checkpoint}")
    print(f"📂 Input: {input_path}")
    print(f"💾 Output: {output}")
    if wandb_project:
        print(f"📊 WandB Project: {wandb_project}")
    
    try:
        # Run prediction with WandB
        results = predict_from_checkpoint(
            checkpoint_path=checkpoint,
            input_path=input_path,
            config=config_data,
            device=device,
            wandb_project=wandb_project
        )
        
        if results:
            # Save results
            df_results = pd.DataFrame(results)
            
            # Create output directory if needed
            output_dir = os.path.dirname(output)
            if output_dir and output_dir.strip():
                os.makedirs(output_dir, exist_ok=True)
            
            df_results.to_csv(output, index=False)
            
            # Create submission format
            output_df = pd.DataFrame({
                'ID': df_results['filename'],
                'target': df_results['predicted_target']
            }).sort_values('ID').reset_index(drop=True)
            
            submission_file = 'submission.csv'
            output_df.to_csv(submission_file, index=False)
            
            ic("✅ Prediction complete!")
            ic(f"Processed {len(results)} images")
            ic(f"Results saved to: {output}")
            ic(f"Submission file: {submission_file}")
            
            # Show summary
            if len(results) > 0:
                print(f"\n📋 Prediction Summary:")
                class_counts = df_results['predicted_class'].value_counts()
                for class_name, count in class_counts.head(10).items():
                    class_name_str = str(class_name)
                    percentage = (count / len(df_results)) * 100
                    print(f"  {class_name_str[:30]:30s}: {count:3d} ({percentage:5.1f}%)")
                
                if len(class_counts) > 10:
                    print(f"  ... and {len(class_counts) - 10} more classes")
                
                # Show confidence statistics
                confidences = df_results['confidence']
                print(f"\n🎯 Confidence Statistics:")
                print(f"  Mean confidence: {confidences.mean():.3f}")
                print(f"  Min confidence:  {confidences.min():.3f}")
                print(f"  Max confidence:  {confidences.max():.3f}")
                print(f"  Std confidence:  {confidences.std():.3f}")
        else:
            print(f"⚠️  No results to save!")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return

def predict_images(
    checkpoint,
    input_path,
    config='configs/config.yaml',
    output='outputs/predictions.csv',
    debug=False,
    batch_size=None,
    device=None
):
    """
    🔮 Predict document classes using a trained model.
    
    Args:
        checkpoint (str): Path to the model checkpoint (.pth file)
        input_path (str): Path to an image file or directory of images
        config (str): Path to the config YAML file
        output (str): Path to save the prediction results CSV file
        debug (bool): Enable debug mode with detailed logging
        batch_size (int): Override batch size for prediction (optional)
        device (str): Override device ('cuda' or 'cpu', optional)
    
    Examples:
        # Predict single image
        python predict.py predict_images checkpoints/best_model.pth data/test_image.jpg
        
        # Predict directory with debug
        python predict.py predict_images checkpoints/best_model.pth data/test_images/ --debug
        
        # Custom output and config
        python predict.py predict_images checkpoints/best_model.pth data/test/ --output results.csv --config my_config.yaml
    """
    
    # Setup debug mode
    if debug:
        ic.enable()
        ic("🔥 Debug mode enabled!")
    else:
        ic.disable()
    
    ic(f"Starting prediction with checkpoint: {checkpoint}")
    ic(f"Input path: {input_path}")
    ic(f"Config: {config}")
    ic(f"Output: {output}")
    
    # Validate inputs
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint file not found: {checkpoint}")
        return
    
    if not os.path.exists(input_path):
        print(f"❌ Input path not found: {input_path}")
        return
        
    if not os.path.exists(config):
        print(f"❌ Config file not found: {config}")
        return

    # Load config
    try:
        config_data = load_config(config)
        ic("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Override config values if provided
    if batch_size:
        config_data['train']['batch_size'] = batch_size
        ic(f"Batch size overridden to: {batch_size}")
    
    if device:
        config_data['device'] = device
        ic(f"Device overridden to: {device}")
    
    # Setup device
    device = torch.device(config_data['device'] if torch.cuda.is_available() else 'cpu')
    ic(f"Using device: {device}")
    
    if torch.cuda.is_available():
        ic(f"CUDA device: {torch.cuda.get_device_name()}")
        ic(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"🚀 Starting prediction...")
    print(f"📁 Checkpoint: {checkpoint}")
    print(f"📂 Input: {input_path}")
    print(f"💾 Output: {output}")
    
    try:
        # Run prediction
        results = predict_from_checkpoint(
            checkpoint_path=checkpoint,
            input_path=input_path,
            config=config_data,
            device=device
        )
        
        ic(f"Prediction completed. Got {len(results)} results")
        

        # FIX: Handle output path properly
        if not output or output.strip() == '':
            output = 'prediction.csv'
        
        # Create output directory if needed
        output_dir = os.path.dirname(output)
        if output_dir and output_dir.strip():
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Created output directory: {output_dir}")

        # Add this right before the save operation in predict_images:
        print(f"🔍 Debug - Output path: '{output}'")
        print(f"🔍 Debug - Output dir: '{os.path.dirname(output)}'")
        print(f"🔍 Debug - Output exists: {os.path.exists(os.path.dirname(output))}")

        # Save results
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output, index=False)
            print(f"✅ Prediction complete!")
            print(f"📊 Processed {len(results)} images")
            print(f"💾 Results saved to: {output}")
        else:
            print(f"⚠️  No results to save!")
            return

       
        output_dir = os.path.dirname(output)
        if output_dir:  # Only create directory if it's not empty
            os.makedirs(output_dir, exist_ok=True)
        else:
            # If output is just a filename (no directory), use current directory
            output = os.path.join('.', output)
      
        # Create output DataFrame and sort in one step
        output_df = pd.DataFrame({
            'ID': df_results['filename'],
            'target': df_results['predicted_target']
        }).sort_values('ID').reset_index(drop=True)

        output_df.to_csv('submission.csv', index=False)
        
        print(f"✅ Prediction complete!")
        print(f"📊 Processed {len(results)} images")
        print(f"💾 Results saved to: {output}")
        
    # Show summary
        if len(results) > 0:
            print(f"\n📋 Prediction Summary:")
            class_counts = df_results['predicted_class'].value_counts()
            for class_name, count in class_counts.head(10).items():
                class_name_str: str = str(class_name)  # Explicit conversion
                percentage = (count / len(df_results)) * 100
                print(f"  {class_name_str[:30]:30s}: {count:3d} ({percentage:5.1f}%)")
            
            if len(class_counts) > 10:
                print(f"  ... and {len(class_counts) - 10} more classes")
            # Show confidence statistics
            confidences = df_results['confidence']
            print(f"\n🎯 Confidence Statistics:")
            print(f"  Mean confidence: {confidences.mean():.3f}")
            print(f"  Min confidence:  {confidences.min():.3f}")
            print(f"  Max confidence:  {confidences.max():.3f}")
            print(f"  Std confidence:  {confidences.std():.3f}")
            
            # Show top predictions
            print(f"\n🏆 Top 5 Most Confident Predictions:")
            # Show top predictions
            print(f"\n🏆 Top 5 Most Confident Predictions:")
            top_predictions = df_results.nlargest(5, 'confidence')
            for _, row in top_predictions.iterrows():
                print(f"  {row['filename'][:25]:25s} → {row['predicted_class'][:30]:30s} ({row['confidence']:.3f})")
            
            # Show least confident predictions (potential issues)
            print(f"\n⚠️  5 Least Confident Predictions:")
            low_predictions = df_results.nsmallest(5, 'confidence')
            for _, row in low_predictions.iterrows():
                print(f"  {row['filename'][:25]:25s} → {row['predicted_class'][:30]:30s} ({row['confidence']:.3f})")
        
        ic("Prediction pipeline completed successfully")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        ic(f"Error details: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return

def predict_test_set(
    checkpoint,
    config='configs/config.yaml',
    output='outputs/test_predictions.csv',
    debug=False
):
    """
    🧪 Predict on the official test set for competition submission.
    
    Args:
        checkpoint (str): Path to the model checkpoint (.pth file)
        config (str): Path to the config YAML file
        output (str): Path to save the submission CSV file
        debug (bool): Enable debug mode
    
    Example:
        python predict.py predict_test_set checkpoints/best_model.pth --output submission.csv
    """
    
    if debug:
        ic.enable()
        ic("🔥 Debug mode enabled for test set prediction!")
    else:
        ic.disable()
    
    # Load config to get dataset path
    config_data = load_config(config)
    test_dir = os.path.join(config_data['data']['root_dir'], 'test')
    
    ic(f"Test directory: {test_dir}")
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    print(f"🧪 Predicting on test set...")
    print(f"📁 Test directory: {test_dir}")
    
    # Run prediction on test directory
    predict_images(
        checkpoint=checkpoint,
        input_path=test_dir,
        config='configs/config.yaml',
        output='outputs/predictions.csv',
        debug=debug
    )
    
    print(f"🎯 Test set prediction completed!")
    print(f"📄 Submission file ready: {output}")

def batch_predict(
    checkpoint,
    input_list,
    config='configs/config.yaml',
    output_dir='outputs/batch_predictions',
    debug=False
):
    """
    📦 Run predictions on multiple input sources in batch.
    
    Args:
        checkpoint (str): Path to the model checkpoint (.pth file)
        input_list (str): Path to text file containing list of input paths
        config (str): Path to the config YAML file
        output_dir (str): Directory to save batch prediction results
        debug (bool): Enable debug mode
    
    Example:
        python predict.py batch_predict checkpoints/best_model.pth input_list.txt
    """
    
    if debug:
        ic.enable()
        ic("🔥 Debug mode enabled for batch prediction!")
    else:
        ic.disable()
    
    if not os.path.exists(input_list):
        print(f"❌ Input list file not found: {input_list}")
        return
    
    # Read input paths
    with open(input_list, 'r') as f:
        input_paths = [line.strip() for line in f if line.strip()]
    
    ic(f"Found {len(input_paths)} input paths")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📦 Starting batch prediction...")
    print(f"📋 Processing {len(input_paths)} inputs")
    
    results_summary = []
    
    for i, input_path in enumerate(input_paths, 1):
        print(f"\n🔄 Processing {i}/{len(input_paths)}: {input_path}")
        
        if not os.path.exists(input_path):
            print(f"⚠️  Skipping missing path: {input_path}")
            continue
        
        # Generate output filename
        input_name = os.path.basename(input_path).replace('.', '_')
        output_file = os.path.join(output_dir, f"{input_name}_predictions.csv")
        
        try:
            predict_images(
                checkpoint=checkpoint,
                input_path=input_path,
                config=config,
                output=output_file,
                debug=False  # Disable debug for batch to reduce noise
            )
            
            # Read results for summary
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                results_summary.append({
                    'input_path': input_path,
                    'output_file': output_file,
                    'num_predictions': len(df),
                    'avg_confidence': df['confidence'].mean(),
                    'top_class': df['predicted_class'].mode().iloc[0] if len(df) > 0 else 'N/A'
                })
                print(f"✅ Completed: {len(df)} predictions, avg confidence: {df['confidence'].mean():.3f}")
            else:
                print(f"❌ Failed to generate predictions for {input_path}")
                
        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")
            continue

    # Save batch summary
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_file = os.path.join(output_dir, 'batch_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n📊 Batch Prediction Summary:")
        print(f"✅ Successfully processed: {len(results_summary)} inputs")
        print(f"📁 Results saved in: {output_dir}")
        print(f"📋 Summary saved to: {summary_file}")
        
        # Show summary statistics
        total_predictions = summary_df['num_predictions'].sum()
        avg_confidence = summary_df['avg_confidence'].mean()
        print(f"📈 Total predictions: {total_predictions}")
        print(f"🎯 Average confidence: {avg_confidence:.3f}")

    ic("Batch prediction completed")

def analyze_predictions(
    predictions_csv,
    debug=False,
    show_plots=True
):
    """
    📊 Analyze prediction results and generate insights.

    Args:
        predictions_csv (str): Path to predictions CSV file
        debug (bool): Enable debug mode
        show_plots (bool): Generate and show analysis plots

    Example:
        python predict.py analyze_predictions outputs/predictions.csv --show_plots
    """

    if debug:
        ic.enable()
        ic("🔥 Debug mode enabled for prediction analysis!")
    else:
        ic.disable()

    if not os.path.exists(predictions_csv):
        print(f"❌ Predictions file not found: {predictions_csv}")
        return

    # Load predictions
    df = pd.read_csv(predictions_csv)
    ic(f"Loaded {len(df)} predictions")

    print(f"📊 Prediction Analysis Report")
    print(f"=" * 50)

    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"Total predictions: {len(df)}")
    print(f"Unique classes predicted: {df['predicted_class'].nunique()}")
    print(f"Average confidence: {df['confidence'].mean():.3f}")
    print(f"Confidence std: {df['confidence'].std():.3f}")

    print(f"\n🏷️  Class Distribution:")
    class_counts = df['predicted_class'].value_counts()
    for class_name, count in class_counts.head(10).items():
        class_name_str = str(class_name)  # Convert to string
        percentage = (count / len(df)) * 100
        print(f"  {class_name_str[:35]:35s}: {count:4d} ({percentage:5.1f}%)")

    if len(class_counts) > 10:
        print(f"  ... and {len(class_counts) - 10} more classes")

    # Confidence analysis
    print(f"\n🎯 Confidence Analysis:")
    confidence_ranges = [
        (0.9, 1.0, "Very High"),
        (0.8, 0.9, "High"),
        (0.7, 0.8, "Medium"),
        (0.6, 0.7, "Low"),
        (0.0, 0.6, "Very Low")
    ]

    for min_conf, max_conf, label in confidence_ranges:
        count = len(df[(df['confidence'] >= min_conf) & (df['confidence'] < max_conf)])
        percentage = (count / len(df)) * 100
        print(f"  {label:10s} ({min_conf:.1f}-{max_conf:.1f}): {count:4d} ({percentage:5.1f}%)")

    # Potential issues
    print(f"\n⚠️  Potential Issues:")
    low_confidence = df[df['confidence'] < 0.5]
    if len(low_confidence) > 0:
        print(f"  • {len(low_confidence)} predictions with confidence < 0.5")
        print(f"  • Lowest confidence: {df['confidence'].min():.3f}")
        print(f"  • Files with lowest confidence:")
        for _, row in low_confidence.nsmallest(3, 'confidence').iterrows():
            print(f"    - {row['filename']}: {row['predicted_class']} ({row['confidence']:.3f})")
    else:
        print(f"  • No predictions with very low confidence detected")

    # Generate plots if requested
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print(f"\n📊 Generating analysis plots...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Confidence distribution
            confidence_data = df['confidence'].to_numpy()  # Convert to numpy array
            sns.histplot(confidence_data, bins=30, ax=axes[0,0], kde=True)
            axes[0,0].set_title('Confidence Score Distribution')
            axes[0,0].axvline(df['confidence'].mean(), color='red', linestyle='--', label=f'Mean: {df["confidence"].mean():.3f}')
            axes[0,0].legend()
            
            # Top classes
            top_classes = class_counts.head(10)
            sns.barplot(y=top_classes.index, x=top_classes.values, ax=axes[0,1])
            axes[0,1].set_title('Top 10 Predicted Classes')
            
            # Confidence by class (top 10)
            top_class_names = class_counts.head(10).index
            df_top = df[df['predicted_class'].isin(top_class_names)]
            sns.boxplot(data=df_top, y='predicted_class', x='confidence', ax=axes[1,0])
            axes[1,0].set_title('Confidence Distribution by Class (Top 10)')
            
            # Confidence vs prediction order (if filename has order info)
            if len(df) > 1:
                df_sample = df.sample(min(1000, len(df)))  # Sample for performance
                axes[1,1].scatter(range(len(df_sample)), df_sample['confidence'].values, alpha=0.6)
                axes[1,1].set_title('Confidence vs Prediction Order (Sample)')
                axes[1,1].set_xlabel('Prediction Index')
                axes[1,1].set_ylabel('Confidence')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = predictions_csv.replace('.csv', '_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Analysis plots saved to: {plot_path}")
            plt.show()
            
        except ImportError:
            print(f"⚠️  Matplotlib/Seaborn not available. Skipping plots.")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    ic("Prediction analysis completed")

def main():
    """
    🔥 Main entry point using Fire CLI.
    
    Available commands:
        predict_images        - Predict on single image or directory
        predict_images_wandb  - Predict with WandB logging
        predict_test_set      - Predict on official test set
        batch_predict         - Run batch predictions
        analyze_predictions   - Analyze prediction results
    
    Examples:
        # Regular prediction
        python predict.py predict_images checkpoints/best_model.pth data/test_image.jpg
        
        # Prediction with WandB logging
        python predict.py predict_images_wandb checkpoints/best_model.pth data/test/ --wandb-project document-classifier
        
        # Prediction without WandB
        python predict.py predict_images_wandb checkpoints/best_model.pth data/test/
        
        python predict.py predict_test_set checkpoints/best_model.pth
        python predict.py analyze_predictions outputs/predictions.csv
    """
    fire.Fire({
        'predict_images': predict_images,
        'predict_images_wandb': predict_images_wandb,  # ADD THIS LINE
        'predict_test_set': predict_test_set,
        'batch_predict': batch_predict,
        'analyze_predictions': analyze_predictions
    })

if __name__ == '__main__':
    main()