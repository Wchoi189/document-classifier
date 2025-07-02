import argparse
import torch
import os
import pandas as pd

from utils.utils import load_config
from inference.predict import predict_from_checkpoint

def main(args):
    # We need some base config for model name, image size etc.
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}. A config is needed for model and data parameters.")
    config = load_config(args.config)

    print(f"Running prediction with checkpoint: {args.checkpoint}")
    print(f"Input path: {args.input}")
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    results = predict_from_checkpoint(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        config=config,
        device=device
    )

    # --- Save Results in Submission Format ---
    if results:
        df_results = pd.DataFrame(results)

        # Create the two-column submission DataFrame
        submission_df = pd.DataFrame({
            'ID': df_results['filename'],
            'target': df_results['predicted_target']
        }).sort_values('ID').reset_index(drop=True)

        # Save the final, sorted CSV
        output_path = "outputs/predictions1.csv" # Or use args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        
        print(f"✅ Prediction complete. Results saved to {output_path}")
        print("--- Top 5 Predictions ---")
        print(submission_df.head())
    else:
        print("⚠️ No results to save!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict document classes using a trained model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint (.pth file).")
    parser.add_argument('--input', type=str, required=True, help="Path to an image file or a directory of images.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help="Path to the config file to get model/data params.")
    parser.add_argument('--output', type=str, default='outputs/predictions.csv', help="Path to save the prediction results CSV file.")
    args = parser.parse_args()
    main(args)
