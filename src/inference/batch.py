# In src/inference/batch.py
import os
import torch
import pandas as pd
from pathlib import Path

from .predictor import predict_from_checkpoint, save_predictions

def run_batch(checkpoint_path: str, input_list_file: str, config: dict):
    """
    Engine for running predictions on a list of inputs from a file.
    """
    print(f"📦 Starting batch prediction for {input_list_file}...")
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    # --- 1. Construct the specific output directory for this batch run ---
    paths_config = config.get('paths', {})
    base_output_dir = Path(paths_config.get('output_dir', 'outputs'))
    batch_output_dir = base_output_dir / paths_config.get('batch_dir', 'batch') / Path(input_list_file).stem
    
    if not os.path.exists(input_list_file):
        print(f"❌ Input list file not found: {input_list_file}")
        return

    with open(input_list_file, 'r') as f:
        input_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(input_paths)} items to process. Results will be in: {batch_output_dir}")
    
    batch_summary = []

    for i, path in enumerate(input_paths):
        print(f"\n--- Processing item {i+1}/{len(input_paths)}: {path} ---")
        if not os.path.exists(path):
            print(f"⚠️ Skipping missing path: {path}")
            continue

        results = predict_from_checkpoint(
            checkpoint_path=checkpoint_path,
            input_path=path,
            config=config,
            device=device
        )

        if results:
            # Save individual results to the specific batch directory
            saved_files = save_predictions(results, output_dir=batch_output_dir)
            batch_summary.append({ 'input_source': path, 'output_files': saved_files })

    # --- 2. Save the final summary file to the same directory ---
    if batch_summary:
        df_summary = pd.DataFrame(batch_summary)
        summary_filename_str = paths_config.get('batch_summary_filename', 'batch_summary.csv')
        summary_file = batch_output_dir / summary_filename_str # Use the correct directory
        df_summary.to_csv(summary_file, index=False)
        
        # Now the summary printout uses the correct variables
        print("\n" + "✨" * 20)
        print("  📊 Batch Prediction Summary 📊")
        print("✨" * 20)
        print(f"✅ Successfully processed: {len(batch_summary)} input sources.")
        print("-" * 40)
        print(f"💾 All results saved in: {batch_output_dir}")
        print(f"📋 Final summary saved to: {summary_file}")
        print("✨" * 20)
    else:
        print("\n⚠️ Batch run finished with no results.")