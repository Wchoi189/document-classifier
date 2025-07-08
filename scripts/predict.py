import sys
from pathlib import Path

# Add parent to path for setup import
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.project_setup import setup_project_environment
setup_project_environment()
from typing import Optional

import torch
import fire
from pathlib import Path
from src.utils.utils import load_config
from src.inference.predictor import predict_from_checkpoint, save_predictions
from src.inference.batch import run_batch
from src.analysis.analyzer import analyze as analyze_file

class PredictionCLI:
    """A clean, organized CLI for all prediction-related tasks."""

    def run(self,
            input_path: str,
            checkpoint_path: Optional[str] = None, # Make this optional
            model_name: Optional[str] = None,
            use_last: bool = False,                 
            config_path: str = "configs/config.yaml", # default if not specified
            device: str = "cuda", # Default to CUDA if available
            wandb: bool = False):
        
        """Runs model inference on an input and saves the timestamped results."""
        print("🚀 Starting prediction...")
        config = load_config(config_path)

        


        # Determine the checkpoint path
        if use_last:
            paths_config = config.get('paths', {})
            model_dir = Path(paths_config.get('output_dir', 'outputs')) / paths_config.get('model_dir', 'models')
            final_checkpoint_path = model_dir / "last_model.pth"
            print(f"🔎 Using last saved model: {final_checkpoint_path}")
        elif checkpoint_path:
            final_checkpoint_path = Path(checkpoint_path)
        else:
            raise ValueError("You must specify either --checkpoint_path or use the --use-last flag.")


        if not final_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {final_checkpoint_path}")
   

        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        wandb_project = config.get("wandb", {}).get("project") if wandb else None

        results = predict_from_checkpoint(
            checkpoint_path=final_checkpoint_path,
            input_path=input_path,
            config=config,
            device=torch_device,
            wandb_project=wandb_project,
            model_name_override=model_name
        )
        save_predictions(results, config=config)
        print("✨ Prediction finished successfully.")

    def on_test_set(self, checkpoint_path: str, config_path: str = "configs/config.yaml", output: Optional[str] = None, wandb: bool = False):
        """Runs prediction on the official test set defined in the config."""
        print("🧪 Running prediction on the official test set...")
        config = load_config(config_path)
        test_set_path = config.get('paths', {}).get('test_data_dir''data/raw/test')
        self.run(
            input_path=test_set_path,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            wandb=wandb
        )
        
        # If output is specified, rename the saved predictions file
        if output:
            import shutil
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictions_dir = Path(config.get('paths', {}).get('predictions_dir', 'predictions'))
            latest_file = f"{predictions_dir}/predictions_{timestamp}.csv"
            shutil.copy(latest_file, output)
            print(f"✅ Predictions saved to {output}")

    def batch(self, checkpoint_path: str, input_list_file: str, config_path: str = "configs/config.yaml"):
        """Runs predictions on a list of inputs from a text file."""
        config = load_config(config_path)
        run_batch(
            checkpoint_path=checkpoint_path,
            input_list_file=input_list_file,
            config=config
        )


    def analyze(self,
                predictions_csv: str,
                ground_truth_csv: Optional[str] = None): # Make optional
        """
        Analyzes prediction results. If ground truth is provided, a full
        performance evaluation is run. Otherwise, analyzes prediction characteristics.
        """
        analyze_file(
            predictions_csv=predictions_csv,
            ground_truth_csv=ground_truth_csv
        )

if __name__ == '__main__':
    fire.Fire(PredictionCLI)