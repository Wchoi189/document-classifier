# scripts/merge_val_csvs.py
"""
Merges all val.csv files from stratified k-fold directories into a single file,
and provides functionality to aggregate validation images.
"""
import pandas as pd
from pathlib import Path
import sys
import shutil
import fire

def merge_validation_files():
    """
    Finds, reads, and merges all val.csv files from the k-fold directories.
    """
    # The script is expected to be run from the project root directory
    project_root = Path.cwd()
    
    # Define the base directory where the fold folders are located
    base_dir = project_root / "data/augmented_datasets/straitifed-k-fold-bk"
    
    # Create a pattern to find all 'val.csv' files within the fold directories
    file_pattern = "phase*_fold_*/metadata/val.csv"
    
    print(f"Searching for files in: {base_dir}")
    print(f"Using pattern: {file_pattern}\n")
    
    # Use glob to find all matching file paths
    val_csv_files = list(base_dir.glob(file_pattern))
    
    if not val_csv_files:
        print("❌ No 'val.csv' files found. Please check the base directory and pattern.")
        sys.exit(1)
        
    print(f"✅ Found {len(val_csv_files)} files to merge:")
    for f in val_csv_files:
        print(f"  - {f.relative_to(project_root)}")
        
    df_list = []
    for file in val_csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Could not read file {file}: {e}")
            
    if not df_list:
        print("❌ No data could be read from the found files. Exiting.")
        sys.exit(1)

    print("\nMerging files...")
    merged_df = pd.concat(df_list, ignore_index=True)
    
    output_path = base_dir / "combined_validation_set.csv"
    merged_df.to_csv(output_path, index=False)
    
    print("\n🎉 Merge complete!")
    print(f"Total rows in merged file: {len(merged_df)}")
    print(f"Combined file saved to: {output_path.relative_to(project_root)}")

def aggregate_validation_images():
    """
    Aggregates all validation images from folds 1-4 into the val directory of fold_0 for each phase.
    """
    project_root = Path.cwd()
    base_dir = project_root / "data/augmented_datasets/straitifed-k-fold-bk"
    
    # Define the different phases based on your directory structure
    phases = ["phase1_mild", "phase2_variety", "phase3_full"]

    print("🚀 Starting validation image aggregation.")

    for phase in phases:
        print(f"\n--- Processing Phase: {phase} ---")
        
        # Define the destination directory (e.g., .../phase1_mild_fold_0/val)
        dest_dir = base_dir / f"{phase}_fold_0/val"
        if not dest_dir.is_dir():
            print(f"⚠️ Destination directory not found, skipping phase: {dest_dir.relative_to(project_root)}")
            continue
        
        print(f"Destination: {dest_dir.relative_to(project_root)}")
        
        phase_moved_count = 0
        
        # Iterate through folds 1 to 4 to find source images
        for i in range(1, 5):
            source_dir = base_dir / f"{phase}_fold_{i}/val"
            
            if not source_dir.is_dir():
                print(f"  - Source directory not found, skipping: {source_dir.relative_to(project_root)}")
                continue
            
            # Find all .jpg files in the source validation directory
            images_to_move = list(source_dir.glob("*.jpg"))
            
            if not images_to_move:
                print(f"  - No images found in {source_dir.relative_to(project_root)}")
                continue
                
            print(f"  - Moving {len(images_to_move)} images from {source_dir.relative_to(project_root)}...")
            
            # Move each image
            for img_path in images_to_move:
                try:
                    # Use shutil.move to move the file
                    shutil.move(str(img_path), str(dest_dir))
                    phase_moved_count += 1
                except Exception as e:
                    print(f"    ❌ Error moving {img_path.name}: {e}")
                    
        print(f"  ✅ Moved a total of {phase_moved_count} images for phase '{phase}'.")

    print("\n🎉 Image aggregation complete!")

class TaskRunner:
    """A class to run different data management tasks using Fire CLI."""
    def merge_csvs(self):
        """Merges all val.csv files."""
        merge_validation_files()

    def aggregate_images(self):
        """Aggregates all validation images from folds 1-4 into fold_0 for each phase."""
        aggregate_validation_images()

if __name__ == "__main__":
    fire.Fire(TaskRunner)
