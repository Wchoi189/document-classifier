# scripts/merge_val_csvs.py
"""
Merges all val.csv files from stratified k-fold directories into a single file.
"""
import pandas as pd
from pathlib import Path
import sys

def merge_validation_files():
    """
    Finds, reads, and merges all val.csv files from the k-fold directories.
    """
    # The script is expected to be run from the project root directory
    # '/home/wb2x/projects/ml-projects/document-classifier'
    project_root = Path.cwd()
    
    # Define the base directory where the fold folders are located
    base_dir = project_root / "data/augmented_datasets/straitifed-k-fold-bk"
    
    # Create a pattern to find all 'val.csv' files within the fold directories
    # file_pattern = "phase1_mild_fold_*/metadata/val.csv"
    # file_pattern = "phase2_variety_fold_*/metadata/val.csv"
    file_pattern = "phase3_full_fold_*/metadata/val.csv"
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
        
    # List to hold each DataFrame
    df_list = []
    
    # Read each CSV file and append its DataFrame to the list
    for file in val_csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Could not read file {file}: {e}")
            
    if not df_list:
        print("❌ No data could be read from the found files. Exiting.")
        sys.exit(1)

    # Concatenate all DataFrames in the list into a single DataFrame
    print("\nMerging files...")
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Define the output path for the combined CSV file
    output_path = base_dir / "combined_validation_set.csv"
    
    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)
    
    print("\n🎉 Merge complete!")
    print(f"Total rows in merged file: {len(merged_df)}")
    print(f"Combined file saved to: {output_path.relative_to(project_root)}")

if __name__ == "__main__":
    merge_validation_files()
