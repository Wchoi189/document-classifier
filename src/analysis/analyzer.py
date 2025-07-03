# Final version of: src/analysis/analyzer.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from typing import Optional

def analyze(
    predictions_csv: str,
    ground_truth_csv: Optional[str] = None,
    meta_csv: str = "data/raw/metadata/meta.csv"
):
    """
    Performs a comprehensive analysis of model predictions.
    - If ground_truth_csv is provided, it calculates performance metrics.
    - Otherwise, it analyzes the prediction distribution and confidence.
    """
    print(f"📊 Starting Analysis for: {predictions_csv}")

    # --- 1. Load Data ---
    try:
        df_pred = pd.read_csv(predictions_csv)
        df_meta = pd.read_csv(meta_csv)
        class_names = df_meta.sort_values('target')['class_name'].tolist()
        print("✅ Prediction and metadata files loaded.")
    except FileNotFoundError as e:
        print(f"❌ Error: File not found. {e}")
        return

    # --- 2. Check if Ground Truth is available ---
    if ground_truth_csv and os.path.exists(ground_truth_csv):
        print("✅ Ground truth found. Performing full performance evaluation.")
        df_true = pd.read_csv(ground_truth_csv)
        
        # Standardize keys and merge
        df_pred['join_key'] = df_pred['filename'].str.replace(r'\.jpg|\.jpeg|\.png', '', regex=True)
        df_true['join_key'] = df_true['ID'].str.replace(r'\.jpg|\.jpeg|\.png', '', regex=True)
        df_merged = pd.merge(df_pred, df_true, on='join_key')

        if df_merged.empty:
            print("⚠️ Warning: No matching filenames found between predictions and ground truth. Skipping performance metrics.")
            analyze_predictions_only(df_pred, predictions_csv) # Fallback to prediction-only analysis
        else:
            analyze_with_ground_truth(df_merged, class_names, predictions_csv)
            
    else:
        print("⚠️ No ground truth provided. Analyzing prediction characteristics only.")
        analyze_predictions_only(df_pred, predictions_csv)

def analyze_predictions_only(df_pred, predictions_csv):
    """Analyzes the distribution and confidence of predictions."""
    print("\n📈 **Prediction Statistics:**")
    print(f"Total predictions: {len(df_pred)}")
    print(f"Unique classes predicted: {df_pred['predicted_class'].nunique()}")
    print(f"Average confidence: {df_pred['confidence'].mean():.3f}")

    print("\n🏷️ **Predicted Class Distribution (Top 10):**")
    class_counts = df_pred['predicted_class'].value_counts()
    print(class_counts.head(10))

    # Plotting logic for predictions only
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(df_pred['confidence'], bins=30, ax=ax, kde=True)
    ax.set_title('Prediction Confidence Distribution')
    ax.axvline(df_pred['confidence'].mean(), color='red', linestyle='--', label=f"Mean: {df_pred['confidence'].mean():.3f}")
    ax.legend()
    
    plot_path = predictions_csv.replace('.csv', '_confidence_analysis.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\n💾 Confidence plot saved to: {plot_path}")

def analyze_with_ground_truth(df_merged, class_names, predictions_csv):
    """Performs a full analysis against ground truth labels."""
    y_true = df_merged['target']
    y_pred = df_merged['predicted_target']

    print("\n" + "="*50)
    print("📈 **Overall Performance Metrics**")
    print("="*50)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\n📋 **Classification Report:**")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    # ... (Error Analysis and Confusion Matrix plotting can go here as before) ...
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plot_output_path = predictions_csv.replace('.csv', '_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, plot_output_path)


def plot_confusion_matrix(cm, class_names, output_path):
    """Creates and saves a confusion matrix plot."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n💾 Confusion matrix plot saved to: {output_path}")