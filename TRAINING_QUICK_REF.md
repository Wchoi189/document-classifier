
🔥 QUICK REFERENCE FOR TRAINING TEAM
==================================================

📁 Dataset Files:
  • train.csv: 1,570 samples, 17 classes
  • meta.csv: Class ID to name mapping
  • Images: src/data/raw/train/*.jpg

⚙️  Recommended Config Updates:
  • image_size: 224
  • batch_size: 32
  • model: resnet34
  • use_weighted_sampling: False

🎯 Key Challenges:
  • Class imbalance (ratio: 2.2)
  • Variable image sizes
  • 0.0% missing files in sample

💡 Training Tips:
  • Monitor both accuracy and F1-score
  • Use stratified validation split
  • Implement early stopping
  • Consider focal loss for severe imbalance
  💡 Training Tips:
  • Monitor both accuracy and F1-score
  • Use stratified validation split
  • Implement early stopping
  • Consider focal loss for severe imbalance

🚀 Ready to Train:
  python -m scripts.train --config configs/config.yaml
