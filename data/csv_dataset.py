import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CSVDocumentDataset(Dataset):
    """Document dataset that reads labels from CSV file"""
    
    def __init__(self, root_dir, csv_file, meta_file, transform=None, split='train', val_size=0.2, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        
        # Read CSV files
        self.df = pd.read_csv(csv_file)
        self.meta_df = pd.read_csv(meta_file)
        
        print(f"Loaded train CSV: {self.df.shape} samples")
        print(f"Loaded meta CSV: {self.meta_df.shape} classes")
        
        # Create class mappings
        self.target_to_class = dict(zip(self.meta_df['target'], self.meta_df['class_name']))
        self.classes = [self.target_to_class[i] for i in sorted(self.target_to_class.keys())]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Found {len(self.classes)} classes")
        print(f"Sample classes: {self.classes[:5]}...")
        
        # Split data if needed
        if split in ['train', 'val'] and val_size > 0:
            train_df, val_df = train_test_split(
                self.df, 
                test_size=val_size, 
                random_state=seed, 
                stratify=self.df['target']
            )
            self.df = train_df if split == 'train' else val_df
        
        print(f"Dataset split '{split}' has {len(self.df)} samples")
        
        # Verify some images exist
        self._verify_sample_images()
    
    def _verify_sample_images(self, sample_size=5):
        """Check if a sample of image files exist"""
        sample_files = self.df['ID'].head(sample_size).tolist()
        missing_count = 0
        
        for filename in sample_files:
            img_path = os.path.join(self.root_dir, 'train', filename)
            if not os.path.exists(img_path):
                missing_count += 1
                print(f"⚠️  Missing: {img_path}")
        
        if missing_count == 0:
            print(f"✅ Sample verification passed - all {sample_size} sample images found")
        else:
            print(f"⚠️  {missing_count}/{sample_size} sample images missing")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, 'train', row['ID'])
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # The target is already the class index (0-16)
        return image, row['target']