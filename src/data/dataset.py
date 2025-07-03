import os
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DocumentDataset(Dataset):
    """문서 이미지 데이터셋 클래스"""
    def __init__(self, root_dir, transform=None, split='train', val_size=0.2, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        
        # 클래스 정보 로드
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 모든 샘플 수집
        all_samples = self._collect_samples()
        labels = [s[1] for s in all_samples]
        
        # train/val 분할
        if split in ['train', 'val'] and val_size > 0:
            train_samples, val_samples = train_test_split(
                all_samples, 
                test_size=val_size, 
                random_state=seed, 
                stratify=labels
            )
            self.samples = train_samples if split == 'train' else val_samples
        else: # 'test' 또는 전체 데이터셋
            self.samples = all_samples

    def _collect_samples(self):
        """데이터셋 루트 디렉토리에서 모든 이미지 경로와 라벨을 수집합니다."""
        samples = []
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    path = os.path.join(class_dir, fname)
                    item = (path, self.class_to_idx[class_name])
                    samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # OpenCV로 이미지 로드 후 RGB로 변환 (PIL보다 빠름)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, target
