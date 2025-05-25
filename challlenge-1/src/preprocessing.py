import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Label encodings
soil2idx = {
    'Black': 0,
    'Laterite': 1,
    'Red': 2,
    'Alluvial': 3,
    'Desert': 4
}

idx2soil = {v: k for k, v in soil2idx.items()}


class SoilDataset(Dataset):
    def __init__(self, img_dir, csv_path=None, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        if is_test:
            self.data = pd.read_csv(csv_path)
            self.labels = None
        else:
            self.data = pd.read_csv(csv_path)
            self.labels = self.data['soil_type'].map(soil2idx).values

        self.image_ids = self.data['image_id'].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_ids[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, self.image_ids[idx]
        else:
            label = self.labels[idx]
            return image, label


# Standard transformation
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
