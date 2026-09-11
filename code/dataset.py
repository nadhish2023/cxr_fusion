from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


class CXRDataset(Dataset):
    def __init__(self, split_path, image_root=None, train=False):
        self.rows = pd.read_csv(split_path)
        self.image_root = Path(image_root or config.IMAGE_ROOT).expanduser().resolve()
        transform_list = [transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))]
        if train:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_path = self.image_root / str(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by CSV is missing: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        tabular = torch.tensor(row[config.TABULAR_FEATURE_COLUMNS].to_numpy(dtype="float32"))
        labels = torch.tensor(row[config.LABEL_COLUMNS].to_numpy(dtype="float32"))
        return image, tabular, labels, str(row["subject_id"])
