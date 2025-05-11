from PIL import Image
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SportsDataset(Dataset):
    def __init__(self, csv_file, file_path, split='train', transform=None):

        self.data_info = pd.read_csv(csv_file)
        # self.root_dir = os.path.join(file_path, split)
        self.root_dir = os.path.join(file_path, 'train')
        self.transform = transform
        self.split = split
        # Build string → index mapping from all labels in this split
        # if split == 'train':
        # assuming 2nd column is the label
        label_column = self.data_info.iloc[:, 1]
        classes = sorted(label_column.unique())
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(classes)}
        print(f"Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 1) Load image
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        # 2) Apply transforms
        if self.transform:
            image = self.transform(image)

        # if self.split=='train':
        # 3) String label → integer index
        label_str = self.data_info.iloc[idx, 1]
        label_idx = self.class_to_idx[label_str]

        # 4) Return image tensor, label tensor
        return image, torch.tensor(label_idx, dtype=torch.int8)
        # else:
        #     # 3) Return image tensor, label tensor
        #     return image
