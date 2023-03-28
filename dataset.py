import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import pandas as pd

from global_variables import IMG_HEIGHT, IMG_WIDTH

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        file_name = self.img_labels.iloc[idx, 0].split('/')[-1]
        img_dir = '/'.join(self.img_labels.iloc[idx, 0].split('/')[:-1])
        image = Image.open(img_path)
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image = torch.tensor(np.array(image), dtype=torch.float32).transpose(1,2).transpose(0,1)
        label = self.img_labels.iloc[idx, -1] # under_extrusion
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, file_name, img_dir
