from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import os
from PIL import Image

class HealthDataset(Dataset):
    """HealthDataset dataset."""

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.image_names=os.listdir(self.data_dir)
        self.transform = None


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
       image=self.read_image(idx)
       


    def read_image(self, index):
        image_path = self.data_dir+self.image_names[index]
        return Image.open(image_path)