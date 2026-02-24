"""
Data Loading for Sequential Removal Experiment
===============================================
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader

from config import MNIST_DATA_DIR, BATCH_SIZE


class MNISTDataset(Dataset):
    """PyTorch Dataset for MNIST from parquet files."""
    
    def __init__(self, images, labels):
        self.images = torch.from_numpy(np.array(images)).float().unsqueeze(1)
        self.labels = torch.from_numpy(np.array(labels)).long()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_mnist(data_dir=None, max_train=None, max_test=None):
    """
    Load MNIST data from parquet files.
    
    Returns:
        train_dataset, test_dataset: PyTorch datasets
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", MNIST_DATA_DIR)
    
    train_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    test_path = os.path.join(data_dir, "test-00000-of-00001.parquet")
    
    def load_parquet(path, max_n):
        df = pd.read_parquet(path)
        if max_n is not None:
            df = df.head(max_n)
        
        images = []
        labels = []
        
        for _, row in df.iterrows():
            img_data = row['image']
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img_bytes = img_data['bytes']
            else:
                img_bytes = img_data
            
            img = Image.open(BytesIO(img_bytes)).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(int(row['label']))
        
        return images, labels
    
    print("Loading training data...")
    train_images, train_labels = load_parquet(train_path, max_train)
    print(f"  Loaded {len(train_images)} training samples")
    
    print("Loading test data...")
    test_images, test_labels = load_parquet(test_path, max_test)
    print(f"  Loaded {len(test_images)} test samples")
    
    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)
    
    return train_dataset, test_dataset


def get_data_loaders(train_dataset, test_dataset, batch_size=BATCH_SIZE):
    """Create data loaders."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
