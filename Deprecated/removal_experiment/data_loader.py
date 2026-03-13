"""
MNIST Data Loading
==================
Loads MNIST dataset from parquet files.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

from config import MNIST_DATA_DIR


def load_mnist(data_dir=None, max_train=None, max_test=None):
    """
    Load MNIST data from parquet files.
    
    Args:
        data_dir: Path to directory containing parquet files
        max_train: Limit training samples (None for all)
        max_test: Limit test samples (None for all)
    
    Returns:
        (train_images, train_labels, test_images, test_labels)
        Images are numpy arrays (28x28), normalized to 0-1
    """
    if data_dir is None:
        # Navigate up from removal_experiment to find mnist
        data_dir = os.path.join(os.path.dirname(__file__), "..", MNIST_DATA_DIR)
    
    train_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    test_path = os.path.join(data_dir, "test-00000-of-00001.parquet")
    
    def load_parquet(path, max_n):
        """Load images and labels from a parquet file."""
        df = pd.read_parquet(path)
        
        if max_n is not None:
            df = df.head(max_n)
        
        images = []
        labels = []
        
        for _, row in df.iterrows():
            # Extract image bytes and convert to numpy
            img_data = row['image']
            if isinstance(img_data, dict) and 'bytes' in img_data:
                img_bytes = img_data['bytes']
            else:
                img_bytes = img_data
            
            # Load image and normalize to 0-1
            from io import BytesIO
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
    
    return train_images, train_labels, test_images, test_labels


def get_examples_per_digit(images, labels, n_examples=1):
    """
    Get n_examples of each digit (0-9).
    
    Returns:
        dict mapping digit -> list of (image, label) tuples
    """
    examples = {d: [] for d in range(10)}
    
    for image, label in zip(images, labels):
        if len(examples[label]) < n_examples:
            examples[label].append((image, label))
        
        # Check if we have enough of each digit
        if all(len(v) >= n_examples for v in examples.values()):
            break
    
    return examples
