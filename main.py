"""MNIST data loader for VeeThree experiments."""

import numpy as np
import pandas as pd
from PIL import Image
import io
import os


def load_mnist(data_dir="mnist/mnist", max_samples=None):
    """Load MNIST data from parquet files.

    Returns:
        (train_images, train_labels, test_images, test_labels)
        Images are numpy arrays (28×28), normalized to [0, 1].
    """
    train_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    test_path  = os.path.join(data_dir, "test-00000-of-00001.parquet")

    def load_parquet(path, max_n):
        df = pd.read_parquet(path)
        if max_n:
            df = df.head(max_n)
        images, labels = [], []
        for _, row in df.iterrows():
            img_data  = row['image']
            img_bytes = img_data['bytes'] if isinstance(img_data, dict) else img_data
            img       = Image.open(io.BytesIO(img_bytes)).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            labels.append(int(row['label']))
        return images, labels

    print("Loading training data...")
    train_images, train_labels = load_parquet(train_path, max_samples)
    print(f"  Loaded {len(train_images)} training samples")

    print("Loading test data...")
    test_images, test_labels = load_parquet(
        test_path, max_samples // 6 if max_samples else None
    )
    print(f"  Loaded {len(test_images)} test samples")

    return train_images, train_labels, test_images, test_labels
