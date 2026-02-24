import numpy as np
import pandas as pd
from PIL import Image
import io
import os

def load_mnist_generator(data_dir="../mnist/mnist", max_samples=None):
    """
    Generator that yields (image, label) pairs from MNIST parquet files.
    
    Args:
        data_dir: Path to directory containing parquet files (relative to this file)
        max_samples: Limit total samples yielded (None for all)
    
    Yields:
        (image, label) tuples where:
        - image is a 28x28 numpy array (0.0 to 1.0)
        - label is an integer (0-9)
    """
    # Adjust path if running from nddplayground subdirectory
    train_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    
    try:
        df = pd.read_parquet(train_path)
    except FileNotFoundError:
        print(f"Error: Could not find MNIST data at {train_path}")
        return

    if max_samples:
        df = df.head(max_samples)
    
    for _, row in df.iterrows():
        # Image is stored as dict with 'bytes' key
        img_data = row['image']
        if isinstance(img_data, dict):
            img_bytes = img_data['bytes']
        else:
            img_bytes = img_data
        
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        yield img_array, int(row['label'])
