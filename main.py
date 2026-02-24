# MNIST Training Pipeline

import numpy as np
import pandas as pd
from PIL import Image
import io
import os
import argparse
from typical_cnn import CNN, improve_filters, get_outputs, get_best_filters, save_activation_maps


def load_mnist(data_dir="mnist/mnist", max_samples=None):
    """
    Load MNIST data from parquet files.
    
    Args:
        data_dir: Path to directory containing parquet files
        max_samples: Limit samples (None for all)
    
    Returns:
        (train_images, train_labels, test_images, test_labels)
        Images are numpy arrays (28x28), normalized to 0-1
    """
    train_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
    test_path = os.path.join(data_dir, "test-00000-of-00001.parquet")
    
    def load_parquet(path, max_n):
        df = pd.read_parquet(path)
        if max_n:
            df = df.head(max_n)
        
        images = []
        labels = []
        
        for _, row in df.iterrows():
            # Image is stored as dict with 'bytes' key
            img_data = row['image']
            if isinstance(img_data, dict):
                img_bytes = img_data['bytes']
            else:
                img_bytes = img_data
            
            img = Image.open(io.BytesIO(img_bytes)).convert('L')
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(int(row['label']))
        
        return images, labels
    
    print("Loading training data...")
    train_images, train_labels = load_parquet(train_path, max_samples)
    print(f"  Loaded {len(train_images)} training samples")
    
    print("Loading test data...")
    test_images, test_labels = load_parquet(test_path, max_samples // 6 if max_samples else None)
    print(f"  Loaded {len(test_images)} test samples")
    
    return train_images, train_labels, test_images, test_labels

def train(cnn, images, labels, epochs=5, learning_rate=0.01, batch_size=32):
    """
    Train the CNN on the dataset.
    
    Args:
        cnn: CNN instance
        images: List of 2D numpy arrays
        labels: List of integer labels
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Mini-batch size (for progress reporting)
    """
    n_samples = len(images)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        total_loss = 0
        correct = 0
        
        for i, idx in enumerate(indices):
            image = images[idx]
            label = labels[idx]
            
            # Forward + backward
            loss = improve_filters(cnn, image, label, learning_rate)
            total_loss += loss
            
            # Check prediction
            output = get_outputs(cnn, image)
            if np.argmax(output) == label:
                correct += 1
            
            # Progress update
            if (i + 1) % batch_size == 0:
                avg_loss = total_loss / (i + 1)
                acc = correct / (i + 1) * 100
                print(f"\r  Epoch {epoch+1}/{epochs} | "
                      f"Sample {i+1}/{n_samples} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {acc:.1f}%", end="")
        
        # Epoch summary
        avg_loss = total_loss / n_samples
        acc = correct / n_samples * 100
        print(f"\r  Epoch {epoch+1}/{epochs} complete | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {acc:.1f}%          ")

def evaluate(cnn, images, labels):
    """
    Evaluate the CNN on a test set.
    
    Returns: accuracy (0-1)
    """
    correct = 0
    for image, label in zip(images, labels):
        output = get_outputs(cnn, image)
        if np.argmax(output) == label:
            correct += 1
    
    accuracy = correct / len(images)
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{len(images)})")
    return accuracy

def visualize_filters(cnn, sample_image, output_dir="visualizations"):
    """
    Save filter visualizations and activation maps for inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the sample input
    img = Image.fromarray((sample_image * 255).astype(np.uint8), mode='L')
    img.save(os.path.join(output_dir, "input.png"))
    
    # Get activations
    _, cache = cnn.forward(sample_image)
    
    # Save conv activations for each layer
    for layer_idx, conv_out in enumerate(cache['conv_outputs']):
        layer_dir = os.path.join(output_dir, f"conv_layer_{layer_idx}")
        save_activation_maps(conv_out, layer_dir, prefix="activation")
        
        # Also save the filters themselves
        filter_dir = os.path.join(output_dir, f"filters_layer_{layer_idx}")
        os.makedirs(filter_dir, exist_ok=True)
        for f_idx, filt in enumerate(cnn.conv_layers[layer_idx]):
            # Normalize filter for visualization
            f_norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
            f_img = (f_norm * 255).astype(np.uint8)
            # Scale up for visibility
            f_img = Image.fromarray(f_img, mode='L').resize((64, 64), Image.NEAREST)
            f_img.save(os.path.join(filter_dir, f"filter_{f_idx:03d}.png"))
    
    print(f"Visualizations saved to {output_dir}/")

def main():
    """Main training pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNIST CNN Training Pipeline')
    parser.add_argument('--model', type=str, default='model',
                        help='Path to save/load model (default: model)')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training, just evaluate existing model')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (ignore existing model)')
    args = parser.parse_args()
    
    model_path = args.model
    if not model_path.endswith('.npz'):
        model_path_full = model_path + '.npz'
    else:
        model_path_full = model_path
    
    print("=" * 50)
    print("MNIST CNN Training Pipeline")
    print("=" * 50)
    
    # Configuration
    config = {
        'max_samples': 1000,      # Use subset for faster testing (None for full)
        'epochs': 5,
        'learning_rate': 0.01,     # Higher LR for faster learning
        'conv_filters': [6, 6],
        'filter_size': 3,
        'pool_size': 2,
        'pool_method': 1,         # Max pooling
        'dense_size': 128
    }
    
    print("\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    
    # Load data
    print("\n" + "-" * 50)
    train_images, train_labels, test_images, test_labels = load_mnist(
        max_samples=config['max_samples']
    )
    
    # Initialize or load network
    print("\n" + "-" * 50)
    if os.path.exists(model_path_full) and not args.fresh:
        print(f"Loading existing model from {model_path_full}...")
        cnn = CNN.load(model_path)
    else:
        print("Initializing new CNN...")
        cnn = CNN(
            input_shape=(28, 28),
            num_classes=10,
            conv_filters=config['conv_filters'],
            filter_size=config['filter_size'],
            pool_size=config['pool_size'],
            pool_method=config['pool_method'],
            dense_size=config['dense_size']
        )
    print(f"  Flatten size: {cnn.flatten_size}")
    print(f"  Dense layer: {cnn.flatten_size} → {cnn.dense_size} → 10")
    
    # Train (unless --no-train flag)
    if not args.no_train:
        print("\n" + "-" * 50)
        print("Training...")
        train(cnn, train_images, train_labels, 
              epochs=config['epochs'], 
              learning_rate=config['learning_rate'])
        
        # Save model after training
        print("\n" + "-" * 50)
        cnn.save(model_path)
    
    # Evaluate
    print("\n" + "-" * 50)
    print("Evaluating on test set...")
    evaluate(cnn, test_images, test_labels)

    # Save filter visualizations
    print("\n" + "-" * 50)
    visualize_filters(cnn, train_images[0])

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
