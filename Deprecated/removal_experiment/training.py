"""
Training Pipeline
=================
Training loop with progress tracking and metrics collection.
"""

import numpy as np
import time
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, LR_DECAY


def train_model(cnn, train_images, train_labels, 
                epochs=EPOCHS, learning_rate=LEARNING_RATE, 
                lr_decay=LR_DECAY, batch_size=BATCH_SIZE):
    """
    Train the CNN on the dataset.
    
    Args:
        cnn: SparseCNN instance
        train_images: List of 2D numpy arrays
        train_labels: List of integer labels
        epochs: Number of training epochs
        learning_rate: Initial SGD learning rate
        lr_decay: Learning rate decay per epoch
        batch_size: Mini-batch size for progress reporting
    
    Returns:
        history: Dict with 'epoch_losses' and 'epoch_accuracies'
    """
    n_samples = len(train_images)
    history = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'epoch_times': []
    }
    
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    print(f"Samples: {n_samples}")
    print(f"Epochs: {epochs}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"LR decay: {lr_decay}")
    print()
    
    total_start_time = time.time()
    current_lr = learning_rate
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        
        epoch_loss = 0.0
        correct = 0
        
        for i, idx in enumerate(indices):
            image = train_images[idx]
            label = train_labels[idx]
            
            # Training step with current learning rate
            loss = cnn.train_step(image, label, current_lr)
            epoch_loss += loss
            
            # Check prediction for accuracy
            prediction = cnn.predict(image)
            if prediction == label:
                correct += 1
            
            # Progress update every batch_size samples
            if (i + 1) % batch_size == 0 or (i + 1) == n_samples:
                progress = (i + 1) / n_samples * 100
                avg_loss = epoch_loss / (i + 1)
                acc = correct / (i + 1) * 100
                print(f"\rEpoch {epoch + 1}/{epochs} | "
                      f"Progress: {progress:5.1f}% | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {acc:.1f}% | "
                      f"LR: {current_lr:.5f}", end="", flush=True)
        
        epoch_time = time.time() - epoch_start_time
        epoch_loss_avg = epoch_loss / n_samples
        epoch_accuracy = correct / n_samples
        
        history['epoch_losses'].append(epoch_loss_avg)
        history['epoch_accuracies'].append(epoch_accuracy)
        history['epoch_times'].append(epoch_time)
        
        print(f"\rEpoch {epoch + 1}/{epochs} | "
              f"Loss: {epoch_loss_avg:.4f} | "
              f"Acc: {epoch_accuracy * 100:.2f}% | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {current_lr:.5f}")
        
        # Decay learning rate
        current_lr *= lr_decay
    
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time / 60:.1f} minutes")
    
    history['total_time'] = total_time
    
    return history


def evaluate_model(cnn, test_images, test_labels):
    """
    Evaluate the CNN on a test set.
    
    Args:
        cnn: SparseCNN instance
        test_images: List of 2D numpy arrays
        test_labels: List of integer labels
    
    Returns:
        overall_accuracy: Float (0-1)
        per_digit_accuracy: Dict mapping digit -> accuracy
        confusion_matrix: 10x10 numpy array
    """
    n_samples = len(test_images)
    correct = 0
    
    # Per-digit tracking
    digit_correct = {d: 0 for d in range(10)}
    digit_total = {d: 0 for d in range(10)}
    
    # Confusion matrix
    confusion = np.zeros((10, 10), dtype=int)
    
    print("\nEvaluating...")
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        prediction = cnn.predict(image)
        
        if prediction == label:
            correct += 1
            digit_correct[label] += 1
        
        digit_total[label] += 1
        confusion[label, prediction] += 1
        
        if (i + 1) % 1000 == 0:
            print(f"\r  Processed {i + 1}/{n_samples}", end="", flush=True)
    
    overall_accuracy = correct / n_samples
    
    per_digit_accuracy = {}
    for d in range(10):
        if digit_total[d] > 0:
            per_digit_accuracy[d] = digit_correct[d] / digit_total[d]
        else:
            per_digit_accuracy[d] = 0.0
    
    print(f"\r  Processed {n_samples}/{n_samples}")
    print(f"\nOverall Test Accuracy: {overall_accuracy * 100:.2f}% ({correct}/{n_samples})")
    
    print("\nPer-Digit Accuracy:")
    for d in range(10):
        acc = per_digit_accuracy[d] * 100
        bar = "█" * int(acc / 5)
        print(f"  {d}: {acc:5.1f}% {bar}")
    
    return overall_accuracy, per_digit_accuracy, confusion
