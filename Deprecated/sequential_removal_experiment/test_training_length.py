"""
Quick test: Does FixedRank just need more training at higher n_pass?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_mnist, get_data_loaders
from model import SequentialCNN

def train_and_track(encoding_type, n_pass, epochs=50, num_filters=4):
    """Train and return epoch-by-epoch test accuracy."""
    
    print(f"\n{'='*60}")
    print(f"Training: {encoding_type} n_pass={n_pass} for {epochs} epochs")
    print(f"{'='*60}")
    
    # Load data
    train_ds, test_ds = load_mnist(max_train=10000, max_test=1000)
    train_loader, test_loader = get_data_loaders(train_ds, test_ds, batch_size=32)
    
    # Create model
    model = SequentialCNN(encoding_type, n_pass, num_filters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100.0 * correct / total
        test_accs.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: {test_acc:.1f}%")
    
    return test_accs


if __name__ == "__main__":
    epochs = 50
    
    # Test FixedRank at different n_pass values
    results = {}
    
    for n_pass in [1, 2, 3, 4]:
        results[n_pass] = train_and_track('fixed_rank', n_pass, epochs=epochs)
    
    # Print comparison at key epochs
    print("\n" + "="*60)
    print("FixedRank Test Accuracy by Epoch")
    print("="*60)
    print(f"{'Epoch':<10} {'n=1':<10} {'n=2':<10} {'n=3':<10} {'n=4':<10}")
    print("-"*50)
    
    for epoch in [9, 19, 29, 39, 49]:  # Epochs 10, 20, 30, 40, 50
        row = f"{epoch+1:<10}"
        for n_pass in [1, 2, 3, 4]:
            row += f"{results[n_pass][epoch]:.1f}%{'':<5}"
        print(row)
    
    print("\n" + "="*60)
    print("Final Accuracies (Epoch 50)")
    print("="*60)
    for n_pass in [1, 2, 3, 4]:
        print(f"n_pass={n_pass}: {results[n_pass][-1]:.1f}%")
