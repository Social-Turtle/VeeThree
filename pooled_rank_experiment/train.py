"""
Training Loop for Sequential Removal Experiment
================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from config import EPOCHS, LEARNING_RATE, LR_DECAY


def train_model(model, train_loader, test_loader, epochs=EPOCHS, 
                learning_rate=LEARNING_RATE, lr_decay=LR_DECAY, device='cpu'):
    """
    Train the model.
    
    Returns:
        history: Dict with training metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_times': []
    }
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}, decay: {lr_decay}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={train_loss:.4f}, "
              f"Train={train_acc:.1f}%, "
              f"Test={test_acc:.1f}%, "
              f"Time={epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"-" * 60)
    print(f"Training complete in {total_time/60:.1f} minutes")
    
    history['total_time'] = total_time
    
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def get_confusion_matrix(model, test_loader, device='cpu'):
    """Compute confusion matrix and per-digit accuracy."""
    model.eval()
    confusion = np.zeros((10, 10), dtype=int)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for true, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[true][pred] += 1
    
    # Per-digit accuracy
    per_digit = {}
    for digit in range(10):
        total = confusion[digit].sum()
        correct = confusion[digit][digit]
        per_digit[digit] = 100.0 * correct / total if total > 0 else 0.0
    
    return confusion, per_digit
