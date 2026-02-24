"""Test FixedVector at n_pass=4 with 50 epochs."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_mnist, get_data_loaders
from model import SequentialCNN

train_ds, test_ds = load_mnist(max_train=10000, max_test=1000)
train_loader, test_loader = get_data_loaders(train_ds, test_ds, batch_size=32)

model = SequentialCNN('fixed_vector', 4, num_filters=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

for epoch in range(50):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(1) == y).sum().item()
        print(f'FixedVector n_pass=4 Epoch {epoch+1}: {100*correct/1000:.1f}%')
