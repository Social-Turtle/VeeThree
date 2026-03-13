"""
Models for Performance Comparison Experiment
=============================================

Two model types:
1. EncodedCNN: Fixed edge filters → Rank encoding → Lowest-passing pooling → Conv2
2. ConventionalCNN: Learned Conv1 → Standard max pooling → Conv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rank_encodings import get_encoding
from custom_pooling import LowestPassingMaxPool
from config import (
    NUM_FILTERS_CONV1, NUM_FILTERS_CONV2, CONV2_KERNEL_SIZE,
    POOL_SIZE, DENSE_HIDDEN_SIZE, NUM_CLASSES
)


def get_edge_filters(kernel_size=3):
    """
    Returns 4 edge detection filters of specified size.
    
    Args:
        kernel_size: 3, 4, or 5
    
    Returns:
        Tensor of shape (4, 1, kernel_size, kernel_size)
    """
    k = kernel_size
    filters = []
    
    if k == 3:
        # Standard 3x3 edge detectors
        horiz = np.array([[ 1,  1,  1], [ 0,  0,  0], [-1, -1, -1]], dtype=np.float32)
        vert = np.array([[ 1,  0, -1], [ 1,  0, -1], [ 1,  0, -1]], dtype=np.float32)
        diag45 = np.array([[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]], dtype=np.float32)
        diag135 = np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]], dtype=np.float32)
        filters = [horiz, vert, diag45, diag135]
        
    elif k == 4:
        # 4x4 edge detectors
        horiz = np.array([
            [ 1,  1,  1,  1],
            [ 0.5, 0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5, -0.5],
            [-1, -1, -1, -1]
        ], dtype=np.float32)
        
        vert = np.array([
            [ 1,  0.5, -0.5, -1],
            [ 1,  0.5, -0.5, -1],
            [ 1,  0.5, -0.5, -1],
            [ 1,  0.5, -0.5, -1]
        ], dtype=np.float32)
        
        diag45 = np.array([
            [ 2,  1,  0, -1],
            [ 1,  0, -1, -2],
            [ 0, -1, -2, -1],
            [-1, -2, -1,  0]
        ], dtype=np.float32)
        
        diag135 = np.array([
            [-1,  0,  1,  2],
            [-2, -1,  0,  1],
            [-1, -2, -1,  0],
            [ 0, -1, -2, -1]
        ], dtype=np.float32)
        
        filters = [horiz, vert, diag45, diag135]
        
    elif k == 5:
        # 5x5 edge detectors
        horiz = np.array([
            [ 1,  1,  1,  1,  1],
            [ 0.5, 0.5, 0.5, 0.5, 0.5],
            [ 0,  0,  0,  0,  0],
            [-0.5, -0.5, -0.5, -0.5, -0.5],
            [-1, -1, -1, -1, -1]
        ], dtype=np.float32)
        
        vert = np.array([
            [ 1,  0.5,  0, -0.5, -1],
            [ 1,  0.5,  0, -0.5, -1],
            [ 1,  0.5,  0, -0.5, -1],
            [ 1,  0.5,  0, -0.5, -1],
            [ 1,  0.5,  0, -0.5, -1]
        ], dtype=np.float32)
        
        diag45 = np.array([
            [ 2,  1.5,  1,  0.5,  0],
            [ 1.5,  1,  0.5,  0, -0.5],
            [ 1,  0.5,  0, -0.5, -1],
            [ 0.5,  0, -0.5, -1, -1.5],
            [ 0, -0.5, -1, -1.5, -2]
        ], dtype=np.float32)
        
        diag135 = np.array([
            [ 0,  0.5,  1,  1.5,  2],
            [-0.5,  0,  0.5,  1,  1.5],
            [-1, -0.5,  0,  0.5,  1],
            [-1.5, -1, -0.5,  0,  0.5],
            [-2, -1.5, -1, -0.5,  0]
        ], dtype=np.float32)
        
        filters = [horiz, vert, diag45, diag135]
    
    else:
        raise ValueError(f"Unsupported kernel size: {k}. Use 3, 4, or 5.")
    
    # Stack and normalize
    filters = np.stack(filters, axis=0)
    for i in range(len(filters)):
        norm = np.linalg.norm(filters[i])
        if norm > 0:
            filters[i] = filters[i] / norm
    
    return torch.from_numpy(filters).unsqueeze(1)  # (4, 1, k, k)


class EncodedCNN(nn.Module):
    """
    CNN with fixed edge filters, rank encoding, and lowest-passing pooling.
    
    Architecture:
        Conv1 (fixed edge filters) → ReLU → Encoding → LowestPassingMaxPool →
        Conv2 (16 learned 3x3 filters) → ReLU → Pool →
        Flatten → Dense → ReLU → Dense → Softmax
    """
    
    def __init__(self, encoding_type, n_pass, conv1_kernel_size=3):
        super().__init__()
        
        self.encoding_type = encoding_type
        self.n_pass = n_pass
        self.conv1_kernel_size = conv1_kernel_size
        self.num_filters_conv1 = NUM_FILTERS_CONV1
        self.num_filters_conv2 = NUM_FILTERS_CONV2
        
        # Layer 1: Fixed edge detection filters (not trained)
        self.conv1_filters = nn.Parameter(
            get_edge_filters(conv1_kernel_size), requires_grad=False
        )
        
        # Encoding layer
        self.encoding = get_encoding(encoding_type, NUM_FILTERS_CONV1, n_pass)
        
        # Custom pooling layer
        self.custom_pool = LowestPassingMaxPool(n_pass)
        
        # Layer 2: 16 learned 3x3 filters
        self.conv2 = nn.Conv2d(
            NUM_FILTERS_CONV1, NUM_FILTERS_CONV2, 
            CONV2_KERNEL_SIZE, padding=0
        )
        
        # Calculate spatial dimensions
        # Input: 28x28
        # After conv1: 28 - kernel_size + 1
        # After pool: (28 - kernel_size + 1) // 2
        # After conv2: ((28 - kernel_size + 1) // 2) - 2
        # After pool2: (((28 - kernel_size + 1) // 2) - 2) // 2
        after_conv1 = 28 - conv1_kernel_size + 1
        after_pool1 = after_conv1 // 2
        after_conv2 = after_pool1 - CONV2_KERNEL_SIZE + 1
        after_pool2 = after_conv2 // 2
        
        self.flat_size = NUM_FILTERS_CONV2 * after_pool2 * after_pool2
        
        # Dense layers
        self.fc1 = nn.Linear(self.flat_size, DENSE_HIDDEN_SIZE)
        self.fc2 = nn.Linear(DENSE_HIDDEN_SIZE, NUM_CLASSES)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        # Conv1: Fixed edge filters
        x = F.conv2d(x, self.conv1_filters, padding=0)
        raw_activations = F.relu(x)
        
        # Rank encoding
        encoded = self.encoding(raw_activations)
        
        # Lowest-passing pooling
        x = self.custom_pool(encoded, raw_activations)
        
        # Conv2: Learned filters
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, POOL_SIZE)
        
        # Flatten and dense
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"EncodedCNN Summary")
        print(f"{'='*60}")
        print(f"Encoding: {self.encoding_type}")
        print(f"N_pass: {self.n_pass}")
        print(f"Conv1: {self.num_filters_conv1} fixed edge filters ({self.conv1_kernel_size}x{self.conv1_kernel_size})")
        print(f"Pool1: LowestPassingMaxPool (2x2)")
        print(f"Conv2: {self.num_filters_conv2} learned filters (3x3)")
        print(f"Pool2: Standard MaxPool (2x2)")
        print(f"Dense: {self.flat_size} → {DENSE_HIDDEN_SIZE} → {NUM_CLASSES}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*60}\n")


class ConventionalCNN(nn.Module):
    """
    Standard CNN with learned conv1 and standard max pooling.
    
    Architecture:
        Conv1 (4 learned filters) → ReLU → MaxPool →
        Conv2 (16 learned 3x3 filters) → ReLU → Pool →
        Flatten → Dense → ReLU → Dense → Softmax
    """
    
    def __init__(self, conv1_kernel_size=3):
        super().__init__()
        
        self.conv1_kernel_size = conv1_kernel_size
        self.num_filters_conv1 = NUM_FILTERS_CONV1
        self.num_filters_conv2 = NUM_FILTERS_CONV2
        
        # Layer 1: Learned convolution
        self.conv1 = nn.Conv2d(1, NUM_FILTERS_CONV1, conv1_kernel_size, padding=0)
        
        # Layer 2: 16 learned 3x3 filters
        self.conv2 = nn.Conv2d(
            NUM_FILTERS_CONV1, NUM_FILTERS_CONV2, 
            CONV2_KERNEL_SIZE, padding=0
        )
        
        # Calculate spatial dimensions (same as EncodedCNN)
        after_conv1 = 28 - conv1_kernel_size + 1
        after_pool1 = after_conv1 // 2
        after_conv2 = after_pool1 - CONV2_KERNEL_SIZE + 1
        after_pool2 = after_conv2 // 2
        
        self.flat_size = NUM_FILTERS_CONV2 * after_pool2 * after_pool2
        
        # Dense layers
        self.fc1 = nn.Linear(self.flat_size, DENSE_HIDDEN_SIZE)
        self.fc2 = nn.Linear(DENSE_HIDDEN_SIZE, NUM_CLASSES)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        # Conv1: Learned filters
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, POOL_SIZE)
        
        # Conv2: Learned filters
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, POOL_SIZE)
        
        # Flatten and dense
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"ConventionalCNN Summary")
        print(f"{'='*60}")
        print(f"Conv1: {self.num_filters_conv1} learned filters ({self.conv1_kernel_size}x{self.conv1_kernel_size})")
        print(f"Pool1: Standard MaxPool (2x2)")
        print(f"Conv2: {self.num_filters_conv2} learned filters (3x3)")
        print(f"Pool2: Standard MaxPool (2x2)")
        print(f"Dense: {self.flat_size} → {DENSE_HIDDEN_SIZE} → {NUM_CLASSES}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*60}\n")
