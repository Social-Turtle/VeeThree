"""
PyTorch CNN Model with Rank-Based Encoding
===========================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rank_encodings import get_encoding
from config import (
    NUM_FILTERS, FILTER_SIZE, POOL_SIZE, 
    DENSE_HIDDEN_SIZE, NUM_CLASSES
)


def get_edge_filters(num_filters=8):
    """
    Returns edge detection filters (3x3).
    
    Args:
        num_filters: 4 for single-direction, 8 for bidirectional pairs
    """
    filters = []
    
    # Horizontal edge (one direction)
    horiz = np.array([[ 1,  1,  1], [ 0,  0,  0], [-1, -1, -1]], dtype=np.float32)
    filters.append(horiz)
    
    # Vertical edge (one direction)
    vert = np.array([[ 1,  0, -1], [ 1,  0, -1], [ 1,  0, -1]], dtype=np.float32)
    filters.append(vert)
    
    # Diagonal 45° (one direction)
    diag45 = np.array([[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]], dtype=np.float32)
    filters.append(diag45)
    
    # Diagonal 135° (one direction)
    diag135 = np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]], dtype=np.float32)
    filters.append(diag135)
    
    if num_filters == 8:
        # Add inverse directions
        horiz_inv = np.array([[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]], dtype=np.float32)
        vert_inv = np.array([[-1,  0,  1], [-1,  0,  1], [-1,  0,  1]], dtype=np.float32)
        diag45_inv = np.array([[-2, -1,  0], [-1,  0,  1], [ 0,  1,  2]], dtype=np.float32)
        diag135_inv = np.array([[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]], dtype=np.float32)
        filters.extend([horiz_inv, vert_inv, diag45_inv, diag135_inv])
    
    # Stack and normalize
    filters = np.stack(filters, axis=0)
    for i in range(len(filters)):
        norm = np.linalg.norm(filters[i])
        if norm > 0:
            filters[i] = filters[i] / norm
    
    return torch.from_numpy(filters).unsqueeze(1)  # (N, 1, 3, 3) for conv2d


FILTER_NAMES_4 = ["Horiz", "Vert", "Diag45", "Diag135"]
FILTER_NAMES_8 = [
    "Horiz↓", "Horiz↑", "Vert→", "Vert←",
    "Diag↘", "Diag↗", "Diag↙", "Diag↖"
]


class SequentialCNN(nn.Module):
    """
    CNN with rank-based encoding between first and second conv layers.
    
    Architecture:
        Conv1 (fixed edge filters) → ReLU → Encoding → Pool →
        Conv2 (learned) → ReLU → Pool →
        Flatten → Dense → ReLU → Dense → Softmax
    """
    
    def __init__(self, encoding_type, n_pass, num_filters=NUM_FILTERS):
        super().__init__()
        
        self.encoding_type = encoding_type
        self.n_pass = n_pass
        self.num_filters = num_filters
        
        # Layer 1: Fixed edge detection filters (not trained)
        self.conv1_filters = nn.Parameter(get_edge_filters(num_filters), requires_grad=False)
        
        # Encoding layer (applied after conv1)
        self.encoding = get_encoding(encoding_type, num_filters, n_pass)
        
        # Layer 2: Learned convolution
        self.conv2 = nn.Conv2d(num_filters, num_filters, FILTER_SIZE, padding=0)
        
        # Calculate flattened size after two conv+pool operations
        # Input: 28x28
        # After conv1: 26x26, after pool: 13x13
        # After conv2: 11x11, after pool: 5x5
        self.flat_size = num_filters * 5 * 5
        
        # Dense layers
        self.fc1 = nn.Linear(self.flat_size, DENSE_HIDDEN_SIZE)
        self.fc2 = nn.Linear(DENSE_HIDDEN_SIZE, NUM_CLASSES)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, 1, 28, 28) input images
        
        Returns:
            logits: (batch, 10) class logits
        """
        # Conv1: Apply fixed edge filters
        # x: (batch, 1, 28, 28) → (batch, 8, 26, 26)
        x = F.conv2d(x, self.conv1_filters, padding=0)
        x = F.relu(x)
        
        # Apply rank-based encoding
        x = self.encoding(x)
        
        # Pool
        x = F.max_pool2d(x, POOL_SIZE)  # → (batch, 8, 13, 13)
        
        # Conv2: Learned filters
        x = self.conv2(x)  # → (batch, 8, 11, 11)
        x = F.relu(x)
        x = F.max_pool2d(x, POOL_SIZE)  # → (batch, 8, 5, 5)
        
        # Flatten
        x = x.view(x.size(0), -1)  # → (batch, 200)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """Get predicted class."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def summary(self):
        """Print model summary."""
        print(f"\n{'='*60}")
        print(f"SequentialCNN Summary")
        print(f"{'='*60}")
        print(f"Encoding: {self.encoding_type}")
        print(f"N_pass: {self.n_pass}")
        print(f"Num filters: {self.num_filters}")
        print(f"Conv1: {self.num_filters} fixed edge filters (3x3)")
        print(f"Conv2: {self.num_filters} learned filters (3x3)")
        print(f"Dense: {self.flat_size} → {DENSE_HIDDEN_SIZE} → {NUM_CLASSES}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if self.encoding_type == 'learned_rank':
            print(f"Learned rank weights shape: {self.encoding.rank_weights.shape}")
        print(f"{'='*60}\n")
