"""
Encoding Functions for Sequential Removal
==========================================
Convert first-layer activations to rank-based representations.
"""

import torch
import torch.nn as nn
import numpy as np


def get_ranks(activations):
    """
    Get ranks at each spatial position across filters.
    
    Args:
        activations: (batch, num_filters, H, W) tensor
        
    Returns:
        ranks: (batch, num_filters, H, W) tensor where 0 = highest activation
    """
    # argsort twice gives ranks
    sorted_indices = torch.argsort(activations, dim=1, descending=True)
    ranks = torch.argsort(sorted_indices, dim=1)
    return ranks


class BinaryEncoding(nn.Module):
    """
    Binary presence encoding: 1 if filter is in top-k, else 0.
    Output: (batch, num_filters, H, W) with binary values.
    """
    def __init__(self, num_filters, n_pass):
        super().__init__()
        self.num_filters = num_filters
        self.n_pass = n_pass
    
    def forward(self, activations):
        """
        Args:
            activations: (batch, num_filters, H, W)
        Returns:
            encoded: (batch, num_filters, H, W) binary
        """
        ranks = get_ranks(activations)
        # 1 if rank < n_pass (i.e., in top-k), else 0
        encoded = (ranks < self.n_pass).float()
        return encoded


class FixedRankEncoding(nn.Module):
    """
    Fixed linear decay encoding: rank 0 → 1.0, rank 1 → (n-1)/n, etc.
    Output: (batch, num_filters, H, W) with values in [0, 1].
    """
    def __init__(self, num_filters, n_pass):
        super().__init__()
        self.num_filters = num_filters
        self.n_pass = n_pass
    
    def forward(self, activations):
        """
        Args:
            activations: (batch, num_filters, H, W)
        Returns:
            encoded: (batch, num_filters, H, W) with normalized rank scores
        """
        ranks = get_ranks(activations)
        
        # Score: (n_pass - rank) / n_pass for top-k, else 0
        # rank 0 → n_pass/n_pass = 1.0
        # rank 1 → (n_pass-1)/n_pass
        # rank n_pass-1 → 1/n_pass
        scores = (self.n_pass - ranks.float()) / self.n_pass
        
        # Zero out anything not in top-k
        mask = ranks < self.n_pass
        encoded = scores * mask.float()
        
        return encoded


class FixedVectorEncoding(nn.Module):
    """
    Fixed vector encoding: output the rank position directly (1, 2, 3, ...) for top-k.
    Output: (batch, num_filters, H, W) where value = (n_pass - rank) if in top-k.
    
    This gives rank 0 → n_pass, rank 1 → n_pass-1, ..., rank n_pass-1 → 1, else 0.
    """
    def __init__(self, num_filters, n_pass):
        super().__init__()
        self.num_filters = num_filters
        self.n_pass = n_pass
    
    def forward(self, activations):
        """
        Args:
            activations: (batch, num_filters, H, W)
        Returns:
            encoded: (batch, num_filters, H, W) with rank values
        """
        ranks = get_ranks(activations)
        
        # Value = n_pass - rank (so rank 0 → n_pass, rank 1 → n_pass-1, etc.)
        values = (self.n_pass - ranks.float())
        
        # Zero out anything not in top-k
        mask = ranks < self.n_pass
        encoded = values * mask.float()
        
        return encoded


class LearnedRankEncoding(nn.Module):
    """
    Learnable rank weights: each (filter, rank) pair has a learned weight.
    Initialized uniformly to 1.0.
    Output: (batch, num_filters, H, W) with learned weights.
    """
    def __init__(self, num_filters, n_pass):
        super().__init__()
        self.num_filters = num_filters
        self.n_pass = n_pass
        
        # Learnable weights: (num_filters, n_pass)
        # rank_weights[f, r] = weight for filter f when it has rank r
        self.rank_weights = nn.Parameter(torch.ones(num_filters, n_pass))
    
    def forward(self, activations):
        """
        Args:
            activations: (batch, num_filters, H, W)
        Returns:
            encoded: (batch, num_filters, H, W) with learned weights
        """
        batch, num_filters, H, W = activations.shape
        ranks = get_ranks(activations)  # (batch, num_filters, H, W)
        
        # Create output tensor
        encoded = torch.zeros_like(activations)
        
        # For each rank position, apply the corresponding learned weight
        for r in range(self.n_pass):
            # Mask: where rank == r
            mask = (ranks == r)  # (batch, num_filters, H, W)
            
            # Get weights for this rank position for each filter
            weights = self.rank_weights[:, r]  # (num_filters,)
            weights = weights.view(1, num_filters, 1, 1)  # broadcast shape
            
            # Apply weights where mask is true
            encoded = encoded + mask.float() * weights
        
        return encoded


def get_encoding(encoding_type, num_filters, n_pass):
    """Factory function to get encoding module."""
    encodings = {
        'binary': BinaryEncoding,
        'fixed_rank': FixedRankEncoding,
        'fixed_vector': FixedVectorEncoding,
        'learned_rank': LearnedRankEncoding,
    }
    if encoding_type not in encodings:
        raise ValueError(f"Unknown encoding: {encoding_type}. Choose from {list(encodings.keys())}")
    return encodings[encoding_type](num_filters, n_pass)
