"""
Encoding Functions for Performance Comparison
==============================================
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
        ranks = get_ranks(activations)
        scores = (self.n_pass - ranks.float()) / self.n_pass
        mask = ranks < self.n_pass
        encoded = scores * mask.float()
        return encoded


class FixedVectorEncoding(nn.Module):
    """
    Fixed vector encoding: rank 0 → n_pass, rank 1 → n_pass-1, ..., else 0.
    """
    def __init__(self, num_filters, n_pass):
        super().__init__()
        self.num_filters = num_filters
        self.n_pass = n_pass
    
    def forward(self, activations):
        ranks = get_ranks(activations)
        values = (self.n_pass - ranks.float())
        mask = ranks < self.n_pass
        encoded = values * mask.float()
        return encoded


class LearnedRankEncoding(nn.Module):
    """
    Learnable rank weights: each (filter, rank) pair has a learned weight.
    """
    def __init__(self, num_filters, n_pass):
        super().__init__()
        self.num_filters = num_filters
        self.n_pass = n_pass
        self.rank_weights = nn.Parameter(torch.ones(num_filters, n_pass))
    
    def forward(self, activations):
        batch, num_filters, H, W = activations.shape
        ranks = get_ranks(activations)
        encoded = torch.zeros_like(activations)
        
        for r in range(self.n_pass):
            mask = (ranks == r)
            weights = self.rank_weights[:, r].view(1, num_filters, 1, 1)
            encoded = encoded + mask.float() * weights
        
        return encoded


def get_encoding(encoding_type, num_filters, n_pass):
    """Factory function to get encoding module."""
    encodings = {
        'fixed_rank': FixedRankEncoding,
        'fixed_vector': FixedVectorEncoding,
        'learned_rank': LearnedRankEncoding,
    }
    if encoding_type not in encodings:
        raise ValueError(f"Unknown encoding: {encoding_type}. Choose from {list(encodings.keys())}")
    return encodings[encoding_type](num_filters, n_pass)
