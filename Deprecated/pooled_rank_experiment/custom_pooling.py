"""
Custom Pooling Layer: Lowest-Passing Max Pooling
=================================================

For each 2x2 region, selects the pixel whose n_pass-th ranked filter
has the highest raw activation value. Only that pixel's encoded values
are passed forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowestPassingMaxPool(nn.Module):
    """
    Custom 2x2 max pooling based on "lowest-passing" signal strength.
    
    For each 2x2 region:
    1. Look at each pixel's n_pass-th ranked filter's raw activation
    2. The pixel with the highest such value "wins"
    3. Pass the winner's encoded rank values forward
    
    Args:
        n_pass: Number of filters being passed (determines which rank to compare)
    """
    
    def __init__(self, n_pass):
        super().__init__()
        self.n_pass = n_pass
        self.pool_size = 2
    
    def forward(self, encoded, raw_activations):
        """
        Args:
            encoded: (B, C, H, W) - Rank-encoded values (sparse, only n_pass non-zero per pixel)
            raw_activations: (B, C, H, W) - Original ReLU activations from conv1
        
        Returns:
            pooled: (B, C, H//2, W//2) - Pooled encoded values
        """
        B, C, H, W = encoded.shape
        
        # Pad if needed to make dimensions even
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            encoded = F.pad(encoded, (0, pad_w, 0, pad_h), mode='constant', value=0)
            raw_activations = F.pad(raw_activations, (0, pad_w, 0, pad_h), mode='constant', value=0)
            H += pad_h
            W += pad_w
        
        # Get the n_pass-th highest activation at each pixel
        # Sort descending along channel dimension
        sorted_acts, _ = torch.sort(raw_activations, dim=1, descending=True)
        
        # The "lowest-passing" is the n_pass-th value (index n_pass-1)
        # Shape: (B, H, W)
        lowest_passing = sorted_acts[:, self.n_pass - 1, :, :]
        
        # Reshape into 2x2 blocks
        # (B, H, W) → (B, H//2, 2, W//2, 2) → (B, H//2, W//2, 4)
        H_out, W_out = H // 2, W // 2
        lp_blocks = lowest_passing.view(B, H_out, 2, W_out, 2)
        lp_blocks = lp_blocks.permute(0, 1, 3, 2, 4).contiguous()  # (B, H_out, W_out, 2, 2)
        lp_blocks = lp_blocks.view(B, H_out, W_out, 4)  # (B, H_out, W_out, 4)
        
        # Find which of the 4 pixels in each block has the max lowest-passing value
        # Shape: (B, H_out, W_out)
        winner_idx = torch.argmax(lp_blocks, dim=3)
        
        # Now we need to gather the encoded values from the winner pixels
        # Reshape encoded: (B, C, H, W) → (B, C, H_out, 2, W_out, 2)
        encoded_blocks = encoded.view(B, C, H_out, 2, W_out, 2)
        encoded_blocks = encoded_blocks.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H_out, W_out, C, 2, 2)
        encoded_blocks = encoded_blocks.view(B, H_out, W_out, C, 4)  # (B, H_out, W_out, C, 4)
        
        # Expand winner_idx to gather from all channels
        # winner_idx: (B, H_out, W_out) → (B, H_out, W_out, C, 1)
        winner_idx_expanded = winner_idx.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, C, 1)
        
        # Gather the winner's encoded values
        # (B, H_out, W_out, C, 4) gather at dim=4 → (B, H_out, W_out, C, 1)
        pooled = torch.gather(encoded_blocks, dim=4, index=winner_idx_expanded)
        pooled = pooled.squeeze(4)  # (B, H_out, W_out, C)
        
        # Permute back to (B, C, H_out, W_out)
        pooled = pooled.permute(0, 3, 1, 2).contiguous()
        
        return pooled


def test_pooling():
    """Quick test of the custom pooling layer."""
    import numpy as np
    
    # Create simple test case
    B, C, H, W = 1, 4, 4, 4
    n_pass = 2
    
    # Raw activations
    torch.manual_seed(42)
    raw = torch.rand(B, C, H, W)
    
    # Simulate encoded values (sparse - only top n_pass per pixel are non-zero)
    encoded = torch.zeros_like(raw)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                top_indices = torch.topk(raw[b, :, h, w], n_pass).indices
                for i, idx in enumerate(top_indices):
                    encoded[b, idx, h, w] = n_pass - i  # Rank values
    
    print("Raw activations at (0,0):", raw[0, :, 0, 0].numpy())
    print("Encoded at (0,0):", encoded[0, :, 0, 0].numpy())
    
    # Apply pooling
    pool = LowestPassingMaxPool(n_pass)
    pooled = pool(encoded, raw)
    
    print(f"\nInput shape: {encoded.shape}")
    print(f"Output shape: {pooled.shape}")
    print(f"Expected: ({B}, {C}, {H//2}, {W//2})")
    
    # Verify one block manually
    # Look at top-left 2x2 block
    print("\n--- Verifying top-left 2x2 block ---")
    for i, (y, x) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        sorted_vals = torch.sort(raw[0, :, y, x], descending=True).values
        lowest_passing = sorted_vals[n_pass - 1].item()
        print(f"Pixel ({y},{x}): lowest-passing = {lowest_passing:.4f}")
    
    print(f"\nPooled output at (0,0): {pooled[0, :, 0, 0].numpy()}")


if __name__ == "__main__":
    test_pooling()
