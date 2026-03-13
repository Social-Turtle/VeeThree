"""
Custom Filter Definitions
=========================
Hand-crafted 3x3 convolutional filters for edge detection.
"""

import numpy as np


def get_edge_filters():
    """
    Returns 8 edge detection filters (3x3):
    
    0: Horizontal edge (light above, dark below) - detects bottom edges
    1: Horizontal edge (dark above, light below) - detects top edges
    2: Vertical edge (light left, dark right) - detects right edges
    3: Vertical edge (dark left, light right) - detects left edges
    4: Diagonal 45° (light top-left, dark bottom-right)
    5: Diagonal 45° (dark top-left, light bottom-right)
    6: Diagonal 135° (light top-right, dark bottom-left)
    7: Diagonal 135° (dark top-right, light bottom-left)
    
    All filters are zero-sum (brightness invariant) and normalized.
    """
    filters = []
    
    # Horizontal edges (top-bottom gradient)
    # Filter 0: Light above, dark below (detects bottom edges of strokes)
    horiz_1 = np.array([
        [ 1,  1,  1],
        [ 0,  0,  0],
        [-1, -1, -1]
    ], dtype=np.float32)
    filters.append(horiz_1)
    
    # Filter 1: Dark above, light below (detects top edges of strokes)
    horiz_2 = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)
    filters.append(horiz_2)
    
    # Vertical edges (left-right gradient)
    # Filter 2: Light left, dark right (detects right edges)
    vert_1 = np.array([
        [ 1,  0, -1],
        [ 1,  0, -1],
        [ 1,  0, -1]
    ], dtype=np.float32)
    filters.append(vert_1)
    
    # Filter 3: Dark left, light right (detects left edges)
    vert_2 = np.array([
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]
    ], dtype=np.float32)
    filters.append(vert_2)
    
    # Diagonal 45° (top-left to bottom-right)
    # Filter 4: Light top-left, dark bottom-right
    diag45_1 = np.array([
        [ 2,  1,  0],
        [ 1,  0, -1],
        [ 0, -1, -2]
    ], dtype=np.float32)
    filters.append(diag45_1)
    
    # Filter 5: Dark top-left, light bottom-right
    diag45_2 = np.array([
        [-2, -1,  0],
        [-1,  0,  1],
        [ 0,  1,  2]
    ], dtype=np.float32)
    filters.append(diag45_2)
    
    # Diagonal 135° (top-right to bottom-left)
    # Filter 6: Light top-right, dark bottom-left
    diag135_1 = np.array([
        [ 0,  1,  2],
        [-1,  0,  1],
        [-2, -1,  0]
    ], dtype=np.float32)
    filters.append(diag135_1)
    
    # Filter 7: Dark top-right, light bottom-left
    diag135_2 = np.array([
        [ 0, -1, -2],
        [ 1,  0, -1],
        [ 2,  1,  0]
    ], dtype=np.float32)
    filters.append(diag135_2)
    
    # Stack into 3D array and normalize each filter
    filters = np.stack(filters, axis=0)
    
    # Normalize each filter to have unit norm
    for i in range(len(filters)):
        norm = np.linalg.norm(filters[i])
        if norm > 0:
            filters[i] = filters[i] / norm
    
    return filters


# Filter names for visualization/logging
FILTER_NAMES = [
    "Horiz↓ (bottom edge)",
    "Horiz↑ (top edge)",
    "Vert→ (right edge)",
    "Vert← (left edge)",
    "Diag↘ (45°)",
    "Diag↗ (45° inv)",
    "Diag↙ (135°)",
    "Diag↖ (135° inv)"
]


if __name__ == "__main__":
    # Quick test
    filters = get_edge_filters()
    print(f"Shape: {filters.shape}")
    print(f"Sum of each filter (should be ~0):")
    for i, name in enumerate(FILTER_NAMES):
        print(f"  {i}: {name}: sum={filters[i].sum():.6f}")
