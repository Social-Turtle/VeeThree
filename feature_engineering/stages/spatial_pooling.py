import numpy as np


def spatial_pooling(values: np.ndarray, dir_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    values:  (28, 28, 8) float64 — stage 2 outputs.
    dir_ids: (28, 28, 8) int8    — direction IDs.
    Returns same shapes.

    Tile 28×28 into 14×14 non-overlapping 2×2 blocks.
    Within each block: find the single (pixel, direction) pair with minimum value
    across all 4 pixels × 8 directions = 32 candidates.
    Only that winner keeps its value; the other 31 entries become np.inf.
    dir_ids passes through unchanged.
    """
    # Step 1: Ensure the array is C-contiguous, then reshape to expose 2×2 tiles.
    # (28, 28, 8) → (14, 2, 14, 2, 8)
    v = np.ascontiguousarray(values).reshape(14, 2, 14, 2, 8)

    # Step 2: Transpose to group tile dimensions first, then local spatial dims.
    # (14, 2, 14, 2, 8) → (14, 14, 2, 2, 8)
    v_t = v.transpose(0, 2, 1, 3, 4)

    # Step 3: Flatten the 32 candidates per tile.
    # (14, 14, 2, 2, 8) → (14, 14, 32)
    v_flat = v_t.reshape(14, 14, 32)

    # Step 4: Find argmin along the 32-candidate axis for each of the 14×14 tiles.
    # Result shape: (14, 14)
    winner_flat_idx = np.argmin(v_flat, axis=2)

    # Step 5: Guard — tiles where the minimum is still np.inf have no valid winner.
    tile_rows, tile_cols = np.mgrid[0:14, 0:14]  # each (14, 14)
    min_vals = v_flat[tile_rows, tile_cols, winner_flat_idx]  # (14, 14)
    active_mask = np.isfinite(min_vals)  # (14, 14)

    # Step 6: Decompose the flat winner index into local spatial + direction offsets.
    # The flat index encodes (local_row, local_col, dir) in row-major order over
    # a (2, 2, 8) sub-array.
    #   local_row = flat_idx // (2 * 8)          range [0, 1]
    #   local_col = (flat_idx // 8) % 2           range [0, 1]
    #   dir       = flat_idx % 8                  range [0, 7]
    local_row = winner_flat_idx // 16        # (14, 14)
    local_col = (winner_flat_idx // 8) % 2  # (14, 14)
    direction = winner_flat_idx % 8         # (14, 14)

    # Convert tile + local coordinates to global (28×28) pixel coordinates.
    # tile_rows, tile_cols already broadcast correctly as (14, 14) grids.
    global_r = tile_rows * 2 + local_row    # (14, 14)
    global_c = tile_cols * 2 + local_col    # (14, 14)

    # Step 7: Build the output — start all-inf, then scatter winner values back.
    out_values = np.full_like(values, np.inf)

    # Restrict scatter to tiles that have at least one finite candidate.
    active_tr = tile_rows[active_mask]  # 1-D: tile row indices of active tiles
    active_tc = tile_cols[active_mask]  # 1-D: tile col indices of active tiles
    gr = global_r[active_mask]          # 1-D: global row of winner pixel
    gc = global_c[active_mask]          # 1-D: global col of winner pixel
    gd = direction[active_mask]         # 1-D: direction channel of winner

    # Retrieve the winning values from the original array (not the reshaped view,
    # to avoid any ambiguity about writability) and write them into the output.
    out_values[gr, gc, gd] = values[gr, gc, gd]

    return out_values, dir_ids.copy()
