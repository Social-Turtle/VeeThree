import numpy as np


def spatial_pooling(values: np.ndarray, dir_ids: np.ndarray, n: int = 2, s: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    values:  (28, 28, 8) float64 — stage 2 outputs.
    dir_ids: (28, 28, 8) int8    — direction IDs.
    n: Pool window size (NxN blocks). Default: 2
    s: Stride for pooling. Default: 2
    Returns (H, W, 8) where H = W = (28 - n) // s + 1

    Tile input into non-overlapping NxN blocks with stride s.
    Within each block: find the single (pixel, direction) pair with minimum value
    across all n² pixels × 8 directions candidates.
    The winner's value is placed at the tile's position in the output grid.
    All other entries are np.inf.

    Placing winners at uniform tile positions ensures adjacent tiles are always
    exactly s steps apart in the output, which is required for correct sweep-detection alignment.
    """
    # Calculate output dimensions
    out_h = (values.shape[0] - n) // s + 1
    out_w = (values.shape[1] - n) // s + 1

    # Step 1: Ensure the array is C-contiguous, then reshape to expose NxN tiles.
    # (28, 28, 8) → (out_h, n, out_w, n, 8)
    v = np.ascontiguousarray(values).reshape(out_h, n, out_w, n, 8)

    # Step 2: Transpose to group tile dimensions first, then local spatial dims.
    # (out_h, n, out_w, n, 8) → (out_h, out_w, n, n, 8)
    v_t = v.transpose(0, 2, 1, 3, 4)

    # Step 3: Flatten the n²×8 candidates per tile.
    # (out_h, out_w, n, n, 8) → (out_h, out_w, n²×8)
    v_flat = v_t.reshape(out_h, out_w, n * n * 8)

    # Step 4: Find argmin along the candidate axis for each tile.
    # Result shape: (out_h, out_w)
    winner_flat_idx = np.argmin(v_flat, axis=2)

    # Step 5: Guard — tiles where the minimum is still np.inf have no valid winner.
    tile_rows, tile_cols = np.mgrid[0:out_h, 0:out_w]  # each (out_h, out_w)
    min_vals = v_flat[tile_rows, tile_cols, winner_flat_idx]  # (out_h, out_w)
    active_mask = np.isfinite(min_vals)  # (out_h, out_w)

    # Step 6: Decompose the flat winner index into local spatial + direction offsets.
    # The flat index encodes (local_row, local_col, dir) in row-major order over
    # an (n, n, 8) sub-array.
    #   local_row = flat_idx // (n * 8)       range [0, n-1]
    #   local_col = (flat_idx // 8) % n       range [0, n-1]
    #   dir       = flat_idx % 8              range [0, 7]
    local_row = winner_flat_idx // (n * 8)        # (out_h, out_w)
    local_col = (winner_flat_idx // 8) % n        # (out_h, out_w)
    direction = winner_flat_idx % 8               # (out_h, out_w)

    # Convert tile + local coordinates to global pixel coordinates.
    # tile_rows, tile_cols already broadcast correctly as (out_h, out_w) grids.
    global_r = tile_rows * s + local_row    # (out_h, out_w)
    global_c = tile_cols * s + local_col    # (out_h, out_w)

    # Step 7: Build the (out_h, out_w, 8) output — place each winner at its tile position.
    out_values = np.full((out_h, out_w, 8), np.inf)

    # Restrict scatter to tiles that have at least one finite candidate.
    active_tr = tile_rows[active_mask]  # 1-D: tile row indices of active tiles
    active_tc = tile_cols[active_mask]  # 1-D: tile col indices of active tiles
    gr = global_r[active_mask]          # 1-D: global row of winner pixel (for lookup)
    gc = global_c[active_mask]          # 1-D: global col of winner pixel (for lookup)
    gd = direction[active_mask]         # 1-D: direction channel of winner

    # Retrieve winner values from the original array and write them at
    # the tile position — NOT at the original pixel position.  This guarantees
    # adjacent tiles are exactly s steps apart in the output.
    out_values[active_tr, active_tc, gd] = values[gr, gc, gd]

    # dir_ids for the output grid: channel index is just the channel dimension.
    out_dir_ids = np.tile(np.arange(8, dtype=np.int8), (out_h, out_w, 1))
    return out_values, out_dir_ids
