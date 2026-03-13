"""Test specifications for run_training.py.

Each entry in SPECS is a dict with:
  - 'model'     : one of 'lut', 'fe_lut', 'edge_lut', 'cnn'
  - 'epochs'    : number of training epochs
  - 'n_train'   : training samples (None = full 60 000)
  - model-specific hyperparameters (see per-model notes below)

Model-specific hyperparameters that can be set:

  lut       — N_T, N_C, EMBED_DIM, N_GRID, N_LOCAL, N_GLOBAL, WARMUP, INPUT_STRIDE
  fe_lut    — N_GLOBAL, N_T, N_C, EMBED_DIM, WARMUP
  edge_lut  — EDGE_STRIDE, N_T, N_C, EMBED_DIM, N_GRID, N_LOCAL, N_GLOBAL, WARMUP
  cnn       — arch ('linear'|'small'|'lenet'), n_filters, n_filters2, hidden_size

Omitted hyperparameters fall back to the defaults in lut_model.py /
fe_lut_model.py / edge_lut_model.py.

Add more dicts to SPECS to run additional experiments; run_training.py will
execute them in order and append new rows to the output CSV.
"""

# ---------------------------------------------------------------------------
# Shared training defaults — override per-spec as needed
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "epochs":  100,
    "n_train": 10000,
}

# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------
SPECS = [
    # --- LUT baseline (default constants from lut_model.py) ---
    {
        "model": "lut",
        "N_T": 32, "EMBED_DIM": 64, "N_GRID": 2, "INPUT_STRIDE": 2,
        **_DEFAULTS,
    },

    # --- Hybrid 1: FE→LUT ---
    {
        "model": "fe_lut",
        "N_GLOBAL": 2,
        **_DEFAULTS,
    },

    # --- Hybrid 2: Edge-LUT ---
    {
        "model": "edge_lut",
        "EDGE_STRIDE": 1, "N_GRID": 2,
        **_DEFAULTS,
    },

    # --- CNN baseline ---
    {
        "model": "cnn",
        "arch": "small", "n_filters": 2, "n_filters2": 2, "hidden_size": 3, "bits": 2,
        **_DEFAULTS,
    },
]
