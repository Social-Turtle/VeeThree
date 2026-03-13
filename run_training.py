"""Unified training runner for all VeeThree MNIST models.

Usage:
    uv run python run_training.py --all              # run every spec in test_specs.py
    uv run python run_training.py --lut              # only LUT specs
    uv run python run_training.py --fe_lut           # only FE-LUT specs
    uv run python run_training.py --edge_lut         # only Edge-LUT specs
    uv run python run_training.py --cnn              # only CNN specs
    uv run python run_training.py --lut --edge_lut   # combine flags freely

Output:
    results/benchmark.csv  — appends one row per spec (skips duplicate configs)

CSV columns:
    model                  model name
    config                 JSON of model-specific hyperparameters
    test_acc_pct           test-set accuracy (%)
    seq2s_per_img          ordering-primitive comparisons per image (0 for CNN)
    spikes_per_img         fired comparison bits per image        (0 for CNN)
    active_signals_per_img transmitted non-zero signal-bits per image
    adds_per_img           scalar additions × 32 per image
    multiplies_per_img     scalar multiplications × 32 per image  (0 for LUT models)
    train_time_s           wall-clock training time (seconds)
"""

import os
import sys
import csv
import json
import time
import argparse

import datetime
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FE_DIR     = os.path.join(_SCRIPT_DIR, 'feature_engineering')
_RESULTS    = os.path.join(_SCRIPT_DIR, 'results')

# VeeThree must come first (feature_engineering/main.py must not shadow ours).
sys.path.insert(0, _FE_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from main import load_mnist                              # noqa: E402 (path set above)

# Lazy imports of model modules happen inside each runner so that
# module-level constants can be patched before any class is instantiated.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRAINING_KEYS = {"model", "epochs", "n_train"}


def _model_config(spec: dict) -> dict:
    """Return only the model-specific keys from a spec (strip training keys)."""
    return {k: v for k, v in spec.items() if k not in _TRAINING_KEYS}


def _patch_lut_globals(spec: dict):
    """Monkey-patch lut_model constants from spec before instantiating a model."""
    import lut_model as lm
    for attr in ("N_T", "N_C", "EMBED_DIM", "N_GRID", "N_LOCAL", "N_GLOBAL", "WARMUP", "MIN_LR"):
        if attr in spec:
            setattr(lm, attr, spec[attr])


def _load_lut_data(spec: dict):
    """Return (train_imgs, train_lbls, test_imgs, test_lbls) from parquet files."""
    data_dir = os.path.join(_SCRIPT_DIR, "mnist", "mnist")
    return load_mnist(data_dir=data_dir, max_samples=spec.get("n_train"))


def _train_lut_family(model, images, labels, epochs: int) -> float:
    """Minimal per-sample SGD loop. Returns elapsed seconds."""
    t0 = time.time()
    n  = len(images)
    global_step = 0
    for _ in range(epochs):
        for idx in np.random.permutation(n):
            global_step += 1
            model.step(images[idx], labels[idx], t=global_step)
    return time.time() - t0


def _eval_accuracy_lut(model, images, labels) -> float:
    """Test accuracy as percentage."""
    correct = sum(
        int(np.argmax(model.forward(img)[0])) == lbl
        for img, lbl in zip(images, labels)
    )
    return 100.0 * correct / len(images)


_BIT_WIDTH = 32


def _lut_adds_per_img(model) -> float:
    """Fixed adds cost for one LUT-family forward pass (same for every image).

    Each LUT layer computes y = sum_{i=0..N_T-1} S[i, j[i]], which is
    (N_T - 1) scalar additions per output element.  Scaled by BIT_WIDTH.
    """
    import lut_model as lm
    n_t      = lm.N_T
    n_regions = lm.N_GRID * lm.N_GRID
    adds = 0
    if hasattr(model, "local_layers"):
        for layer in model.local_layers:
            adds += n_regions * layer.y_dim * max(n_t - 1, 0)
    if hasattr(model, "global_layers"):
        for layer in model.global_layers:
            adds += layer.y_dim * max(n_t - 1, 0)
    adds += model.output_lut.y_dim * max(n_t - 1, 0)
    return float(adds * _BIT_WIDTH)


def _measure_costs_lut(model, images) -> dict:
    """Mean per-image seq2s, spikes, active signals, adds, and multiplies."""
    from benchmark.metrics import lut_active_signals

    adds_per_img = _lut_adds_per_img(model)   # constant across images

    total_seq2s  = 0
    total_spikes = 0
    total_active = 0
    for img in images:
        _, caches, sc, seq2 = model.forward(img)
        total_seq2s  += int(seq2["total"])
        total_spikes += int(sc["total"])
        # First-layer inputs: regions list for LUT/Edge-LUT, flat vector for FE-LUT
        if caches.get("local_inputs"):
            first_inputs = caches["local_inputs"][0]
        else:
            first_inputs = caches.get("x")
        total_active += lut_active_signals(sc, caches.get("y_outputs", []), first_inputs)
    n = len(images)
    return {
        "seq2s_per_img":          total_seq2s  / n,
        "active_signals_per_img": total_active / n,
        "adds_per_img":           adds_per_img,
        "multiplies_per_img":     0,
    }


# ---------------------------------------------------------------------------
# Per-model runners
# ---------------------------------------------------------------------------

def run_lut_spec(spec: dict) -> dict:
    _patch_lut_globals(spec)
    import lut_model as lm

    stride = spec.get("INPUT_STRIDE", lm.INPUT_STRIDE)
    model  = lm.LUTModel(stride=stride)

    train_imgs, train_lbls, test_imgs, test_lbls = _load_lut_data(spec)
    train_time = _train_lut_family(model, train_imgs, train_lbls, spec["epochs"])
    accuracy   = _eval_accuracy_lut(model, test_imgs, test_lbls)
    costs      = _measure_costs_lut(model, test_imgs)

    return {"model": "lut", "config": _model_config(spec),
            "test_acc_pct": accuracy, "train_time_s": train_time, **costs}


def run_fe_lut_spec(spec: dict) -> dict:
    _patch_lut_globals(spec)
    import fe_lut_model as fm
    if "N_GLOBAL" in spec:
        fm.N_GLOBAL = spec["N_GLOBAL"]

    model = fm.FELUTModel()

    train_imgs, train_lbls, test_imgs, test_lbls = _load_lut_data(spec)
    train_time = _train_lut_family(model, train_imgs, train_lbls, spec["epochs"])
    accuracy   = _eval_accuracy_lut(model, test_imgs, test_lbls)
    costs      = _measure_costs_lut(model, test_imgs)

    return {"model": "fe_lut", "config": _model_config(spec),
            "test_acc_pct": accuracy, "train_time_s": train_time, **costs}


def run_edge_lut_spec(spec: dict) -> dict:
    _patch_lut_globals(spec)
    import edge_lut_model as em

    stride = spec.get("EDGE_STRIDE", em.EDGE_STRIDE)
    model  = em.EdgeLUTModel(stride=stride)

    train_imgs, train_lbls, test_imgs, test_lbls = _load_lut_data(spec)
    train_time = _train_lut_family(model, train_imgs, train_lbls, spec["epochs"])
    accuracy   = _eval_accuracy_lut(model, test_imgs, test_lbls)
    costs      = _measure_costs_lut(model, test_imgs)

    return {"model": "edge_lut", "config": _model_config(spec),
            "test_acc_pct": accuracy, "train_time_s": train_time, **costs}


def run_cnn_spec(spec: dict) -> dict:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    from conventional_cnn import ConventionalCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model from spec kwargs (strip training keys + 'model')
    cnn_kwargs = {k: v for k, v in spec.items()
                  if k not in _TRAINING_KEYS and k != "model"}
    model = ConventionalCNN(**cnn_kwargs).to(device)

    # Data loading (torchvision, normalised)
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    n_train    = spec.get("n_train")
    train_ds   = torchvision.datasets.MNIST(root='/tmp/mnist', train=True,
                                             download=True, transform=transform)
    test_ds    = torchvision.datasets.MNIST(root='/tmp/mnist', train=False,
                                             download=True, transform=transform)
    if n_train is not None:
        train_ds = Subset(train_ds, list(range(min(n_train, len(train_ds)))))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    # Training
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion  = nn.CrossEntropyLoss()
    t0 = time.time()
    model.train()
    for _ in range(spec["epochs"]):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    train_time = time.time() - t0

    # Accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    accuracy = 100.0 * correct / total

    # Costs — measure all 4 counters per image using cost hooks.
    # Hooks fire on every Conv2d and Linear layer:
    #   active_signals: count_nonzero(input) × 16  +  count_nonzero(output) × 16
    #                   (output of the final tracked layer is excluded)
    #   adds:           (MACs - 1 + bias) × BIT_WIDTH per layer
    #   multiplies:     MACs × BIT_WIDTH per layer
    #   seq2s:          always 0 for CNN
    model.register_cost_hooks()
    totals   = {"active_signals": 0.0, "seq2s": 0.0, "adds": 0.0, "multiplies": 0.0}
    n_images = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            for i in range(x.size(0)):
                model.reset_costs()
                model(x[i:i+1])
                c = model.get_costs()
                for k in totals:
                    totals[k] += c[k]
                n_images += 1
    model.remove_cost_hooks()
    denom = max(n_images, 1)

    costs = {
        "seq2s_per_img":          0,
        "active_signals_per_img": totals["active_signals"] / denom,
        "adds_per_img":           totals["adds"]           / denom,
        "multiplies_per_img":     totals["multiplies"]     / denom,
    }
    return {"model": "cnn", "config": _model_config(spec),
            "test_acc_pct": accuracy, "train_time_s": train_time, **costs}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_RUNNERS = {
    "lut":      run_lut_spec,
    "fe_lut":   run_fe_lut_spec,
    "edge_lut": run_edge_lut_spec,
    "cnn":      run_cnn_spec,
}

# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "model", "config",
    "test_acc_pct",
    "seq2s_per_img", "active_signals_per_img",
    "adds_per_img", "multiplies_per_img",
    "train_time_s", "timestamp",
]


def _append_rows(path: str, rows: list[dict]):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "config":                 json.dumps(row["config"]),
                "test_acc_pct":           round(row["test_acc_pct"],               2),
                "seq2s_per_img":          round(row["seq2s_per_img"],              1),
                "active_signals_per_img": round(row["active_signals_per_img"],     1),
                "adds_per_img":           round(row["adds_per_img"],               1),
                "multiplies_per_img":     round(row["multiplies_per_img"],         1),
                "train_time_s":           round(row["train_time_s"],               2),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train and benchmark VeeThree MNIST models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lut",      action="store_true", help="Run LUT specs")
    parser.add_argument("--fe_lut",   action="store_true", help="Run FE-LUT specs")
    parser.add_argument("--edge_lut", action="store_true", help="Run Edge-LUT specs")
    parser.add_argument("--cnn",      action="store_true", help="Run CNN specs")
    parser.add_argument("--all",      action="store_true", help="Run all specs")
    parser.add_argument("--out", default=os.path.join(_RESULTS, "benchmark.csv"),
                        help="Output CSV path (default: results/benchmark.csv)")
    args = parser.parse_args()

    if not any([args.lut, args.fe_lut, args.edge_lut, args.cnn, args.all]):
        parser.error("Specify at least one model flag or --all.")

    selected = (
        {"lut", "fe_lut", "edge_lut", "cnn"} if args.all
        else {m for m in ("lut", "fe_lut", "edge_lut", "cnn")
              if getattr(args, m)}
    )

    # Load specs
    sys.path.insert(0, _SCRIPT_DIR)
    from test_specs import SPECS

    specs = [s for s in SPECS if s["model"] in selected]
    if not specs:
        print("No matching specs found in test_specs.py for selected models.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    new_rows = []
    for i, spec in enumerate(specs, 1):
        config_str = json.dumps(_model_config(spec))
        print(f"\n[{i}/{len(specs)}] {spec['model'].upper()}  config={config_str}")
        runner = _RUNNERS[spec["model"]]
        result = runner(spec)
        result["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")

        print(
            f"  acc={result['test_acc_pct']:.2f}%  "
            f"seq2s/img={result['seq2s_per_img']:.0f}  "
            f"active_signals/img={result['active_signals_per_img']:.0f}  "
            f"adds/img={result['adds_per_img']:.0f}  "
            f"mults/img={result['multiplies_per_img']:.0f}  "
            f"time={result['train_time_s']:.1f}s"
        )
        new_rows.append(result)

    _append_rows(args.out, new_rows)
    print(f"\n{len(new_rows)} row(s) written to {args.out}")


if __name__ == "__main__":
    main()
