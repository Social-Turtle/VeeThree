# MNIST Recognition via Ordering Primitives

## Core Primitive
N(A, B, C) = { C if A < B < C; ∞ otherwise }
Details: primitives/core.py  |  also has `apply_cassian` (threshold gate on non-∞ count)

## Pipeline Architecture
- **Stage 1** — Directional edge detection: 8 ordering primitives per interior pixel
- **Stage 2** — Per-pixel winner-take-all: keep the min-value direction
- **Stage 3** — Spatial pooling (2×2 min) → 14×14, channels labeled by direction (h/v/d1/d2)
- **Stage 4** — Sweep detection: H_DETECTORS pattern ["h","h"] → h_line, V_DETECTORS ["v","v"] → v_line. No spatial compression (view=1, scan_view=1). Output: 14×14×2 with channels ["h_line","v_line"]. Stages 1–3 use value ordering; stage 4+ uses **presence detection** only.
- **Stage 5** — Removed (was Cassian pooling; Stage 4 output fed directly to classifier)
- **Stage 6** — Classifier: tree of HPrimitive/VPrimitive/Cassian nodes (digit_classifier.py)

## Classifier Design (Stage 6)
- `HPrimitive(*pattern)` — scans each row left→right; returns (14,) fire positions or inf
- `VPrimitive(*pattern)` — scans each col top→bottom; returns (14,) fire positions or inf
- `Cassian(*children, threshold)` — soft AND gate using apply_cassian from core.py
- **Template rule**: keep primitive patterns short (1–2 elements). To detect separated struts (e.g. digit 8 = 3 horizontal bars), interleave v_line between h_lines: `VPrimitive("h_line","v_line","h_line","v_line","h_line")`. Without the v_line separators, a single thick stroke also matches.
- Score = sum of child primitive fires; highest score wins on ties.

## Code Layout
```
feature_engineering/
├── primitives/      # core.py — apply_primitive, apply_cassian, apply_or, apply_and
├── stages/          # one file per stage
├── visualization/   # diagnostic image renderers
└── experiments/     # mnist_pipeline.py  ← main entrypoint
```

## Working Conventions
- Stages pass `(values: ndarray, metadata)` tuples
- ∞ = inactive; use np.inf throughout
- Visualize one example per digit class (0–9) at each stage
- Stay inside /feature_engineering/
- when running code, always use: uv run
- Write very clear code. Prioritize clarity over efficiency.
- Before reading more than 2 files, stop and ask a clarifying question.
- Talk to me early and often. I'd rather you give me a good question then spend time trying to sort it out yourself.
