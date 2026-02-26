# MNIST Recognition via Ordering Primitives

## Core Primitive
N(A, B, C) = { C if A < B < C; ∞ otherwise }
Details: primitives/core.py  |  also has `apply_cassian` (threshold gate on non-∞ count)

## Pipeline Architecture
- **Stage 1** — Directional edge detection: 8 ordering primitives per interior pixel
- **Stage 2** — Per-pixel winner-take-all: keep the min-value direction
- **Stage 3** — Spatial pooling (2×2 min)
- **Stage 4** — Sweep detection: sliding pattern-matching along rows/cols
- **Stage 5** — (in progress)
- **Stage 6** — Classification: one ordering primitive per digit class; winner = predicted digit

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
