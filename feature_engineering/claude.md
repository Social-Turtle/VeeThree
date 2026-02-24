Hello friend! Here's what we're working on:

Ordering-Sensitive Computational Primitives for MNIST

Quick Reference
Core Primitive:

N(A, B, C) = { C if A < B < C; ∞ otherwise }
Full Details: ./sequence_functional_description.txt

Current Task: Implement MNIST Recognition Pipeline (Stages 1-3)
(Full details @ ./project_description.txt) - check out this file with a Haiku subagent. Don't fill your own context unless necessary.

Architecture Summary
Stage 1: 8 directional ordering primitives per interior pixel (26×26)

Output: 255 - |first - last| when strictly increasing, else ∞
Track: direction type + value
Stage 2: Per-pixel winner-take-all (keep min across 8 directions)

Stage 3: 2×2 spatial pooling (keep min, maintain 28×28 geometry)

Visualization Requirements
Color Encoding: Red=vertical, Blue=horizontal, Green=diagonal-1, Yellow=diagonal-2

Format: Each pixel → 2×2 sub-grid (4 direction types), intensity normalized per stage (lower=brighter, ∞=black)

Output: Show 1 example per digit class (0-9) at each stage

Code Organization
feature_engineering/
├── primitives/          # Core N(A,B,C) implementation
├── stages/              # edge_detection, winner_take_all, spatial_pooling
├── visualization/       # Stage diagnostic images
└── experiments/         # mnist_pipeline.py
Implementation Priorities
Modularity: Each stage accepts/returns (values, metadata) tuples
Extensibility: Design for future stages beyond Stage 3
Vectorization: Use NumPy/PyTorch for efficiency
∞ Handling: Use np.inf or masked arrays
Key Design Decisions to Validate
Value assignment formula: 255 - |difference|
Pooling strategy: 2×2 minimum
Metadata preservation through pipeline
Global vs. local normalization in visualization
Communication Preferences
Mathematical rigor expected (CS/EE background)
Prioritize clarity over optimization
Heavy visualization for pipeline understanding
Iterative experimental refinement

Do not venture outside of /feature_engineering/ !