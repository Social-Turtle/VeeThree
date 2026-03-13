"""Analyze why n_pass=8 underperforms."""
import numpy as np
from cnn import SparseCNN, sparsify_first_layer_output
from custom_filters import get_edge_filters, FILTER_NAMES
from data_loader import load_mnist

# Load data
train_images, train_labels, _, _ = load_mnist(max_train=100, max_test=10)

# Create model with custom filters
custom_filters = get_edge_filters()
model = SparseCNN(2, 8, n_pass=None, custom_filters=custom_filters)

print("=== Analyzing Custom Filter Activations ===\n")

# Analyze a few images
for img_idx in range(5):
    img = train_images[img_idx]
    label = train_labels[img_idx]
    _, cache = model.forward(img)
    
    conv_out = cache['conv_outputs'][0]
    stacked = np.stack(conv_out, axis=0)  # (8, 26, 26)
    
    print(f"Image {img_idx} (digit {label}):")
    print(f"  Activation stats per filter:")
    for i, name in enumerate(FILTER_NAMES):
        fmap = stacked[i]
        print(f"    {i} {name:20s}: mean={fmap.mean():.3f}, std={fmap.std():.3f}, "
              f"max={fmap.max():.3f}, min={fmap.min():.3f}")
    print()

# Now check how sparsification affects different n_pass values
print("\n=== Sparsification Analysis (averaged over 100 images) ===\n")

all_activations = []
for img in train_images:
    _, cache = model.forward(img)
    conv_out = cache['conv_outputs'][0]
    stacked = np.stack(conv_out, axis=0)
    all_activations.append(stacked)

all_activations = np.stack(all_activations, axis=0)  # (100, 8, 26, 26)
print(f"Activation tensor shape: {all_activations.shape}")
print(f"Overall mean: {all_activations.mean():.4f}")
print(f"Overall std: {all_activations.std():.4f}")

# Key question: Are opposite-direction filters redundant?
print("\n=== Filter Pair Correlations ===")
print("(Opposite-direction filters may be anti-correlated)")
pairs = [(0, 1, "Horiz↓ vs Horiz↑"), 
         (2, 3, "Vert→ vs Vert←"),
         (4, 5, "Diag↘ vs Diag↗"),
         (6, 7, "Diag↙ vs Diag↖")]

for i, j, name in pairs:
    f1 = all_activations[:, i].flatten()
    f2 = all_activations[:, j].flatten()
    corr = np.corrcoef(f1, f2)[0, 1]
    print(f"  {name}: correlation = {corr:.3f}")

# What happens at each position?
print("\n=== Position-wise Analysis ===")
print("At each (i,j) position, which filters tend to 'win'?")

# Reshape to (100*26*26, 8)
flat = all_activations.transpose(0, 2, 3, 1).reshape(-1, 8)
winners = np.argmax(flat, axis=1)
print("\nFilter win frequency (how often each is the top activation):")
for i in range(8):
    pct = (winners == i).mean() * 100
    print(f"  {i} {FILTER_NAMES[i]:20s}: {pct:.1f}%")

# Check if keeping all 8 adds noise
print("\n=== Signal-to-Noise Analysis ===")
print("Mean activation of top-k filters at each position:")
for k in [1, 2, 4, 6, 8]:
    topk_means = []
    for pos_vals in flat:
        sorted_vals = np.sort(pos_vals)[::-1]
        topk_means.append(sorted_vals[:k].mean())
    print(f"  Top {k}: mean={np.mean(topk_means):.4f}")
