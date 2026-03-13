import numpy as np
from cnn import SparseCNN, sparsify_first_layer_output
from data_loader import load_mnist

# Load a few images
train_images, train_labels, _, _ = load_mnist(max_train=100, max_test=10)
X = train_images

# Create model and check activation distributions
model = SparseCNN(2, 6, n_pass=None)

# Forward pass on first image
img = X[0].reshape(28, 28)
_, cache = model.forward(img)

# Get first layer output (before any sparsification)
conv_out = cache['conv_outputs'][0]
stacked = np.stack(conv_out, axis=0)  # (6, 26, 26)

print('=== First Layer Activation Analysis ===')
print(f'Shape: {stacked.shape} (filters, height, width)')
print(f'Value range: [{stacked.min():.4f}, {stacked.max():.4f}]')
print(f'Mean: {stacked.mean():.4f}, Std: {stacked.std():.4f}')
print()

# Check per-position variance across filters
per_position_std = stacked.std(axis=0)
print(f'Per-position std across filters:')
print(f'  Mean: {per_position_std.mean():.4f}')
print(f'  Min:  {per_position_std.min():.4f}')
print(f'  Max:  {per_position_std.max():.4f}')
print()

# What does sparsification actually remove?
print('=== Sparsification Impact ===')
for n_pass in [6, 3, 1]:
    sparsified = sparsify_first_layer_output(conv_out, n_pass)
    sparse_stack = np.stack(sparsified, axis=0)
    zeros_pct = (sparse_stack == 0).sum() / sparse_stack.size * 100
    nonzero_mean = sparse_stack[sparse_stack != 0].mean() if (sparse_stack != 0).any() else 0
    print(f'n_pass={n_pass}: {zeros_pct:.1f}% zeros, remaining mean={nonzero_mean:.4f}')

print()
print('=== Key Insight ===')
# The real question: how similar are filter responses?
correlations = []
for i in range(6):
    for j in range(i+1, 6):
        corr = np.corrcoef(stacked[i].flatten(), stacked[j].flatten())[0,1]
        correlations.append(corr)
print(f'Filter-filter correlations: mean={np.mean(correlations):.3f}, range=[{min(correlations):.3f}, {max(correlations):.3f}]')
