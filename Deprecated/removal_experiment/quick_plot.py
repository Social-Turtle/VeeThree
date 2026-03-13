import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Parse results file
results = defaultdict(list)

with open('grid_search_results.txt', 'r') as f:
    for line in f:
        match = re.match(r'(\d+), (\d+), (\d+), ([\d.]+)%, success', line.strip())
        if match:
            layers, filters, n_pass, acc = int(match[1]), int(match[2]), int(match[3]), float(match[4])
            results[(layers, filters)].append((n_pass, acc))

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, num_layers in zip(axes, [2, 3]):
    ax.set_title(f'{num_layers}-Layer Models', fontsize=14)
    ax.set_xlabel('N_Pass (filters kept per position)')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, alpha=0.3)
    
    for filters in [6, 7, 8, 9, 10]:
        data = results.get((num_layers, filters), [])
        if data:
            data.sort(key=lambda x: x[0])
            x = [d[0] for d in data]
            y = [d[1] for d in data]
            ax.plot(x, y, 'o-', label=f'{filters} filters', linewidth=2, markersize=8)
    
    ax.legend()
    ax.set_ylim(0, 100)

plt.suptitle('Accuracy vs N_Pass (First Layer Sparsification)', fontsize=16)
plt.tight_layout()
plt.savefig('n_pass_comparison.png', dpi=150)
print('Saved: n_pass_comparison.png')
