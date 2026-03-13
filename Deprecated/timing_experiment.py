"""
Timing Experiment: Analyze which pixel positions fire most intensely per digit class.

Generates 1D histograms showing frequency of "very intense" pixels at each position
in activation maps, broken down by digit class and filter.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from main import load_mnist
from typical_cnn import CNN


def get_top_pixel_positions(activation_map):
    """
    Find pixel positions for top 3 intensity levels in an activation map.
    
    Returns:
        (green_positions, yellow_positions, red_positions, tie_counts)
        where tie_counts = {'green': n, 'yellow': n, 'red': n}
    """
    flat = activation_map.flatten()
    unique_values = np.unique(flat)[::-1]  # Sorted descending
    
    if len(unique_values) < 3:
        # Handle edge case of very uniform maps
        unique_values = np.pad(unique_values, (0, 3 - len(unique_values)), 
                               mode='constant', constant_values=-np.inf)
    
    top1, top2, top3 = unique_values[0], unique_values[1], unique_values[2]
    
    green_positions = np.where(flat == top1)[0]
    yellow_positions = np.where(flat == top2)[0]
    red_positions = np.where(flat == top3)[0]
    
    tie_counts = {
        'green': len(green_positions),
        'yellow': len(yellow_positions),
        'red': len(red_positions)
    }
    
    return green_positions, yellow_positions, red_positions, tie_counts


def run_experiment(model_path='model', num_samples=10000, output_dir='visualizations/timings'):
    """
    Run the timing experiment.
    
    Args:
        model_path: Path to saved model
        num_samples: Number of images to process
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    cnn = CNN.load(model_path)
    num_layers = len(cnn.conv_layers)
    num_filters_per_layer = [len(layer) for layer in cnn.conv_layers]
    
    print(f"Model has {num_layers} conv layer(s) with {num_filters_per_layer} filters")
    
    # Load data
    print(f"\nLoading {num_samples} images...")
    train_images, train_labels, _, _ = load_mnist(max_samples=num_samples)
    
    # Initialize accumulators
    # Structure: [layer][filter][digit] -> {'green': counter, 'yellow': counter, 'red': counter}
    # where counter is a dict mapping pixel_position -> count
    accumulators = []
    tie_totals = []  # Track total ties per layer/filter/digit
    
    for layer_idx in range(num_layers):
        layer_acc = []
        layer_ties = []
        for filter_idx in range(num_filters_per_layer[layer_idx]):
            filter_acc = {}
            filter_ties = {}
            for digit in range(10):
                filter_acc[digit] = {
                    'green': defaultdict(int),
                    'yellow': defaultdict(int),
                    'red': defaultdict(int)
                }
                filter_ties[digit] = {
                    'green': [],
                    'yellow': [],
                    'red': []
                }
            layer_acc.append(filter_acc)
            layer_ties.append(filter_ties)
        accumulators.append(layer_acc)
        tie_totals.append(layer_ties)
    
    # Process images
    print("\nProcessing images...")
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(train_images)} images...")
        
        # Forward pass
        _, cache = cnn.forward(image)
        
        # Analyze each conv layer's output
        for layer_idx, conv_out in enumerate(cache['conv_outputs']):
            for filter_idx in range(conv_out.shape[0]):
                activation_map = conv_out[filter_idx]
                green, yellow, red, ties = get_top_pixel_positions(activation_map)
                
                digit = label
                
                # Accumulate positions
                for pos in green:
                    accumulators[layer_idx][filter_idx][digit]['green'][pos] += 1
                for pos in yellow:
                    accumulators[layer_idx][filter_idx][digit]['yellow'][pos] += 1
                for pos in red:
                    accumulators[layer_idx][filter_idx][digit]['red'][pos] += 1
                
                # Track ties
                tie_totals[layer_idx][filter_idx][digit]['green'].append(ties['green'])
                tie_totals[layer_idx][filter_idx][digit]['yellow'].append(ties['yellow'])
                tie_totals[layer_idx][filter_idx][digit]['red'].append(ties['red'])
    
    # Generate plots
    print("\nGenerating plots...")
    plot_count = 0
    
    for layer_idx in range(num_layers):
        for filter_idx in range(num_filters_per_layer[layer_idx]):
            for digit in range(10):
                data = accumulators[layer_idx][filter_idx][digit]
                ties = tie_totals[layer_idx][filter_idx][digit]
                
                # Determine the size of activation map (get from first non-empty)
                all_positions = set()
                for color in ['green', 'yellow', 'red']:
                    all_positions.update(data[color].keys())
                
                if not all_positions:
                    continue
                    
                num_pixels = max(all_positions) + 1
                
                # Build histogram arrays
                green_hist = np.zeros(num_pixels)
                yellow_hist = np.zeros(num_pixels)
                red_hist = np.zeros(num_pixels)
                
                for pos, count in data['green'].items():
                    green_hist[pos] = count
                for pos, count in data['yellow'].items():
                    yellow_hist[pos] = count
                for pos, count in data['red'].items():
                    red_hist[pos] = count
                
                # Calculate tie statistics
                avg_ties = {
                    'green': np.mean(ties['green']) if ties['green'] else 0,
                    'yellow': np.mean(ties['yellow']) if ties['yellow'] else 0,
                    'red': np.mean(ties['red']) if ties['red'] else 0
                }
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 4))
                
                x = np.arange(num_pixels)
                width = 1.0
                
                ax.bar(x, green_hist, width=width, color='green', alpha=0.6, label='1st (strongest)')
                ax.bar(x, yellow_hist, width=width, color='yellow', alpha=0.6, label='2nd')
                ax.bar(x, red_hist, width=width, color='red', alpha=0.6, label='3rd')
                
                ax.set_xlabel('Pixel Position (flattened)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Layer {layer_idx} | Filter {filter_idx} | Digit {digit}\n'
                             f'Avg ties - Green: {avg_ties["green"]:.1f}, '
                             f'Yellow: {avg_ties["yellow"]:.1f}, '
                             f'Red: {avg_ties["red"]:.1f}')
                ax.legend(loc='upper right')
                
                # Save plot
                filename = f'layer{layer_idx}_filter{filter_idx}_digit{digit}.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                plot_count += 1
    
    print(f"\nExperiment complete! Generated {plot_count} plots in {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Timing experiment for CNN activation analysis')
    parser.add_argument('--model', type=str, default='model',
                        help='Path to saved model (default: model)')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of images to process (default: 10000)')
    parser.add_argument('--output', type=str, default='visualizations/timings',
                        help='Output directory (default: visualizations/timings)')
    
    args = parser.parse_args()
    
    run_experiment(
        model_path=args.model,
        num_samples=args.samples,
        output_dir=args.output
    )
