"""
Visualization Tools
===================
Plotting and saving activation maps for analysis.
"""

import numpy as np
import os
from PIL import Image


def save_activation_maps(activation_maps, output_dir, prefix="activation"):
    """
    Save activation maps as grayscale images.
    
    Args:
        activation_maps: List of 2D numpy arrays
        output_dir: Directory to save images
        prefix: Prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, activation_map in enumerate(activation_maps):
        # Normalize to 0-255
        map_min = np.min(activation_map)
        map_max = np.max(activation_map)
        
        if map_max - map_min > 0:
            normalized = (activation_map - map_min) / (map_max - map_min)
        else:
            normalized = np.zeros_like(activation_map)
        
        img_data = (normalized * 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')
        
        # Scale up for visibility
        img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
        
        filepath = os.path.join(output_dir, f"{prefix}_{idx:03d}.png")
        img.save(filepath)


def save_comparison_maps(pre_sparsify, post_sparsify, output_dir, prefix="compare"):
    """
    Save side-by-side comparison of pre and post sparsification.
    
    Args:
        pre_sparsify: List of 2D arrays before sparsification
        post_sparsify: List of 2D arrays after sparsification
        output_dir: Directory to save images
        prefix: Prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (pre, post) in enumerate(zip(pre_sparsify, post_sparsify)):
        # Normalize both to same scale
        combined_min = min(np.min(pre), np.min(post))
        combined_max = max(np.max(pre), np.max(post))
        
        if combined_max - combined_min > 0:
            pre_norm = (pre - combined_min) / (combined_max - combined_min)
            post_norm = (post - combined_min) / (combined_max - combined_min)
        else:
            pre_norm = np.zeros_like(pre)
            post_norm = np.zeros_like(post)
        
        # Create side-by-side image with separator
        h, w = pre.shape
        separator_width = 2
        combined_width = w * 2 + separator_width
        
        combined = np.zeros((h, combined_width))
        combined[:, :w] = pre_norm
        combined[:, w:w + separator_width] = 0.5  # Gray separator
        combined[:, w + separator_width:] = post_norm
        
        img_data = (combined * 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')
        
        # Scale up for visibility
        img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
        
        filepath = os.path.join(output_dir, f"{prefix}_{idx:03d}.png")
        img.save(filepath)


def plot_training_history(history, output_dir, config_name):
    """
    Generate and save training plots.
    
    Args:
        history: Dict with 'epoch_losses' and 'epoch_accuracies'
        output_dir: Directory to save plots
        config_name: Name for the configuration (for titles)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = list(range(1, len(history['epoch_losses']) + 1))
    
    # We'll create simple text-based plots that can be rendered
    # as images using PIL since we want to avoid matplotlib dependency
    
    # Loss plot data
    _save_plot_data(
        epochs, 
        history['epoch_losses'],
        os.path.join(output_dir, "loss_data.txt"),
        "Training Loss"
    )
    
    # Accuracy plot data
    _save_plot_data(
        epochs,
        history['epoch_accuracies'],
        os.path.join(output_dir, "accuracy_data.txt"),
        "Training Accuracy"
    )
    
    # Create ASCII plots (visual backup)
    _save_ascii_plot(
        epochs,
        history['epoch_losses'],
        os.path.join(output_dir, "loss_plot.txt"),
        f"Training Loss - {config_name}",
        y_label="Loss"
    )
    
    _save_ascii_plot(
        epochs,
        history['epoch_accuracies'],
        os.path.join(output_dir, "accuracy_plot.txt"),
        f"Training Accuracy - {config_name}",
        y_label="Accuracy",
        y_format=lambda x: f"{x*100:.1f}%"
    )
    
    # Try to create image plots
    try:
        _create_image_plot(
            epochs,
            history['epoch_losses'],
            os.path.join(output_dir, "loss_curve.png"),
            f"Training Loss - {config_name}",
            "Loss"
        )
        
        _create_image_plot(
            epochs,
            history['epoch_accuracies'],
            os.path.join(output_dir, "accuracy_curve.png"),
            f"Training Accuracy - {config_name}",
            "Accuracy"
        )
    except Exception as e:
        print(f"  Note: Could not create image plots ({e})")


def plot_per_digit_accuracy(per_digit_accuracy, output_dir, config_name):
    """
    Generate and save per-digit accuracy plot.
    
    Args:
        per_digit_accuracy: Dict mapping digit -> accuracy
        output_dir: Directory to save plot
        config_name: Name for the configuration
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    filepath = os.path.join(output_dir, "per_digit_accuracy.txt")
    with open(filepath, 'w') as f:
        f.write(f"Per-Digit Accuracy - {config_name}\n")
        f.write("=" * 40 + "\n\n")
        for digit in range(10):
            acc = per_digit_accuracy[digit] * 100
            bar = "█" * int(acc / 2)
            f.write(f"Digit {digit}: {acc:6.2f}% |{bar}\n")
        f.write(f"\nMean: {np.mean(list(per_digit_accuracy.values())) * 100:.2f}%\n")
    
    # Create bar chart image
    try:
        _create_bar_chart(
            list(range(10)),
            [per_digit_accuracy[d] for d in range(10)],
            os.path.join(output_dir, "per_digit_accuracy.png"),
            f"Per-Digit Accuracy - {config_name}"
        )
    except Exception as e:
        print(f"  Note: Could not create bar chart ({e})")


def save_digit_examples(cnn, examples_dict, output_dir):
    """
    Save activation maps for example images of each digit.
    
    Args:
        cnn: SparseCNN instance
        examples_dict: Dict mapping digit -> list of (image, label) tuples
        output_dir: Base directory for saving
    """
    for digit, examples in examples_dict.items():
        for ex_idx, (image, label) in enumerate(examples):
            # Forward pass to get activations
            _, cache = cnn.forward(image)
            
            # Create directory for this digit
            digit_dir = os.path.join(output_dir, f"digit_{digit}", f"example_{ex_idx}")
            os.makedirs(digit_dir, exist_ok=True)
            
            # Save input image
            img_data = (image * 255).astype(np.uint8)
            img = Image.fromarray(img_data, mode='L')
            img = img.resize((112, 112), Image.NEAREST)
            img.save(os.path.join(digit_dir, "input.png"))
            
            # Save pre-sparsification maps (if available)
            if cache['first_layer_pre_sparsify'] is not None:
                pre_dir = os.path.join(digit_dir, "layer1_pre_sparsify")
                save_activation_maps(
                    cache['first_layer_pre_sparsify'],
                    pre_dir,
                    prefix="pre"
                )
            
            # Save post-sparsification maps (if available)
            if cache['first_layer_post_sparsify'] is not None:
                post_dir = os.path.join(digit_dir, "layer1_post_sparsify")
                save_activation_maps(
                    cache['first_layer_post_sparsify'],
                    post_dir,
                    prefix="post"
                )
                
                # Save comparison
                compare_dir = os.path.join(digit_dir, "layer1_comparison")
                save_comparison_maps(
                    cache['first_layer_pre_sparsify'],
                    cache['first_layer_post_sparsify'],
                    compare_dir,
                    prefix="compare"
                )
            
            # Save all layer outputs
            for layer_idx, conv_output in enumerate(cache['conv_outputs']):
                layer_dir = os.path.join(digit_dir, f"layer{layer_idx}_output")
                save_activation_maps(conv_output, layer_dir, prefix=f"layer{layer_idx}")


def _save_plot_data(x_values, y_values, filepath, title):
    """Save raw plot data to a text file."""
    with open(filepath, 'w') as f:
        f.write(f"# {title}\n")
        f.write("# epoch, value\n")
        for x, y in zip(x_values, y_values):
            f.write(f"{x}, {y}\n")


def _save_ascii_plot(x_values, y_values, filepath, title, y_label="Value", 
                     y_format=lambda x: f"{x:.4f}"):
    """Create a simple ASCII art plot."""
    width = 60
    height = 15
    
    y_min = min(y_values)
    y_max = max(y_values)
    y_range = y_max - y_min if y_max > y_min else 1
    
    with open(filepath, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * width + "\n\n")
        
        # Create plot area
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        for i, y in enumerate(y_values):
            x_pos = int((i / (len(y_values) - 1)) * (width - 1)) if len(y_values) > 1 else 0
            y_pos = int(((y - y_min) / y_range) * (height - 1))
            y_pos = height - 1 - y_pos  # Invert for top-down
            plot[y_pos][x_pos] = '●'
        
        # Connect with lines
        for row in range(height):
            for col in range(1, width):
                if plot[row][col - 1] == '●' or plot[row][col] == '●':
                    continue
                # Check if between points
                left_point = right_point = None
                for c in range(col - 1, -1, -1):
                    for r in range(height):
                        if plot[r][c] == '●':
                            left_point = (r, c)
                            break
                    if left_point:
                        break
                for c in range(col + 1, width):
                    for r in range(height):
                        if plot[r][c] == '●':
                            right_point = (r, c)
                            break
                    if right_point:
                        break
        
        # Add y-axis labels
        f.write(f"{y_format(y_max):>10} │\n")
        for row in range(height):
            f.write(f"{'':>10} │{''.join(plot[row])}\n")
        f.write(f"{y_format(y_min):>10} │{'─' * width}\n")
        f.write(f"{'':>10}  1{'':^{width-4}}{len(y_values)}\n")
        f.write(f"{'':>10}  {'Epoch':^{width}}\n")
        f.write(f"\n{y_label}\n")


def _create_image_plot(x_values, y_values, filepath, title, y_label):
    """Create a simple line plot as an image."""
    width = 400
    height = 300
    margin = 50
    
    # Create white image
    img_array = np.ones((height, width), dtype=np.uint8) * 255
    
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    
    # Normalize values
    y_min = min(y_values)
    y_max = max(y_values)
    y_range = y_max - y_min if y_max > y_min else 1
    
    # Draw axes
    for x in range(margin, width - margin):
        img_array[height - margin, x] = 0
    for y in range(margin, height - margin):
        img_array[y, margin] = 0
    
    # Plot line
    for i in range(len(y_values) - 1):
        x1 = margin + int((i / (len(y_values) - 1)) * plot_width)
        x2 = margin + int(((i + 1) / (len(y_values) - 1)) * plot_width)
        
        y1 = height - margin - int(((y_values[i] - y_min) / y_range) * plot_height)
        y2 = height - margin - int(((y_values[i + 1] - y_min) / y_range) * plot_height)
        
        # Draw line using Bresenham's algorithm
        _draw_line(img_array, x1, y1, x2, y2)
    
    # Draw points
    for i, y in enumerate(y_values):
        x_pos = margin + int((i / (len(y_values) - 1)) * plot_width) if len(y_values) > 1 else margin
        y_pos = height - margin - int(((y - y_min) / y_range) * plot_height)
        _draw_circle(img_array, x_pos, y_pos, 3)
    
    img = Image.fromarray(img_array, mode='L')
    img.save(filepath)


def _create_bar_chart(categories, values, filepath, title):
    """Create a simple bar chart as an image."""
    width = 400
    height = 300
    margin = 50
    
    img_array = np.ones((height, width), dtype=np.uint8) * 255
    
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    
    bar_width = plot_width // len(categories) - 4
    
    y_max = max(values) if max(values) > 0 else 1
    
    # Draw axes
    for x in range(margin, width - margin):
        img_array[height - margin, x] = 0
    for y in range(margin, height - margin):
        img_array[y, margin] = 0
    
    # Draw bars
    for i, (cat, val) in enumerate(zip(categories, values)):
        bar_height = int((val / y_max) * plot_height)
        x_start = margin + i * (bar_width + 4) + 2
        
        for x in range(x_start, x_start + bar_width):
            for y in range(height - margin - bar_height, height - margin):
                if 0 <= x < width and 0 <= y < height:
                    img_array[y, x] = 100
    
    img = Image.fromarray(img_array, mode='L')
    img.save(filepath)


def _draw_line(img, x1, y1, x2, y2):
    """Draw a line on an image array using Bresenham's algorithm."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        if 0 <= y1 < img.shape[0] and 0 <= x1 < img.shape[1]:
            img[y1, x1] = 0
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def _draw_circle(img, cx, cy, r):
    """Draw a filled circle on an image array."""
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    img[y, x] = 0
