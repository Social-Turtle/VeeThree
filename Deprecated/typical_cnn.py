"""
Conventional CNN Implementation for MNIST
==========================================
A modular, NumPy-based CNN with configurable architecture.

Key design choices:
- Zero-sum (brightness-invariant) filters in conv layers (no bias needed)
- Bias terms in dense layers only
- Sigmoid activation throughout
- Softmax output for classification
- Mean squared error loss
"""

import numpy as np
from PIL import Image
import os


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def sigmoid(x):
    """Sigmoid activation: squashes values to (0, 1)."""
    x = np.clip(x, -500, 500)  # Clip to prevent overflow
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid, given sigmoid output x."""
    return x * (1 - x)


def softmax(x):
    """Softmax: converts logits to probabilities that sum to 1."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


# =============================================================================
# FILTER GENERATION
# =============================================================================

def generate_filters(num_filters, dimension, zero_sum=True):
    """
    Generate random filters for convolution.
    
    Args:
        num_filters: Number of filters to create
        dimension: Filter size (creates dimension x dimension filters)
        zero_sum: If True, normalize to sum to zero (brightness-invariant)
    
    Returns:
        List of numpy arrays, each (dimension x dimension)
    """
    filters = []
    for _ in range(num_filters):
        # Xavier initialization
        scale = np.sqrt(2.0 / (dimension * dimension))
        f = np.random.randn(dimension, dimension) * scale
        
        if zero_sum:
            f = f - f.mean()  # Subtract mean so filter sums to zero
        
        filters.append(f)
    
    return filters


# =============================================================================
# CONVOLUTION
# =============================================================================

def convolve_single(image, filter_matrix, stride=1):
    """
    Convolve a single filter with an image.
    
    Args:
        image: 2D numpy array (height x width)
        filter_matrix: 2D numpy array (filter_h x filter_w)
        stride: Step size for convolution
    
    Returns:
        2D numpy array of convolution outputs
    """
    img_h, img_w = image.shape
    filt_h, filt_w = filter_matrix.shape
    
    out_h = (img_h - filt_h) // stride + 1
    out_w = (img_w - filt_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            row_start = i * stride
            col_start = j * stride
            region = image[row_start:row_start+filt_h, col_start:col_start+filt_w]
            output[i, j] = np.sum(region * filter_matrix)
    
    return output


def convolve_image(image, filters, stride=1, activation=None):
    """
    Apply multiple filters to an image with optional activation.
    
    Args:
        image: 2D or 3D numpy array. If 3D (channels x h x w), sums across channels.
        filters: List of 2D filter arrays
        stride: Step size for convolution
        activation: Activation function (default: sigmoid)
    
    Returns:
        3D numpy array (num_filters x out_h x out_w)
    """
    if activation is None:
        activation = sigmoid
    
    # Handle 3D input by summing convolutions across channels
    if image.ndim == 3:
        activation_maps = []
        for filt in filters:
            conv_sum = sum(convolve_single(channel, filt, stride) for channel in image)
            activation_maps.append(activation(conv_sum))
        return np.array(activation_maps)
    
    # 2D input: simple convolution
    activation_maps = []
    for filt in filters:
        conv_output = convolve_single(image, filt, stride)
        activated = activation(conv_output)
        activation_maps.append(activated)
    
    return np.array(activation_maps)


def save_activation_maps(activation_maps, output_dir, prefix="filter"):
    """
    Save activation maps as numpy arrays and PNG images.
    
    Args:
        activation_maps: 3D array (num_filters x height x width), values 0-1
        output_dir: Directory to save outputs
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, amap in enumerate(activation_maps):
        # Save as numpy array
        np.save(os.path.join(output_dir, f"{prefix}_{idx:03d}.npy"), amap)
        
        # Save as PNG
        img_array = (amap * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(os.path.join(output_dir, f"{prefix}_{idx:03d}.png"))


# =============================================================================
# POOLING
# =============================================================================

def pool_single(activation_map, method=1, size=2):
    """
    Apply pooling to a single activation map.
    
    Args:
        activation_map: 2D numpy array
        method: 1=Max, 2=Average, 3=Stochastic
        size: Pooling window size (size x size)
    
    Returns:
        Pooled 2D numpy array
    """
    h, w = activation_map.shape
    out_h, out_w = h // size, w // size
    
    pooled = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = activation_map[i*size:(i+1)*size, j*size:(j+1)*size]
            
            if method == 1:    # Max pooling
                pooled[i, j] = np.max(region)
            elif method == 2:  # Average pooling
                pooled[i, j] = np.mean(region)
            elif method == 3:  # Stochastic pooling
                probs = region / (np.sum(region) + 1e-8)
                flat_idx = np.random.choice(region.size, p=probs.flatten())
                pooled[i, j] = region.flatten()[flat_idx]
    
    return pooled


def pool_maps(activation_maps, method=1, size=2):
    """
    Apply pooling to all activation maps.
    
    Args:
        activation_maps: 3D array (num_filters x height x width)
        method: 1=Max, 2=Average, 3=Stochastic
        size: Pooling window size
    
    Returns:
        3D array of pooled maps
    """
    return np.array([pool_single(amap, method, size) for amap in activation_maps])


# =============================================================================
# DENSE LAYERS
# =============================================================================

def init_dense_layer(input_size, output_size):
    """
    Initialize weights and biases for a dense layer.
    
    Args:
        input_size: Number of input neurons
        output_size: Number of output neurons
    
    Returns:
        Dictionary with 'weights' and 'biases'
    """
    scale = np.sqrt(2.0 / input_size)  # Xavier initialization
    return {
        'weights': np.random.randn(input_size, output_size) * scale,
        'biases': np.zeros(output_size)
    }


def dense_forward(inputs, layer, activation=None):
    """
    Forward pass through a dense layer.
    
    Args:
        inputs: 1D numpy array
        layer: Dictionary with 'weights' and 'biases'
        activation: Activation function (None for linear)
    
    Returns:
        1D numpy array of outputs
    """
    z = np.dot(inputs, layer['weights']) + layer['biases']
    return activation(z) if activation else z


# =============================================================================
# FULL NETWORK
# =============================================================================

class CNN:
    """
    A configurable CNN for image classification.
    
    Architecture:
        Input → [Conv → Activation → Pool] × N → Flatten → Dense → Output
    """
    
    def __init__(self, input_shape=(28, 28), num_classes=10,
                 conv_filters=[8], filter_size=3, pool_size=2, pool_method=1,
                 dense_size=128, stride=1):
        """
        Initialize the CNN.
        
        Args:
            input_shape: (height, width) of input images
            num_classes: Number of output classes
            conv_filters: List of filter counts per conv layer, e.g. [8] or [8, 16]
            filter_size: Convolution filter size (filter_size x filter_size)
            pool_size: Pooling window size
            pool_method: 1=Max, 2=Average, 3=Stochastic
            dense_size: Neurons in dense hidden layer
            stride: Convolution stride
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.pool_method = pool_method
        self.stride = stride
        
        # Initialize conv filters (zero-sum for brightness invariance)
        self.conv_layers = [
            generate_filters(nf, filter_size, zero_sum=True) 
            for nf in conv_filters
        ]
        
        # Calculate flattened size after conv/pool layers
        h, w = input_shape
        for nf in conv_filters:
            h = (h - filter_size) // stride + 1  # After conv
            w = (w - filter_size) // stride + 1
            h, w = h // pool_size, w // pool_size  # After pool
            num_filters = nf
        
        self.flatten_size = h * w * num_filters
        self.dense_size = dense_size
        
        # Initialize dense layers (with biases)
        self.dense_hidden = init_dense_layer(self.flatten_size, dense_size)
        self.dense_output = init_dense_layer(dense_size, num_classes)
    
    def forward(self, image):
        """
        Forward pass through the network.
        
        Args:
            image: 2D numpy array (height x width), values 0-1
        
        Returns:
            (output_probabilities, cache) where cache has intermediates for backprop
        """
        cache = {'input': image, 'conv_inputs': [], 'conv_outputs': [], 'pool_outputs': []}
        
        current = image
        for filters in self.conv_layers:
            cache['conv_inputs'].append(current)
            conv_out = convolve_image(current, filters, self.stride, sigmoid)
            cache['conv_outputs'].append(conv_out)
            pooled = pool_maps(conv_out, self.pool_method, self.pool_size)
            cache['pool_outputs'].append(pooled)
            current = pooled
        
        # Flatten → Dense hidden → Output
        cache['flattened'] = current.flatten()
        cache['hidden'] = dense_forward(cache['flattened'], self.dense_hidden, sigmoid)
        cache['output'] = dense_forward(cache['hidden'], self.dense_output, sigmoid)
        
        return cache['output'], cache
    
    def compute_loss(self, predicted, target):
        """Mean squared error loss."""
        return np.mean((predicted - target) ** 2)
    
    def backward(self, cache, target, learning_rate=0.01):
        """
        Backward pass: compute gradients and update weights.
        
        Args:
            cache: Dictionary from forward pass
            target: One-hot encoded target
            learning_rate: Learning rate for updates
        """
        output = cache['output']
        
        # Output gradient: MSE derivative * sigmoid derivative
        # d_loss/d_output = 2*(output - target)/n
        # d_output/d_z = output*(1-output)  [sigmoid derivative]
        d_output = 2 * (output - target) / self.num_classes
        d_output *= sigmoid_derivative(output)
        
        # Dense output layer
        d_weights_out = np.outer(cache['hidden'], d_output)
        self.dense_output['weights'] -= learning_rate * d_weights_out
        self.dense_output['biases'] -= learning_rate * d_output
        
        # Backprop to hidden
        d_hidden = np.dot(d_output, self.dense_output['weights'].T)
        d_hidden *= sigmoid_derivative(cache['hidden'])
        
        # Dense hidden layer
        d_weights_hidden = np.outer(cache['flattened'], d_hidden)
        self.dense_hidden['weights'] -= learning_rate * d_weights_hidden
        self.dense_hidden['biases'] -= learning_rate * d_hidden
        
        # Backprop through conv layers
        d_flat = np.dot(d_hidden, self.dense_hidden['weights'].T)
        d_pooled = d_flat.reshape(cache['pool_outputs'][-1].shape)
        
        for layer_idx in range(len(self.conv_layers) - 1, -1, -1):
            conv_out = cache['conv_outputs'][layer_idx]
            conv_in = cache['conv_inputs'][layer_idx]
            pool_out = cache['pool_outputs'][layer_idx]
            
            # Build gradient for this layer's pooled output
            d_layer = np.zeros_like(conv_out)
            
            for f_idx, filt in enumerate(self.conv_layers[layer_idx]):
                # Upsample gradient from pooled to conv size
                d_conv = np.repeat(np.repeat(d_pooled[f_idx], self.pool_size, axis=0),
                                   self.pool_size, axis=1)
                # Pad or crop to match conv_out shape
                target_h, target_w = conv_out[f_idx].shape
                curr_h, curr_w = d_conv.shape
                if curr_h < target_h or curr_w < target_w:
                    d_conv = np.pad(d_conv, ((0, target_h - curr_h), (0, target_w - curr_w)))
                else:
                    d_conv = d_conv[:target_h, :target_w]
                d_conv *= sigmoid_derivative(conv_out[f_idx])
                d_layer[f_idx] = d_conv
                
                # Compute filter gradient (sum over channels if 3D input)
                fh, fw = filt.shape
                d_filter = np.zeros_like(filt)
                if conv_in.ndim == 3:
                    for ch in range(conv_in.shape[0]):
                        for i in range(fh):
                            for j in range(fw):
                                d_filter[i, j] += np.sum(
                                    conv_in[ch, i:i+d_conv.shape[0], j:j+d_conv.shape[1]] * d_conv
                                )
                else:
                    for i in range(fh):
                        for j in range(fw):
                            d_filter[i, j] = np.sum(
                                conv_in[i:i+d_conv.shape[0], j:j+d_conv.shape[1]] * d_conv
                            )
                
                # Update and re-normalize to zero-sum
                filt -= learning_rate * d_filter
                filt -= filt.mean()
            
            # Propagate gradient to previous layer (sum across filters, then pool)
            if layer_idx > 0:
                prev_pool = cache['pool_outputs'][layer_idx - 1]
                d_pooled = np.zeros_like(prev_pool)
                for f_idx in range(len(self.conv_layers[layer_idx])):
                    filt = self.conv_layers[layer_idx][f_idx]
                    # Simplified: average gradient contribution per channel
                    for ch in range(prev_pool.shape[0]):
                        d_pooled[ch] += d_layer[f_idx].mean() * np.ones_like(prev_pool[ch])

    def save(self, filepath):
        """
        Save the model weights and configuration to a file.
        
        Args:
            filepath: Path to save the model (will add .npz extension)
        """
        # Collect all data to save
        save_dict = {
            # Configuration
            'input_shape': np.array(self.input_shape),
            'num_classes': np.array(self.num_classes),
            'filter_size': np.array(self.filter_size),
            'pool_size': np.array(self.pool_size),
            'pool_method': np.array(self.pool_method),
            'stride': np.array(self.stride),
            'flatten_size': np.array(self.flatten_size),
            'dense_size': np.array(self.dense_size),
            'num_conv_layers': np.array(len(self.conv_layers)),
            # Dense layers
            'dense_hidden_weights': self.dense_hidden['weights'],
            'dense_hidden_biases': self.dense_hidden['biases'],
            'dense_output_weights': self.dense_output['weights'],
            'dense_output_biases': self.dense_output['biases'],
        }
        
        # Save conv filters (variable number per layer)
        for layer_idx, filters in enumerate(self.conv_layers):
            save_dict[f'conv_layer_{layer_idx}_num_filters'] = np.array(len(filters))
            for f_idx, filt in enumerate(filters):
                save_dict[f'conv_layer_{layer_idx}_filter_{f_idx}'] = filt
        
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}.npz")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a saved file.
        
        Args:
            filepath: Path to the saved model (.npz file)
        
        Returns:
            CNN instance with loaded weights
        """
        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'
        
        data = np.load(filepath, allow_pickle=True)
        
        # Extract configuration
        input_shape = tuple(data['input_shape'])
        num_classes = int(data['num_classes'])
        filter_size = int(data['filter_size'])
        pool_size = int(data['pool_size'])
        pool_method = int(data['pool_method'])
        stride = int(data['stride'])
        dense_size = int(data['dense_size'])
        num_conv_layers = int(data['num_conv_layers'])
        
        # Reconstruct conv_filters list for initialization
        conv_filters = []
        for layer_idx in range(num_conv_layers):
            num_filters = int(data[f'conv_layer_{layer_idx}_num_filters'])
            conv_filters.append(num_filters)
        
        # Create instance (this will generate random weights)
        cnn = cls(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_filters=conv_filters,
            filter_size=filter_size,
            pool_size=pool_size,
            pool_method=pool_method,
            dense_size=dense_size,
            stride=stride
        )
        
        # Load saved weights
        cnn.dense_hidden['weights'] = data['dense_hidden_weights']
        cnn.dense_hidden['biases'] = data['dense_hidden_biases']
        cnn.dense_output['weights'] = data['dense_output_weights']
        cnn.dense_output['biases'] = data['dense_output_biases']
        
        # Load conv filters
        for layer_idx in range(num_conv_layers):
            num_filters = int(data[f'conv_layer_{layer_idx}_num_filters'])
            for f_idx in range(num_filters):
                cnn.conv_layers[layer_idx][f_idx] = data[f'conv_layer_{layer_idx}_filter_{f_idx}']
        
        print(f"Model loaded from {filepath}")
        return cnn


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def one_hot(label, num_classes=10):
    """Convert integer label to one-hot encoding."""
    vec = np.zeros(num_classes)
    vec[label] = 1
    return vec


def get_outputs(cnn, image):
    """Get probability outputs for an image."""
    output, _ = cnn.forward(image)
    return output


def improve_filters(cnn, image, label, learning_rate=0.01):
    """
    Train the network on a single example (one forward + backward pass).
    
    Returns: loss value
    """
    output, cache = cnn.forward(image)
    target = one_hot(label, cnn.num_classes)
    loss = cnn.compute_loss(output, target)
    cnn.backward(cache, target, learning_rate)
    return loss


def get_best_filters(cnn, images, labels, top_n=5):
    """
    Analyze filter importance via ablation: measure accuracy drop when each filter is removed.
    
    Args:
        cnn: CNN instance
        images: List of test images
        labels: List of corresponding labels
        top_n: Number of top filters to report
    
    Returns:
        Dictionary with filter importance analysis
    """
    # First, get baseline accuracy with all filters
    baseline_correct = 0
    for image, label in zip(images, labels):
        output, _ = cnn.forward(image)
        if np.argmax(output) == label:
            baseline_correct += 1
    baseline_acc = baseline_correct / len(images) * 100
    
    print(f"\n=== Filter Importance (Ablation) ===")
    print(f"Baseline accuracy: {baseline_acc:.1f}% ({baseline_correct}/{len(images)})\n")
    
    # Test each filter by zeroing it out
    num_filters = len(cnn.conv_layers[0])
    accuracy_drops = []
    
    for f_idx in range(num_filters):
        # Save original filter
        original_filter = cnn.conv_layers[0][f_idx].copy()
        
        # Zero out this filter
        cnn.conv_layers[0][f_idx] = np.zeros_like(original_filter)
        
        # Evaluate
        correct = 0
        for image, label in zip(images, labels):
            output, _ = cnn.forward(image)
            if np.argmax(output) == label:
                correct += 1
        
        ablated_acc = correct / len(images) * 100
        drop = baseline_acc - ablated_acc
        accuracy_drops.append((f_idx, drop, ablated_acc))
        
        # Restore filter
        cnn.conv_layers[0][f_idx] = original_filter
    
    # Sort by accuracy drop (most important first)
    accuracy_drops.sort(key=lambda x: x[1], reverse=True)
    
    # Display results
    print(f"Filter importance (by accuracy drop when removed):")
    print(f"{'Filter':<10} {'Drop':<12} {'Acc w/o':<12} {'Status'}")
    print("-" * 46)
    
    for f_idx, drop, acc in accuracy_drops:
        if drop > 5:
            status = "CRITICAL"
        elif drop > 1:
            status = "Important"
        elif drop > 0:
            status = "Useful"
        else:
            status = "Prunable"
        print(f"Filter {f_idx:<3} {drop:>+6.1f}%      {acc:>5.1f}%       {status}")
    
    results = {
        'baseline_accuracy': baseline_acc,
        'filter_drops': [(idx, drop) for idx, drop, _ in accuracy_drops],
        'top_filters': [(idx, drop) for idx, drop, _ in accuracy_drops[:top_n]],
        'prunable': [idx for idx, drop, _ in accuracy_drops if drop <= 0]
    }
    
    print(f"\nSummary:")
    print(f"  Most critical: Filter {accuracy_drops[0][0]} ({accuracy_drops[0][1]:+.1f}% drop)")
    print(f"  Prunable filters (no accuracy loss): {results['prunable'] or 'None'}")
    
    return results
