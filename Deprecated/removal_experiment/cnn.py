"""
CNN Implementation with First-Layer Sparsification
===================================================
A modular CNN where the first layer's output can be sparsified
to pass only the top N activations at each spatial position.

Key design:
- Sparsification happens ONLY between layer 1 and layer 2
- All other layers operate normally
- The sparsified zeros are indistinguishable from natural zeros
- Uses NumPy vectorization for performance
"""

import numpy as np
import os

from config import (
    FILTER_SIZE, POOL_SIZE, DENSE_HIDDEN_SIZE, NUM_CLASSES
)


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def sigmoid(x):
    """Sigmoid activation: squashes values to (0, 1)."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(output):
    """Derivative of sigmoid, given sigmoid output."""
    return output * (1.0 - output)


def softmax(x):
    """Softmax: converts logits to probabilities that sum to 1."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def relu(x):
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)


def relu_derivative(output):
    """Derivative of ReLU."""
    return (output > 0).astype(float)


# =============================================================================
# FILTER GENERATION
# =============================================================================

def generate_filters(num_filters, size=FILTER_SIZE, zero_sum=True):
    """
    Generate random convolutional filters.
    
    Args:
        num_filters: Number of filters to create
        size: Filter dimensions (creates size x size filters)
        zero_sum: If True, normalize to sum to zero (brightness-invariant)
    
    Returns:
        3D numpy array of shape (num_filters, size, size)
    """
    filters = np.random.randn(num_filters, size, size)
    if zero_sum:
        # Subtract mean along spatial dimensions for each filter
        filters = filters - filters.mean(axis=(1, 2), keepdims=True)
    # Normalize each filter
    std = filters.std(axis=(1, 2), keepdims=True) + 1e-8
    filters = filters / std
    return filters


# =============================================================================
# FAST CONVOLUTION (im2col approach)
# =============================================================================

def im2col(image, filter_h, filter_w, stride=1):
    """
    Extract image patches as columns for efficient convolution.
    
    Args:
        image: 2D array (H, W)
        filter_h, filter_w: Filter dimensions
        stride: Convolution stride
    
    Returns:
        2D array where each column is a flattened patch
    """
    H, W = image.shape
    out_h = (H - filter_h) // stride + 1
    out_w = (W - filter_w) // stride + 1
    
    # Create output matrix
    cols = np.zeros((filter_h * filter_w, out_h * out_w))
    
    col_idx = 0
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:i*stride+filter_h, j*stride:j*stride+filter_w]
            cols[:, col_idx] = patch.flatten()
            col_idx += 1
    
    return cols, out_h, out_w


def convolve_fast(image, filters, stride=1):
    """
    Fast convolution using matrix multiplication.
    
    Args:
        image: 2D array (H, W)
        filters: 3D array (num_filters, fh, fw)
        stride: Convolution stride
    
    Returns:
        List of 2D activation maps
    """
    num_filters, fh, fw = filters.shape
    
    # Extract patches
    cols, out_h, out_w = im2col(image, fh, fw, stride)
    
    # Reshape filters to 2D: (num_filters, fh*fw)
    filters_2d = filters.reshape(num_filters, -1)
    
    # Convolve all filters at once: (num_filters, out_h*out_w)
    output = np.dot(filters_2d, cols)
    
    # Reshape to list of 2D maps
    return [output[i].reshape(out_h, out_w) for i in range(num_filters)]


def convolve_multi_channel(input_maps, filters, activation=sigmoid):
    """
    Convolve filters over multiple input channels and sum.
    
    Args:
        input_maps: List of 2D arrays (one per input channel)
        filters: 3D array (num_filters, fh, fw)
        activation: Activation function
    
    Returns:
        List of 2D activation maps
    """
    # For single image input
    if isinstance(input_maps, np.ndarray) and input_maps.ndim == 2:
        output_maps = convolve_fast(input_maps, filters)
    else:
        # Sum over input channels
        output_maps = None
        for input_map in input_maps:
            channel_output = convolve_fast(input_map, filters)
            if output_maps is None:
                output_maps = channel_output
            else:
                for i in range(len(output_maps)):
                    output_maps[i] = output_maps[i] + channel_output[i]
    
    # Apply activation
    if activation is not None:
        output_maps = [activation(m) for m in output_maps]
    
    return output_maps


# =============================================================================
# FIRST-LAYER SPARSIFICATION (THE EXPERIMENTAL MANIPULATION)
# =============================================================================

def sparsify_first_layer_output(activation_maps, n_pass):
    """
    Apply the N_PASS sparsification to first layer activation maps.
    
    At each spatial position (i, j), we keep only the top n_pass
    highest-valued filter responses, setting all others to zero.
    
    This simulates reduced bandwidth from the first layer to subsequent layers.
    
    Args:
        activation_maps: List of 2D arrays (one per filter)
        n_pass: Number of top activations to keep per position.
                If None or >= num_filters, returns maps unchanged.
    
    Returns:
        List of 2D arrays with sparsification applied.
        Zeros from sparsification are indistinguishable from natural zeros.
    """
    num_filters = len(activation_maps)
    
    # If n_pass is None or covers all filters, no sparsification needed
    if n_pass is None or n_pass >= num_filters:
        return activation_maps
    
    # Get spatial dimensions from first map
    height, width = activation_maps[0].shape
    
    # Stack maps into 3D array: (num_filters, height, width)
    stacked = np.stack(activation_maps, axis=0)
    
    # Create output array (all zeros initially)
    sparsified = np.zeros_like(stacked)
    
    # Vectorized: for each position, find top n_pass indices
    # Reshape to (num_filters, height*width) for easier processing
    flat_stacked = stacked.reshape(num_filters, -1)
    
    # Get top n_pass indices for each position
    # argsort returns ascending, so take last n_pass
    top_indices = np.argsort(flat_stacked, axis=0)[-n_pass:, :]
    
    # Create mask
    flat_sparsified = np.zeros_like(flat_stacked)
    for pos in range(flat_stacked.shape[1]):
        for idx in top_indices[:, pos]:
            flat_sparsified[idx, pos] = flat_stacked[idx, pos]
    
    sparsified = flat_sparsified.reshape(num_filters, height, width)
    
    # Convert back to list of 2D arrays
    return [sparsified[k] for k in range(num_filters)]


# =============================================================================
# POOLING
# =============================================================================

def max_pool_fast(activation_map, size=POOL_SIZE):
    """
    Fast max pooling using reshape tricks.
    """
    h, w = activation_map.shape
    pool_h = h // size
    pool_w = w // size
    
    # Crop to exact multiple of pool size
    cropped = activation_map[:pool_h * size, :pool_w * size]
    
    # Reshape and take max
    reshaped = cropped.reshape(pool_h, size, pool_w, size)
    pooled = reshaped.max(axis=(1, 3))
    
    return pooled


def pool_layer(activation_maps, size=POOL_SIZE):
    """Apply max pooling to all activation maps."""
    return [max_pool_fast(am, size) for am in activation_maps]


# =============================================================================
# DENSE LAYERS
# =============================================================================

def init_dense_layer(input_size, output_size):
    """
    Initialize a dense layer with Xavier initialization.
    """
    scale = np.sqrt(2.0 / (input_size + output_size))
    return {
        'weights': np.random.randn(input_size, output_size) * scale,
        'bias': np.zeros(output_size)
    }


def dense_forward(inputs, layer, activation=None):
    """Forward pass through a dense layer."""
    z = np.dot(inputs, layer['weights']) + layer['bias']
    if activation is not None:
        z = activation(z)
    return z


# =============================================================================
# CNN CLASS
# =============================================================================

def compute_output_size(input_size, num_layers, filter_size, pool_size, pool_every=1):
    """
    Compute the spatial output size after all conv+pool layers.
    
    Args:
        input_size: Input image dimension (assumes square)
        num_layers: Number of conv layers
        filter_size: Conv filter size
        pool_size: Pooling window size
        pool_every: Apply pooling every N layers (1 = every layer)
    
    Returns:
        Final spatial size, or negative if architecture is invalid
    """
    size = input_size
    for layer in range(num_layers):
        # After convolution: size - filter_size + 1
        size = size - filter_size + 1
        if size < 1:
            return -1
        # Pool only every pool_every layers
        if (layer + 1) % pool_every == 0:
            size = size // pool_size
            if size < 1:
                return -1
    return size


class SparseCNN:
    """
    CNN with configurable first-layer sparsification.
    
    Architecture:
        [Conv + Pool] x num_layers -> Flatten -> Dense -> Softmax
    
    The n_pass parameter controls how many filter activations
    are passed from layer 1 to layer 2 at each spatial position.
    
    For deeper networks (3+ layers), pooling is applied less frequently
    to preserve spatial information.
    """
    
    def __init__(self, num_layers, filters_per_layer, n_pass=None,
                 filter_size=FILTER_SIZE, pool_size=POOL_SIZE, input_size=28,
                 custom_filters=None, activation='sigmoid'):
        """
        Initialize the CNN.
        
        Args:
            num_layers: Number of convolutional layers
            filters_per_layer: Filters per layer (int for uniform, list for variable)
            n_pass: Number of activations to pass from layer 1 (None = all)
            filter_size: Size of convolutional filters
            pool_size: Size of pooling windows
            input_size: Input image size (assumes square, default 28 for MNIST)
            custom_filters: Optional 3D numpy array of custom filters for layer 1.
                           Shape: (num_filters, filter_size, filter_size).
                           If provided, overrides random filter generation for layer 1.
            activation: Activation function for conv layers ('sigmoid', 'relu', or 'abs').
                       'abs' uses absolute value (detects edges regardless of polarity).
        """
        self.num_layers = num_layers
        self.n_pass = n_pass
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.input_size = input_size
        self.custom_filters = custom_filters
        self.activation_name = activation
        
        # Set activation function
        if activation == 'sigmoid':
            self.conv_activation = sigmoid
        elif activation == 'relu':
            self.conv_activation = relu
        elif activation == 'abs':
            self.conv_activation = lambda x: np.abs(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Determine pooling strategy: pool less frequently for deeper networks
        # to ensure output size stays reasonable (at least 2x2)
        self.pool_every = 1  # default: pool after every layer
        for pool_freq in [1, 2, 3, 4]:
            final_size = compute_output_size(input_size, num_layers, filter_size, 
                                            pool_size, pool_freq)
            if final_size >= 2:
                self.pool_every = pool_freq
                break
        else:
            # If even pooling every 4 layers fails, only pool after last layer
            self.pool_every = num_layers
        
        # Handle uniform vs per-layer filter counts
        if isinstance(filters_per_layer, int):
            self.filters_per_layer = [filters_per_layer] * num_layers
        else:
            self.filters_per_layer = list(filters_per_layer)
        
        # Initialize convolutional layers as 3D arrays (num_filters, h, w)
        self.conv_layers = []
        for layer_idx in range(num_layers):
            num_filters = self.filters_per_layer[layer_idx]
            # Use custom filters for layer 0 if provided
            if layer_idx == 0 and custom_filters is not None:
                if custom_filters.shape[0] != num_filters:
                    raise ValueError(f"custom_filters has {custom_filters.shape[0]} filters, "
                                   f"but filters_per_layer[0] is {num_filters}")
                filters = custom_filters.copy()
            else:
                filters = generate_filters(num_filters, filter_size)
            self.conv_layers.append(filters)
        
        # Dense layers will be initialized on first forward pass
        self.dense_hidden = None
        self.dense_output = None
        self._initialized_dense = False
    
    def _init_dense_layers(self, flattened_size):
        """Initialize dense layers based on flattened conv output size."""
        self.dense_hidden = init_dense_layer(flattened_size, DENSE_HIDDEN_SIZE)
        self.dense_output = init_dense_layer(DENSE_HIDDEN_SIZE, NUM_CLASSES)
        self._initialized_dense = True
    
    def forward(self, image):
        """
        Forward pass through the network.
        
        Args:
            image: 2D numpy array (e.g., 28x28)
        
        Returns:
            output: Probability distribution over classes
            cache: Dict of intermediate values for backprop and visualization
        """
        cache = {
            'input': image,
            'conv_outputs': [],
            'pooled_outputs': [],
            'first_layer_pre_sparsify': None,
            'first_layer_post_sparsify': None
        }
        
        current_input = image
        
        for layer_idx, filters in enumerate(self.conv_layers):
            # Convolve using fast method (with configured activation)
            conv_output = convolve_multi_channel(current_input, filters, 
                                                  activation=self.conv_activation)
            
            # Apply sparsification ONLY after layer 0 (first conv layer)
            if layer_idx == 0 and self.n_pass is not None:
                cache['first_layer_pre_sparsify'] = [m.copy() for m in conv_output]
                conv_output = sparsify_first_layer_output(conv_output, self.n_pass)
                cache['first_layer_post_sparsify'] = conv_output
            
            cache['conv_outputs'].append(conv_output)
            
            # Only pool every pool_every layers (adaptive for deeper networks)
            if (layer_idx + 1) % self.pool_every == 0:
                pooled = pool_layer(conv_output, self.pool_size)
            else:
                pooled = conv_output  # No pooling, just pass through
            cache['pooled_outputs'].append(pooled)
            
            # Output becomes input to next layer
            current_input = pooled
        
        # Flatten pooled outputs from last conv layer
        flattened = np.concatenate([m.flatten() for m in current_input])
        cache['flattened'] = flattened
        
        # Initialize dense layers if needed
        if not self._initialized_dense:
            self._init_dense_layers(len(flattened))
        
        # Dense hidden layer
        hidden = dense_forward(flattened, self.dense_hidden, activation=sigmoid)
        cache['hidden'] = hidden
        
        # Output layer (softmax)
        logits = dense_forward(hidden, self.dense_output, activation=None)
        output = softmax(logits)
        cache['logits'] = logits
        cache['output'] = output
        
        return output, cache
    
    def predict(self, image):
        """Predict class for a single image."""
        output, _ = self.forward(image)
        return np.argmax(output)
    
    def train_step(self, image, label, learning_rate=0.01):
        """
        Perform one training step with backpropagation.
        
        Uses full backprop through dense layers. Conv layers use
        random feature extraction (fixed during training) which 
        works surprisingly well for MNIST.
        
        Args:
            image: 2D input image
            label: Integer class label
            learning_rate: SGD learning rate
        
        Returns:
            loss: Cross-entropy loss for this sample
        """
        # Forward pass
        output, cache = self.forward(image)
        
        # Compute loss (cross-entropy)
        target = np.zeros(NUM_CLASSES)
        target[label] = 1.0
        loss = -np.sum(target * np.log(output + 1e-10))
        
        # =====================================================================
        # BACKPROP THROUGH DENSE LAYERS (full gradient)
        # =====================================================================
        
        # Output layer gradient (softmax + cross-entropy combined)
        d_output = output - target
        
        # Hidden layer gradient
        d_hidden_z = np.dot(d_output, self.dense_output['weights'].T)
        d_hidden = d_hidden_z * sigmoid_derivative(cache['hidden'])
        
        # Update output layer
        self.dense_output['weights'] -= learning_rate * np.outer(cache['hidden'], d_output)
        self.dense_output['bias'] -= learning_rate * d_output
        
        # Update hidden layer  
        self.dense_hidden['weights'] -= learning_rate * np.outer(cache['flattened'], d_hidden)
        self.dense_hidden['bias'] -= learning_rate * d_hidden
        
        # Note: Conv filters are kept fixed (random feature extraction)
        # This is a common approach for quick experiments and works well on MNIST
        # The sparsification experiment focuses on information flow, not filter learning
        
        return loss
    
    def save(self, filepath):
        """Save model to a .npz file."""
        data = {
            'num_layers': self.num_layers,
            'filters_per_layer': self.filters_per_layer,
            'n_pass': self.n_pass if self.n_pass is not None else -1,
            'filter_size': self.filter_size,
            'pool_size': self.pool_size,
        }
        
        # Save conv filters (now 3D arrays)
        for layer_idx, filters in enumerate(self.conv_layers):
            data[f'conv_{layer_idx}'] = filters
        
        # Save dense layers
        if self._initialized_dense:
            data['dense_hidden_weights'] = self.dense_hidden['weights']
            data['dense_hidden_bias'] = self.dense_hidden['bias']
            data['dense_output_weights'] = self.dense_output['weights']
            data['dense_output_bias'] = self.dense_output['bias']
        
        np.savez(filepath, **data)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from a .npz file."""
        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'
        
        data = np.load(filepath)
        
        # Reconstruct model
        num_layers = int(data['num_layers'])
        filters_per_layer = list(data['filters_per_layer'])
        n_pass = int(data['n_pass'])
        if n_pass == -1:
            n_pass = None
        
        model = cls(
            num_layers=num_layers,
            filters_per_layer=filters_per_layer,
            n_pass=n_pass,
            filter_size=int(data['filter_size']),
            pool_size=int(data['pool_size'])
        )
        
        # Load conv filters
        for layer_idx in range(num_layers):
            key = f'conv_{layer_idx}'
            model.conv_layers[layer_idx] = data[key]
        
        # Load dense layers if saved
        if 'dense_hidden_weights' in data:
            model.dense_hidden = {
                'weights': data['dense_hidden_weights'],
                'bias': data['dense_hidden_bias']
            }
            model.dense_output = {
                'weights': data['dense_output_weights'],
                'bias': data['dense_output_bias']
            }
            model._initialized_dense = True
        
        print(f"Model loaded from {filepath}")
        return model
    
    def summary(self):
        """Print model architecture summary."""
        print("\n" + "=" * 50)
        print("SparseCNN Architecture")
        print("=" * 50)
        print(f"Layers: {self.num_layers}")
        print(f"Filters per layer: {self.filters_per_layer}")
        print(f"Filter size: {self.filter_size}x{self.filter_size}")
        print(f"Pool size: {self.pool_size}x{self.pool_size}")
        print(f"N_PASS (first layer): {self.n_pass if self.n_pass else 'ALL'}")
        print(f"Dense hidden size: {DENSE_HIDDEN_SIZE}")
        print(f"Output classes: {NUM_CLASSES}")
        print("=" * 50 + "\n")
