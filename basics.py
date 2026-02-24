import numpy as np
from PIL import Image
import os


def convolve_filter(image, filter_matrix):
    """
    Convolve a filter with an image at all valid positions.
    Returns normalized array of convolution results.
    
    Args:
        image: 2D numpy array (e.g., 28x28 for MNIST)
        filter_matrix: 2D numpy array (e.g., 3x3)
    
    Returns:
        2D numpy array with normalized values (0-1)
        For 28x28 image with 3x3 filter: 26x26 output
    """
    img_h, img_w = image.shape
    filter_h, filter_w = filter_matrix.shape
    
    # Calculate output dimensions based on valid convolution
    out_h = img_h - filter_h + 1
    out_w = img_w - filter_w + 1
    
    result = np.zeros((out_h, out_w))
    
    # Convolve at each position
    for i in range(out_h):
        for j in range(out_w):
            # Extract region
            region = image[i:i+filter_h, j:j+filter_w]
            # Element-wise multiply and sum
            result[i, j] = np.sum(region * filter_matrix)
    
    # Normalize to 0-1 range
    min_val = np.min(result)
    max_val = np.max(result)
    if max_val - min_val > 0:
        result = (result - min_val) / (max_val - min_val)
    
    return result


def apply_filters(image, filters):
    """
    Apply multiple filters to an image.
    
    Args:
        image: 2D numpy array
        filters: List of filter matrices (e.g., 3x3 numpy arrays)
    
    Returns:
        List of activation maps (26x26 for 28x28 input with 3x3 filters)
    """
    activation_maps = []
    for filter_matrix in filters:
        activation_map = convolve_filter(image, filter_matrix)
        activation_maps.append(activation_map)
    return activation_maps


def select_top_activations(activation_maps, top_x):
    """
    Select pixels with top X highest values across all filters.
    
    Args:
        activation_maps: List of 2D numpy arrays
        top_x: Number of top activations to keep per filter
    
    Returns:
        List of 2D arrays with only top X values kept, rest set to 0
    """
    result_maps = []
    
    for activation_map in activation_maps:
        # Create copy to preserve original
        filtered_map = np.zeros_like(activation_map)
        
        # Flatten and get indices of top X values
        flat = activation_map.flatten()
        top_indices = np.argpartition(flat, -top_x)[-top_x:]
        
        # Convert back to 2D indices and set values
        for idx in top_indices:
            i = idx // activation_map.shape[1]
            j = idx % activation_map.shape[1]
            filtered_map[i, j] = activation_map[i, j]
        
        result_maps.append(filtered_map)
    
    return result_maps


def save_activation_maps(activation_maps, output_dir, prefix="layer"):
    """
    Save activation maps as .jpg images.
    
    Args:
        activation_maps: List of 2D numpy arrays
        output_dir: Directory to save images
        prefix: Prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, activation_map in enumerate(activation_maps):
        # Convert to 0-255 range
        img_array = (activation_map * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(img_array, mode='L')
        
        # Save
        filename = f"{prefix}_filter_{idx:03d}.jpg"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)


def max_pool_2x2(activation_map):
    """
    Apply 2x2 max pooling to an activation map.
    
    Args:
        activation_map: 2D numpy array
    
    Returns:
        Pooled array with dimensions halved
    """
    h, w = activation_map.shape
    pool_h = h // 2
    pool_w = w // 2
    
    pooled = np.zeros((pool_h, pool_w))
    
    for i in range(pool_h):
        for j in range(pool_w):
            # Get 2x2 region
            region = activation_map[i*2:i*2+2, j*2:j*2+2]
            # Take maximum
            pooled[i, j] = np.max(region)
    
    return pooled


def max_pool_all(activation_maps):
    """
    Apply max pooling to all activation maps.
    
    Args:
        activation_maps: List of 2D numpy arrays
    
    Returns:
        List of pooled arrays
    """
    return [max_pool_2x2(am) for am in activation_maps]


def process_layer(image, filters, top_x, output_dir, layer_num):
    """
    Process one complete layer: convolve, select top, save, and pool.
    
    Args:
        image: Input image (2D numpy array)
        filters: List of 9x9 filter matrices
        top_x: Number of top activations to keep
        output_dir: Base directory for saving maps
        layer_num: Layer number for naming
    
    Returns:
        List of pooled activation maps
    """
    # Step 1: Apply filters
    activation_maps = apply_filters(image, filters)
    
    # Step 2: Select top X activations
    filtered_maps = select_top_activations(activation_maps, top_x)
    
    # Step 3: Save visualization
    layer_dir = os.path.join(output_dir, f"maps_layer_{layer_num}")
    save_activation_maps(filtered_maps, layer_dir, prefix=f"layer{layer_num}")
    
    # Step 4: Max pool
    pooled_maps = max_pool_all(filtered_maps)
    
    return pooled_maps


def main():
    """
    Main function to demonstrate the pipeline.
    """
    # Example: Load MNIST image (28x28)
    # For demonstration, create a random image
    image = np.random.rand(28, 28)
    
    # Create random 3x3 filters (in practice, these would be learned)
    num_filters_layer1 = 8
    filters_layer1 = [np.random.randn(3, 3) for _ in range(num_filters_layer1)]
    
    # Process layer 1
    print("Processing Layer 1...")
    top_x = 50  # Keep top 50 activations per filter
    pooled_layer1 = process_layer(image, filters_layer1, top_x, ".", layer_num=1)
    
    print(f"Layer 1 complete. Output shape: {pooled_layer1[0].shape}")
    
    # For layer 2, we would need to handle multiple input maps
    # This is a simplified demonstration
    num_filters_layer2 = 16
    filters_layer2 = [np.random.randn(3, 3) for _ in range(num_filters_layer2)]
    
    # Process each pooled map from layer 1 (simplified - in practice you'd combine them)
    print("Processing Layer 2...")
    for i, pooled_map in enumerate(pooled_layer1):
        if pooled_map.shape[0] >= 3 and pooled_map.shape[1] >= 3:
            pooled_layer2 = process_layer(pooled_map, filters_layer2, top_x, ".", layer_num=2)
            print(f"Layer 2 (map {i}) complete. Output shape: {pooled_layer2[0].shape}")
            break  # Just demonstrate with first map
    
    print("Pipeline demonstration complete!")


if __name__ == "__main__":
    main()
