import numpy as np
from PIL import Image
import os

def value_to_color(value, min_val, max_val):
    """
    Map value to color:
    High value (Fast/Strong) -> Red
    Low value (Slow/Weak) -> Blue
    Ranges between are interpolated.
    """
    if max_val == min_val:
        return (0, 0, 255) # Default to blue if uniform
        
    # Normalize 0 to 1
    # Check bounds to ensure we don't go out of range if value is outside min/max
    # (though typically we pass min/max of the data or a superset)
    clamped_val = max(min_val, min(value, max_val))
    
    norm = (clamped_val - min_val) / (max_val - min_val)
    
    # Simple interpolation
    # 1.0 -> (255, 0, 0) Red
    # 0.0 -> (0, 0, 255) Blue
    
    r = int(norm * 255)
    b = int((1 - norm) * 255)
    g = 0
    
    return (r, g, b)

def save_visualization(data_map, path, global_min=None, global_max=None):
    """
    Save a visualization of the data map.
    Zeros are black. Positive values are Red->Blue gradient.
    
    Args:
        data_map: 2D numpy array
        path: Output path
        global_min: logic override for min value of gradient
        global_max: logic override for max value of gradient
    """
    h, w = data_map.shape
    img_array = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Filter valid values
    valid_mask = data_map > 0
    
    # If no valid values, output black image
    if not np.any(valid_mask):
        img = Image.fromarray(img_array, mode='RGB')
        img.save(path)
        return

    # Determine scaling range
    if global_min is not None:
        min_val = global_min
    else:
        min_val = np.min(data_map[valid_mask])
        
    if global_max is not None:
        max_val = global_max
    else:
        max_val = np.max(data_map[valid_mask])
    
    for y in range(h):
        for x in range(w):
            val = data_map[y, x]
            if val > 0:
                img_array[y, x] = value_to_color(val, min_val, max_val)
                
    img = Image.fromarray(img_array, mode='RGB')
    # Resize for visibility (scale up small grids)
    if w < 100:
        scale = 10
        img = img.resize((w * scale, h * scale), Image.NEAREST)
        
    img.save(path)

def save_digit_example(layer_dir, digit, data_map, sub_dir=None, global_min=None, global_max=None):
    """
    Save the example for the given digit.
    
    Args:
        layer_dir: e.g. "nddplayground/ndd_layer_1"
        digit: int (0-9)
        data_map: numpy array
        sub_dir: Optional subdirectory (e.g., "horizontal")
        global_min, global_max: For fixed scaling
    """
    if sub_dir:
        digits_dir = os.path.join(layer_dir, "digits", sub_dir)
    else:
        digits_dir = os.path.join(layer_dir, "digits")
        
    os.makedirs(digits_dir, exist_ok=True)
    
    path = os.path.join(digits_dir, f"{digit}.png")
    save_visualization(data_map, path, global_min, global_max)
