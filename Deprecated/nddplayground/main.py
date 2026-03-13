import os
import sys
import numpy as np

# Add parent directory to path to import local modules if needed, 
# though we are using relative imports from within nddplayground package style
# But for running as script:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nddplayground.data_loader import load_mnist_generator
from nddplayground.l0_layer import L0FilterBank
from nddplayground.pooling_layer import PoolingLayer
from nddplayground.visualization import save_digit_example

def main():
    print("Initializing NDD Playground...")
    
    # Configuration
    CHANNEL_LIMIT = 2 # How many channels allowed to fire per pixel
    
    print(f"Configuration: CHANNEL_LIMIT={CHANNEL_LIMIT}")
    
    # Initialize Layers
    # L0: Length 3 (Pixel, Pixel+1, Pixel+2)
    l0 = L0FilterBank(length=3)
    
    # Pooling: 2x2 grid, keep top 1
    pool1 = PoolingLayer(grid_size=2, top_e=1)
    
    # Track which digits we've seen
    seen_digits = set()
    needed_digits = set(range(10))
    
    print("Starting processing loop...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    l1_dir = os.path.join(base_dir, "ndd_layer_1")
    p1_dir = os.path.join(base_dir, "pooling_layer_1")
    
    mnist_dir = os.path.join(os.path.dirname(base_dir), "mnist/mnist")
    
    data_gen = load_mnist_generator(data_dir=mnist_dir)
    
    if not data_gen:
        print("Failed to load data generator.")
        return

    count = 0
    for image, label in data_gen:
        if label in seen_digits:
            continue
            
        print(f"Processing digit: {label}")
        
        # 1. L0 NDD Layer
        # Returns dict: {'max', 'horizontal', 'vertical', 'diagonal1', 'diagonal2'}
        # Now with channel limiting!
        l0_maps = l0.process(image, channel_limit=CHANNEL_LIMIT)
        
        # Calculate global min/max for L0
        all_vals_0 = []
        for k, m in l0_maps.items():
            valid = m[m > 0]
            if len(valid) > 0:
                all_vals_0.append(valid)
        
        if all_vals_0:
            concat_vals = np.concatenate(all_vals_0)
            g_min_0 = np.min(concat_vals)
            g_max_0 = np.max(concat_vals)
        else:
            g_min_0, g_max_0 = 0, 1
            
        # Save L0 outputs
        save_digit_example(l1_dir, label, l0_maps['max'], global_min=g_min_0, global_max=g_max_0)
        for key in ['horizontal', 'vertical', 'diagonal1', 'diagonal2']:
            save_digit_example(l1_dir, label, l0_maps[key], sub_dir=key, global_min=g_min_0, global_max=g_max_0)
        
        # 2. Pooling Layer
        # Process ALL maps (including individual filters)
        pool1_maps = pool1.process(l0_maps)
        
        # Calculate global min/max for Pool1 (likely similar to L0 but maybe different distribution)
        all_vals_1 = []
        for k, m in pool1_maps.items():
            valid = m[m > 0]
            if len(valid) > 0:
                all_vals_1.append(valid)
                
        if all_vals_1:
            concat_vals = np.concatenate(all_vals_1)
            g_min_1 = np.min(concat_vals)
            g_max_1 = np.max(concat_vals)
        else:
            g_min_1, g_max_1 = 0, 1

        # Save Pool1 outputs (with structure!)
        save_digit_example(p1_dir, label, pool1_maps['max'], global_min=g_min_1, global_max=g_max_1)
        for key in ['horizontal', 'vertical', 'diagonal1', 'diagonal2']:
            save_digit_example(p1_dir, label, pool1_maps[key], sub_dir=key, global_min=g_min_1, global_max=g_max_1)

        seen_digits.add(label)
        count += 1
        
        if seen_digits == needed_digits:
            print("Collected all digits 0-9. Stopping.")
            break
            
    print("Done!")

if __name__ == "__main__":
    main()
