import numpy as np

class PoolingLayer:
    def __init__(self, grid_size=2, top_e=1):
        self.grid_size = grid_size  # D
        self.top_e = top_e          # E
        
    def process_single(self, input_map):
        """
        Pool a single 2D input map.
        """
        h, w = input_map.shape
        new_h = h // self.grid_size
        new_w = w // self.grid_size
        
        pooled_map = np.zeros((new_h, new_w))
        
        for i in range(new_h):
            for j in range(new_w):
                row_start = i * self.grid_size
                row_end = row_start + self.grid_size
                col_start = j * self.grid_size
                col_end = col_start + self.grid_size
                
                grid = input_map[row_start:row_end, col_start:col_end]
                
                flat = grid.flatten()
                flat = flat[flat > 0]
                
                if len(flat) > 0:
                    flat.sort()
                    top_values = flat[-self.top_e:]
                    pooled_map[i, j] = np.max(top_values)
                else:
                    pooled_map[i, j] = 0.0
                    
        return pooled_map

    def process(self, input_data):
        """
        Pool the input data.
        
        Args:
            input_data: 2D numpy array OR dictionary of 2D numpy arrays
            
        Returns:
            pooled_data: Same structure as input (array or dict)
        """
        if isinstance(input_data, dict):
            return {k: self.process_single(v) for k, v in input_data.items()}
        else:
            return self.process_single(input_data)
