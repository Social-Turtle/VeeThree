import numpy as np

class L0NDD:
    def __init__(self, length=3):
        self.length = length

    def process_window(self, window):
        """
        Process a 1D sequence of pixels.
        Returns: 
            contrast (float) if monotonic, else 0.0
            direction (int): 1 for increasing, -1 for decreasing, 0 for none
        """
        if len(window) != self.length:
            return 0.0, 0
            
        # Check increasing
        is_increasing = True
        for i in range(1, self.length):
            if window[i] <= window[i-1]:
                is_increasing = False
                break
        
        if is_increasing:
            contrast = window[-1] - window[0]
            return contrast, 1

        # Check decreasing
        is_decreasing = True
        for i in range(1, self.length):
            if window[i] >= window[i-1]:
                is_decreasing = False
                break
        
        if is_decreasing:
            contrast = window[0] - window[-1]
            return contrast, -1
            
        return 0.0, 0

class L0FilterBank:
    def __init__(self, length=3):
        self.length = length
        self.ndd = L0NDD(length)

    def process(self, image, channel_limit=None):
        """
        Apply L0 NDDs to the image.
        
        Args:
            image: 28x28 numpy array
            channel_limit: int or None. If set, only this many strongest filters
                           will be kept for each pixel. Others set to 0.
            
        Returns:
            results: dict containing:
                - 'max': 28x28 array (fastest firing overall)
                - 'horizontal': 28x28 array
                - 'vertical': 28x28 array
                - 'diagonal1': 28x28 array (down-right)
                - 'diagonal2': 28x28 array (down-left)
        """
        h, w = image.shape
        
        # Initialize maps
        # We'll use a temporary structure to hold values for sorting per pixel if needed
        # Or just build the maps and then post-process?
        # Post-processing per pixel is cleaner.
        
        maps = {
            'horizontal': np.zeros((h, w)),
            'vertical': np.zeros((h, w)),
            'diagonal1': np.zeros((h, w)),
            'diagonal2': np.zeros((h, w))
        }
        
        max_map = np.zeros((h, w))
        
        keys = ['horizontal', 'vertical', 'diagonal1', 'diagonal2']
        
        for y in range(h):
            for x in range(w):
                # Calculate all contrasts first
                current_contrasts = {}
                
                # Horizontal
                c_h = 0.0
                if x + self.length <= w:
                    window = image[y, x:x+self.length]
                    c, _ = self.ndd.process_window(window)
                    if c > 0: c_h = c
                current_contrasts['horizontal'] = c_h
                
                # Vertical
                c_v = 0.0
                if y + self.length <= h:
                    window = image[y:y+self.length, x]
                    c, _ = self.ndd.process_window(window)
                    if c > 0: c_v = c
                current_contrasts['vertical'] = c_v

                # Diagonal 1 (Down-Right)
                c_d1 = 0.0
                if y + self.length <= h and x + self.length <= w:
                    window = np.array([image[y+i, x+i] for i in range(self.length)])
                    c, _ = self.ndd.process_window(window)
                    if c > 0: c_d1 = c
                current_contrasts['diagonal1'] = c_d1

                # Diagonal 2 (Down-Left)
                c_d2 = 0.0
                if y + self.length <= h and x - self.length + 1 >= 0:
                    window = np.array([image[y+i, x-i] for i in range(self.length)])
                    c, _ = self.ndd.process_window(window)
                    if c > 0: c_d2 = c
                current_contrasts['diagonal2'] = c_d2

                # Logic for Channel Limiting
                # Sort values descending
                active_values = [(k, v) for k,v in current_contrasts.items() if v > 0]
                active_values.sort(key=lambda x: x[1], reverse=True)
                
                if not active_values:
                    continue
                    
                max_map[y, x] = active_values[0][1]

                # If limit applied, filter
                if channel_limit is not None:
                    # Keep top N
                    kept_keys = {item[0] for item in active_values[:channel_limit]}
                    
                    for k in keys:
                        if k in kept_keys:
                            maps[k][y, x] = current_contrasts[k]
                        else:
                            maps[k][y, x] = 0.0
                else:
                    # No limit, just assign
                    for k in keys:
                        maps[k][y, x] = current_contrasts[k]

        maps['max'] = max_map
        return maps
