class NDD:
    """
    A Neural Delay Detector (NDD) that operates on discrete tick-based time.
    
    An NDD has n inputs, each receiving a tick value. If the tick values
    from input 0 to input n-1 are in strictly increasing order, the NDD
    transmits the current tick count to its output.
    """
    
    def __init__(self, n: int):
        """
        Initialize an NDD with n inputs.
        
        Args:
            n: Number of inputs for this NDD.
        """
        self.n = n
        self.input_ticks = [None] * n  # Tick values received at each input
        self.output = None  # The output tick value (if fired)
    
    def receive(self, input_index: int, tick_value: int):
        """
        Receive a tick value at a specific input.
        
        Args:
            input_index: Which input (0 to n-1) is receiving the value.
            tick_value: The tick value being received.
        """
        if 0 <= input_index < self.n:
            self.input_ticks[input_index] = tick_value
    
    def evaluate(self, current_tick: int) -> int | None:
        """
        Evaluate if the NDD should fire based on current input tick values.
        
        If all inputs have values and they are in strictly increasing order
        (from input 0 to input n-1), the NDD fires and outputs the current tick.
        
        Args:
            current_tick: The current tick count.
            
        Returns:
            The current tick if the NDD fires, None otherwise.
        """
        # Check if all inputs have received values
        if None in self.input_ticks:
            self.output = None
            return None
        
        # Check if tick values are in strictly increasing order
        for i in range(1, self.n):
            if self.input_ticks[i] <= self.input_ticks[i - 1]:
                self.output = None
                return None
        
        # All conditions met - fire and output current tick
        self.output = current_tick
        return self.output
    
    def reset(self):
        """Reset all inputs and output to None."""
        self.input_ticks = [None] * self.n
        self.output = None
    
    def __repr__(self):
        return f"NDD(n={self.n}, inputs={self.input_ticks}, output={self.output})"
