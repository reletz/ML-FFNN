import numpy as np

class Initializer:
    """Base class for weight initialization methods."""
    def initialize(self, shape):
        """Initialize weights/biases with given shape.
        
        Args:
            shape: Tuple of integers representing the desired array shape
            
        Returns:
            NumPy array of the specified shape with initialized values
        """
        raise NotImplementedError("Subclasses must implement initialize()")


class Zero(Initializer):
    def initialize(self, shape):
        return np.zeros(shape)


class Uniform(Initializer):
    """Initialize with random values from uniform distribution.
    
    Args:
        low: Lower bound of the uniform distribution (default: -1.0)
        high: Upper bound of the uniform distribution (default: 1.0)
        seed: Random seed (default: None)
    """
    def __init__(self, low=-1.0, high=1.0, seed=None):
        self.low = low
        self.high = high
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def initialize(self, shape):
        """
        Returns:
            NumPy array with values sampled from Uniform[low, high]
        """
        return self.rng.uniform(self.low, self.high, shape)


class Normal(Initializer):
    def __init__(self, mean=0.0, variance=1.0, seed=None):
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(variance)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def initialize(self, shape):
        """
        Returns:
            NumPy array with values sampled from N(mean, variance)
        """
        return self.rng.normal(self.mean, self.std, shape)
