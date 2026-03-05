import numpy as np

class Regularizer:
    """ABC"""
    def penalty(self, weights):
        """Compute regularization penalty term
        
        Args:
            weights: NumPy array of weight values
            
        Returns:
            Scalar penalty value
        """
        raise NotImplementedError("Subclasses must implement penalty()")
    
    def gradient(self, weights):
        """Compute regularization gradient w.r.t. weights
        
        Args:
            weights: NumPy array of weight values
            
        Returns:
            NumPy array of same shape as weights with gradient contribution
        """
        raise NotImplementedError("Subclasses must implement gradient()")


class L1(Regularizer):
    """L1 (Lasso) regularization.
    
    Penalty: lambda * Sigma|w|
    Gradient: lambda * sign(w)
    
    Args:
        lambda_: default -> 0.01)
    """
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
    
    def penalty(self, weights):
        return self.lambda_ * np.sum(np.abs(weights))
    
    def gradient(self, weights):
        return self.lambda_ * np.sign(weights)


class L2(Regularizer):
    """L2 (Ridge) regularization.
    
    Penalty: (lambda/2) * Σ w²
    Gradient: lambda * w
    
    Args:
        lambda_: default -> 0.01)
    """
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
    
    def penalty(self, weights):
        return (self.lambda_ / 2) * np.sum(weights ** 2)
    
    def gradient(self, weights):
        return self.lambda_ * weights