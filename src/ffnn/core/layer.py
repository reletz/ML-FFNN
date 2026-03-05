import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_fn, init_fn):
        self.weights = init_fn.initialize((input_size, output_size))
        self.biases = init_fn.initialize((1, output_size))
        self.weight_gradients = None
        self.bias_gradients = None
        self.activation = activation_fn
        self.input_cache = None
        self.z_cache = None

    def forward(self, X):
        self.input_cache = X
        z = X @ self.weights + self.biases
        self.z_cache = z
        return self.activation.forward(z)
    
    def backward(self, grad_output):
        batch_size = self.input_cache[0]
        dz = grad_output * self.activation.backward(self.z_cache)
        self.weight_gradients = self.input_cache.T @ dz / batch_size
        self.bias_gradients = np.mean(dz, axis=0)
        return dz @ self.weights.T

