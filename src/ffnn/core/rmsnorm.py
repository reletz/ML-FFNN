import numpy as np


class RMSNorm:

    def __init__(self, dim: int, eps: float = 1e-8, gamma_init: float = 1.0) -> None:
        self.dim = dim
        self.eps = eps

        self.gamma = np.ones((1, dim), dtype=float) * gamma_init
        self.gamma_grad = np.zeros((1, dim), dtype=float)

        self.input_cache = None
        self.rms_cache = None
        self.normalized_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        mean_square = np.mean(x ** 2, axis=1, keepdims=True)
        self.rms_cache = np.sqrt(mean_square + self.eps)
        self.normalized_cache = x / self.rms_cache
        return self.gamma * self.normalized_cache

    def backward(self, grad_output: np.ndarray) -> np.ndarray:

        if self.input_cache is None or self.rms_cache is None:
            raise ValueError("RMSNorm.backward() called before forward().")

        batch_size, dim = grad_output.shape

        self.gamma_grad = np.mean(grad_output * self.normalized_cache, axis=0, keepdims=True)

        grad_scaled = grad_output * self.gamma
        dot = np.sum(grad_scaled * self.input_cache, axis=1, keepdims=True)

        grad_input = (
            grad_scaled / self.rms_cache
            - self.input_cache * dot / (dim * (self.rms_cache ** 3))
        )

        return grad_input

    def zero_grad(self) -> None:
        self.gamma_grad = np.zeros_like(self.gamma)
