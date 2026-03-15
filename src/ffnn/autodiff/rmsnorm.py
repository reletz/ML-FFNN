import numpy as np
from .tensor import Tensor
from . import ops

class ADRMSNorm:
    def __init__(self, dim: int, eps: float = 1e-8):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, dim)
        rms = ops.sqrt(ops.mean(x * x, axis=1, keepdims=True) + self.eps)
        x_hat = x / rms
        return x_hat * self.gamma

    def parameters(self) -> list:
        return [self.gamma]

    def zero_grad(self) -> None:
        self.gamma.zero_grad()

    def __repr__(self) -> str:
        return f"ADRMSNorm(dim={self.gamma.shape[1]}, eps={self.eps})"