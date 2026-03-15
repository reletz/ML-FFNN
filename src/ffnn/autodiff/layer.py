import numpy as np
from .tensor import Tensor
from .ops import get_activation
from .rmsnorm import ADRMSNorm
from ..initializers.Initializer import Initializer, Normal


class ADLayer:
    """
    A single fully-connected (dense) layer using autodiff Tensors

    input_size  : int
    output_size : int
    activation  : str -> one of 'linear', 'relu', 'sigmoid', 'tanh', 'softmax'
    init_fn     : Initializer -> weight initializer (default: Normal(0, 0.01))
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "linear",
        init_fn: Initializer = None,
        use_rmsnorm: bool = False,
        rmsnorm_eps: float = 1e-8,
    ) -> None:
        if init_fn is None:
            init_fn = Normal(mean=0.0, variance=0.01)

        self.weights = Tensor(
            init_fn.initialize((input_size, output_size)),
            requires_grad=True,
        )
        self.biases = Tensor(
            np.zeros((1, output_size)),
            requires_grad=True,
        )

        self._activation_name: str = activation.lower()
        self._activation_fn = get_activation(self._activation_name)

        self.use_rmsnorm = use_rmsnorm
        self.rmsnorm = (
            ADRMSNorm(dim=output_size, eps=rmsnorm_eps)
            if use_rmsnorm
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns -> Tensor of shape (batch_size, output_size)
        """
        z = x @ self.weights + self.biases

        if self.rmsnorm is not None:
            z = self.rmsnorm.forward(z)

        return self._activation_fn(z)

    def parameters(self) -> list:
        """Used by the optimizer"""
        params = [self.weights, self.biases]
        if self.rmsnorm is not None:
            params.extend(self.rmsnorm.parameters())
        return params

    def zero_grad(self) -> None:
        self.weights.zero_grad()
        self.biases.zero_grad()
        if self.rmsnorm is not None:
            self.rmsnorm.zero_grad()

    def __repr__(self) -> str:
        w_shape = self.weights.shape
        rms = f", rmsnorm={self.rmsnorm is not None}"
        return (
            f"ADLayer(in={w_shape[0]}, out={w_shape[1]}, "
            f"activation='{self._activation_name}'{rms})"
        )