import numpy as np
from .tensor import Tensor
from .ops import get_activation
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns -> Tensor of shape (batch_size, output_size)
        """
        z = x @ self.weights + self.biases
        return self._activation_fn(z)

    def parameters(self) -> list:
        """Used by the optimizer"""
        return [self.weights, self.biases]

    def zero_grad(self) -> None:
        self.weights.zero_grad()
        self.biases.zero_grad()

    def __repr__(self) -> str:
        w_shape = self.weights.shape
        return (
            f"ADLayer(in={w_shape[0]}, out={w_shape[1]}, "
            f"activation='{self._activation_name}')"
        )