from .tensor import Tensor
from .layer import ADLayer


class ADNetwork:
    def __init__(self) -> None:
        self.layers: list[ADLayer] = []

    def add_layer(self, layer: ADLayer) -> None:
        self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : Tensor of shape (batch_size, input_size) -> Tensor of shape (batch_size, output_size)
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def __repr__(self) -> str:
        lines = ["ADNetwork("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)
