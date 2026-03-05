from .layer import Layer
import numpy as np

class Network:
    def __init__(self) -> None:
        self.layers: list[Layer] = []
    
    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, loss_gradient: np.ndarray) -> None:
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)