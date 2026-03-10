import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm

from .tensor import Tensor
from .network import ADNetwork
from .layer import ADLayer
from .ops import get_loss
from ..initializers.Initializer import Initializer, Normal
from ..optimizers.Optimizer import Optimizer
from ..regularizers.Regularizer import Regularizer


class ADModel:
    """
    layer_sizes  : list[int]   e.g. [input, hidden1, hidden2, output]
    activations  : list[str]   one per layer transition, e.g. ['relu', 'softmax']
    loss         : str         'mse' | 'bce' | 'cce' | 'binary_cross_entropy' | 'cross_entropy'
    optimizer    : Optimizer   e.g. GradientDescent(lr=0.01) or Adam()
    initializer  : Initializer weight initializer (default: Normal(0, 0.01))
    regularizer  : Regularizer optional L1/L2 regularizer
    """

    def __init__(
        self,
        layer_sizes: list,
        activations: list,
        loss: str,
        optimizer: Optimizer,
        initializer: Initializer = None,
        regularizer: Optional[Regularizer] = None,
    ) -> None:
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Number of activations ({len(activations)}) must match "
                f"number of layer transitions ({len(layer_sizes) - 1})"
            )

        if initializer is None:
            initializer = Normal(mean=0.0, variance=0.01)

        self.network = ADNetwork()
        for i in range(len(layer_sizes) - 1):
            self.network.add_layer(
                ADLayer(layer_sizes[i], layer_sizes[i + 1], activations[i], initializer)
            )

        self.loss_name: str = loss
        self._loss_fn = get_loss(loss)
        self.optimizer: Optimizer = optimizer
        self.regularizer: Optional[Regularizer] = regularizer

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        verbose: int = 1,
        learning_rate: Optional[float] = None,
    ) -> dict:
        if learning_rate is not None:
            self.optimizer.set_learning_rate(learning_rate)

        history = {"train_loss": [], "val_loss": []}
        n_train = X_train.shape[0]

        epoch_range = range(epochs)
        if verbose == 1:
            epoch_range = tqdm(epoch_range, desc="Training")

        for epoch in epoch_range:
            batches = self._get_mini_batches(X_train, y_train, batch_size)
            epoch_loss = 0.0

            for X_batch, y_batch in batches:
                # 1. forward pass
                x_t = Tensor.from_numpy(X_batch)
                y_t = Tensor.from_numpy(y_batch)
                pred = self.network.forward(x_t)

                # 2. compute loss (returns scalar Tensor)
                loss_t = self._loss_fn(pred, y_t)

                # 3. track raw loss value (+ regularization penalty)
                batch_loss_val = float(loss_t.data)
                if self.regularizer is not None:
                    for layer in self.network.layers:
                        batch_loss_val += self.regularizer.penalty(layer.weights.data)

                epoch_loss += batch_loss_val * X_batch.shape[0]

                # 4. backward -> fills .grad on every Tensor in the graph
                loss_t.backward()

                # 5. add regularizer gradient to weight gradients
                if self.regularizer is not None:
                    for layer in self.network.layers:
                        reg_grad = self.regularizer.gradient(layer.weights.data)
                        if layer.weights.grad is not None:
                            layer.weights.grad += reg_grad
                        else:
                            layer.weights.grad = reg_grad

                # 6. optimizer step -> passes raw numpy arrays (updated in-place)
                params = [p.data for p in self.network.parameters()]
                grads = [
                    p.grad if p.grad is not None else np.zeros_like(p.data)
                    for p in self.network.parameters()
                ]
                self.optimizer.update(params, grads)

                # 7. reset gradients for next iteration
                self.network.zero_grad()

            epoch_avg_loss = epoch_loss / n_train
            history["train_loss"].append(epoch_avg_loss)

            val_loss = self.evaluate(X_val, y_val)
            history["val_loss"].append(val_loss)

            if verbose == 1:
                epoch_range.set_postfix(
                    train_loss=f"{epoch_avg_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass only — returns a NumPy array."""
        x_t = Tensor.from_numpy(X)
        return self.network.forward(x_t).data

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Forward pass + loss — returns scalar loss value."""
        x_t = Tensor.from_numpy(X)
        y_t = Tensor.from_numpy(y)
        pred = self.network.forward(x_t)
        return float(self._loss_fn(pred, y_t).data)

    def save(self, filepath: str) -> None:
        """Serialize the full model to a file via pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "ADModel":
        """Deserialize a model previously saved with save()."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def plot_weight_distribution(self, layer_indices: list) -> None:
        num = len(layer_indices)
        if num == 0:
            print("No layers specified.")
            return
        fig, axes = plt.subplots(1, num, figsize=(5 * num, 4))
        if num == 1:
            axes = [axes]
        for idx, li in enumerate(layer_indices):
            if li < 0 or li >= len(self.network.layers):
                print(f"Warning: layer index {li} out of range. Skipping.")
                continue
            w = self.network.layers[li].weights.data.flatten()
            axes[idx].hist(w, bins=50, alpha=0.7, edgecolor="black")
            axes[idx].set_title(f"Layer {li} Weights")
            axes[idx].set_xlabel("Weight Value")
            axes[idx].set_ylabel("Frequency")
            m, s = np.mean(w), np.std(w)
            axes[idx].axvline(m, color="r", linestyle="--", label=f"Mean: {m:.4f}")
            axes[idx].axvline(m + s, color="g", linestyle="--", label=f"Std: {s:.4f}")
            axes[idx].axvline(m - s, color="g", linestyle="--")
            axes[idx].legend()
        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layer_indices: list) -> None:
        num = len(layer_indices)
        if num == 0:
            print("No layers specified.")
            return
        fig, axes = plt.subplots(1, num, figsize=(5 * num, 4))
        if num == 1:
            axes = [axes]
        for idx, li in enumerate(layer_indices):
            if li < 0 or li >= len(self.network.layers):
                print(f"Warning: layer index {li} out of range. Skipping.")
                continue
            g = self.network.layers[li].weights.grad
            if g is None:
                print(f"Warning: no gradients for layer {li} yet. Run fit() first.")
                continue
            g = g.flatten()
            axes[idx].hist(g, bins=50, alpha=0.7, edgecolor="black", color="orange")
            axes[idx].set_title(f"Layer {li} Gradients")
            axes[idx].set_xlabel("Gradient Value")
            axes[idx].set_ylabel("Frequency")
            m, s = np.mean(g), np.std(g)
            axes[idx].axvline(m, color="r", linestyle="--", label=f"Mean: {m:.4e}")
            axes[idx].axvline(m + s, color="g", linestyle="--", label=f"Std: {s:.4e}")
            axes[idx].axvline(m - s, color="g", linestyle="--")
            axes[idx].legend()
        plt.tight_layout()
        plt.show()

    def _get_mini_batches(self, X, y, batch_size):
        n = X.shape[0]
        indices = np.random.permutation(n)
        X, y = X[indices], y[indices]
        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batches.append((X[start:end], y[start:end]))
        return batches

    def __repr__(self) -> str:
        return f"ADModel(loss='{self.loss_name}', network={self.network})"