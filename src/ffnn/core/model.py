from ..activations.Activation import Activation
from ..losses.Loss import Loss
from ..initializers.Initializer import Initializer
from ..regularizers.Regularizer import Regularizer
from ..optimizers.Optimizer import Optimizer
from .network import Network
from .layer import Layer
from typing import Optional
from tqdm import tqdm

import numpy as np
import pickle
import matplotlib.pyplot as plt

class Model:
    def __init__(
            self, 
            layer_sizes: list[int], 
            activations: list[Activation], 
            loss: Loss, 
            initializer: Initializer,
            optimizer: Optimizer,
            regularizer: Optional[Regularizer]=None
            ) -> None : 
        self.network: Network = Network()
        self.loss_fn: Loss = loss
        self.optimizer: Optimizer = optimizer
        self.regularizer: Optional[Regularizer] = regularizer

        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Number of activations ({len(activations)}) must match "
                f"number of layers ({len(layer_sizes) - 1})"
            )
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            layer = Layer(input_size, output_size, activations[i], initializer)
            self.network.add_layer(layer)
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray, 
            X_val: np.ndarray, 
            y_val: np.ndarray,
            epochs: int,
            batch_size: int,
            verbose: int,
            learning_rate: Optional[float] = None
        ) -> dict[list[float],list[float]]:

        if learning_rate is not None:
            self.optimizer.set_learning_rate(learning_rate)

        history:dict = {"train_loss": [], "val_loss": []}
        train_samples_num = X_train.shape[0]
        epoch_iteator = range(epochs)
        if verbose == 1:
            epoch_iterator = tqdm(epoch_iteator, desc="Training")
        else:
            epoch_iterator = range(epochs)

        for epoch in epoch_iterator:
            mini_batches = self._get_mini_batches(X_train, y_train, batch_size)
            epoch_loss = 0
            for X_batch, y_batch in mini_batches:
                # Forward pass
                preds = self.predict(X_batch)
                batch_loss = self.evaluate(X_batch, y_batch)
                epoch_loss += batch_loss * X_batch.shape[0]

                # backward pass
                loss_grad = self.loss_fn.gradient(y_batch, preds)
                self.network.backward(loss_grad)

                if self.regularizer is not None:
                    for layer in self.network.layers:
                        if hasattr(layer, 'weights') and layer.weights is not None:
                            reg_grad = self.regularizer.gradient(layer.weights)
                            layer.weight_gradients += reg_grad
                
                params = []
                grads = []
                for layer in self.network.layers:
                    params.append(layer.weights)
                    params.append(layer.biases)
                    grads.append(layer.weight_gradients)
                    grads.append(layer.bias_gradients)

                self.optimizer.update(params, grads)

            epoch_avg_loss = epoch_loss / train_samples_num
            history["train_loss"].append(epoch_avg_loss)

            val_preds = self.network.forward(X_val)
            val_loss = self.loss_fn.compute(y_val, val_preds)
            history["val_loss"].append(val_loss)

            if verbose == 1:
                epoch_iterator.set_postfix({
                    'train_loss': f'{epoch_avg_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}'
                })

        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.network.forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.loss_fn.compute(y, self.predict(X))
    
    def plot_weight_distribution(self, layer_indices: list[int]) -> None:
        num_layers = len(layer_indices)
        if num_layers == 0:
            print("No layers specified.")
            return

        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
        if num_layers == 1:
            axes = [axes]
        
        for idx, layer_idx in enumerate(layer_indices):
            if layer_idx < 0 or layer_idx >= len(self.network.layers):
                print(f"Warning: Layer index {layer_idx} out of range. Skipping.")
                continue
            
            layer = self.network.layers[layer_idx]
            weights = layer.weights.flatten()

            axes[idx].hist(weights, bins=50, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Layer {layer_idx} Weights')
            axes[idx].set_xlabel('Weight Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

            mean_val = np.mean(weights)
            std_val = np.std(weights)
            axes[idx].axvline(mean_val, color='r', linestyle='--', 
                            label=f'Mean: {mean_val:.4f}')
            axes[idx].axvline(mean_val + std_val, color='g', linestyle='--', 
                            label=f'Std: {std_val:.4f}')
            axes[idx].axvline(mean_val - std_val, color='g', linestyle='--')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_distribution(self, layer_indices: list[int]) -> None:
        num_layers = len(layer_indices)
        if num_layers == 0:
            print("No layers specified.")
            return

        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
        if num_layers == 1:
            axes = [axes]
        
        for idx, layer_idx in enumerate(layer_indices):
            if layer_idx < 0 or layer_idx >= len(self.network.layers):
                print(f"Warning: Layer index {layer_idx} out of range. Skipping.")
                continue
            
            layer = self.network.layers[layer_idx]
            
            if layer.weight_gradients is None:
                print(f"Warning: No gradients computed for layer {layer_idx} yet. "
                      "Run a forward and backward pass first.")
                continue
            
            gradients = layer.weight_gradients.flatten()
            
            axes[idx].hist(gradients, bins=50, alpha=0.7, edgecolor='black', color='orange')
            axes[idx].set_title(f'Layer {layer_idx} Gradients')
            axes[idx].set_xlabel('Gradient Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)

            mean_val = np.mean(gradients)
            std_val = np.std(gradients)
            axes[idx].axvline(mean_val, color='r', linestyle='--', 
                            label=f'Mean: {mean_val:.4e}')
            axes[idx].axvline(mean_val + std_val, color='g', linestyle='--', 
                            label=f'Std: {std_val:.4e}')
            axes[idx].axvline(mean_val - std_val, color='g', linestyle='--')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str):
        """Save weights, biases, and model config to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> None:
        """Load model from a file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _get_mini_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> np.ndarray:
        """Shuffle and split data into mini-batches"""
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        mini_batches = []

        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]
            mini_batches.append((X_batch,y_batch))

        return mini_batches