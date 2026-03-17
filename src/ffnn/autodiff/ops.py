"""
Differentiable operations for the autodiff engine.

Every function accepts Tensor(s) and returns a new Tensor whose
_backward closure propagates the gradient back to its inputs.
"""

import numpy as np
from .tensor import Tensor, _accum

def clip(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """
    Clip values to [min_val, max_val].
    Gradient is 0 at clipped boundaries (straight-through otherwise).
    """
    safe = np.clip(x.data, min_val, max_val)
    out = Tensor(safe, requires_grad=x.requires_grad, _op="clip")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            mask = ((x.data >= min_val) & (x.data <= max_val)).astype(np.float64)
            x.grad = _accum(x.grad, out.grad * mask)

    out._backward = _backward
    return out

def sum(x: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Sum elements along axis (all axes if None)"""
    out = Tensor(
        np.sum(x.data, axis=axis, keepdims=keepdims),
        requires_grad=x.requires_grad,
        _op="sum",
    )
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis=axis)
            x.grad = _accum(x.grad, np.broadcast_to(g, x.data.shape).copy())

    out._backward = _backward
    return out


def mean(x: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Mean of elements along axis (all axes if None)"""
    n = x.data.size if axis is None else x.data.shape[axis]
    out = Tensor(
        np.mean(x.data, axis=axis, keepdims=keepdims),
        requires_grad=x.requires_grad,
        _op="mean",
    )
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            g = out.grad / n
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis=axis)
            x.grad = _accum(x.grad, np.broadcast_to(g, x.data.shape).copy())

    out._backward = _backward
    return out

def exp(x: Tensor) -> Tensor:
    """Element-wise e^x.  Backward: grad * e^x"""
    result = np.exp(x.data)
    out = Tensor(result, requires_grad=x.requires_grad, _op="exp")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            x.grad = _accum(x.grad, out.grad * result)

    out._backward = _backward
    return out

def log(x: Tensor, eps: float = 1e-15) -> Tensor:
    """
    Values are clipped to [eps, +inf] for numerical stability.
    Backward: grad / x
    """
    safe = np.clip(x.data, eps, None)
    out = Tensor(np.log(safe), requires_grad=x.requires_grad, _op="log")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            x.grad = _accum(x.grad, out.grad / safe)

    out._backward = _backward
    return out


def sqrt(x: Tensor, eps: float = 1e-15) -> Tensor:
    """
    Values are clipped to [eps, +inf] for numerical stability.
    Backward: grad / (2 * sqrt(x))
    """
    safe = np.clip(x.data, eps, None)
    result = np.sqrt(safe)
    out = Tensor(result, requires_grad=x.requires_grad, _op="sqrt")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            x.grad = _accum(x.grad, out.grad / (2.0 * result))

    out._backward = _backward
    return out

def linear(x: Tensor) -> Tensor:
    """Identity activation -> passes value and gradient through unchanged"""
    out = Tensor(x.data.copy(), requires_grad=x.requires_grad, _op="linear")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            x.grad = _accum(x.grad, out.grad)

    out._backward = _backward
    return out


def relu(x: Tensor) -> Tensor:
    """
    ReLU: max(0, x)
    Backward: grad * (x > 0)  -> gradient is 0 for negative inputs
    """
    result = np.maximum(0.0, x.data)
    out = Tensor(result, requires_grad=x.requires_grad, _op="relu")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            x.grad = _accum(x.grad, out.grad * (x.data > 0).astype(np.float64))

    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid: 1 / (1 + exp(-x))
    Backward: grad * sigmoid(x) * (1 - sigmoid(x))
    """
    result = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(result, requires_grad=x.requires_grad, _op="sigmoid")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            x.grad = _accum(x.grad, out.grad * result * (1.0 - result))

    out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    """
    Tanh activation.
    Backward: grad * (1 - tanh²(x)).
    """
    result = np.tanh(x.data)
    out = Tensor(result, requires_grad=x.requires_grad, _op="tanh")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            x.grad = _accum(x.grad, out.grad * (1.0 - result ** 2))

    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Numerically stable softmax along axis
    Backward uses the efficient vector-Jacobian product:
        dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
    """
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    out = Tensor(result, requires_grad=x.requires_grad, _op="softmax")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            dot = np.sum(out.grad * result, axis=axis, keepdims=True)
            x.grad = _accum(x.grad, result * (out.grad - dot))

    out._backward = _backward
    return out


def leakyrelu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Leaky ReLU: x if x > 0, else alpha * x
    Backward: grad * (1 if x > 0, else alpha)
    """
    result = np.where(x.data > 0, x.data, alpha * x.data)
    out = Tensor(result, requires_grad=x.requires_grad, _op="leakyrelu")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            grad_mask = np.where(x.data > 0, 1.0, alpha)
            x.grad = _accum(x.grad, out.grad * grad_mask)

    out._backward = _backward
    return out


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """
    ELU: x if x > 0, else alpha * (exp(x) - 1)
    Backward: grad * (1 if x > 0, else alpha * exp(x))
    """
    result = np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1.0))
    out = Tensor(result, requires_grad=x.requires_grad, _op="elu")
    out._prev = {x}

    def _backward() -> None:
        if x.requires_grad:
            grad_mask = np.where(x.data > 0, 1.0, alpha * np.exp(x.data))
            x.grad = _accum(x.grad, out.grad * grad_mask)

    out._backward = _backward
    return out


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    return mean(diff * diff)


def binary_cross_entropy(pred: Tensor, target: Tensor, eps: float = 1e-15) -> Tensor:
    pred_c = clip(pred, eps, 1.0 - eps)
    ones = Tensor(np.ones_like(target.data))
    return mean(-(target * log(pred_c) + (ones - target) * log(ones - pred_c)))


def cross_entropy(pred: Tensor, target: Tensor, eps: float = 1e-15) -> Tensor:
    log_pred = log(pred, eps=eps)
    per_sample = sum(target * log_pred, axis=1)
    return mean(-per_sample)

_ACTIVATIONS = {
    "linear":  linear,
    "relu":    relu,
    "sigmoid": sigmoid,
    "tanh":    tanh,
    "softmax": softmax,
    "leakyrelu": leakyrelu,
    "elu": elu
}

_LOSSES = {
    "mse":                   mse_loss,
    "binary_cross_entropy":  binary_cross_entropy,
    "bce":                   binary_cross_entropy,
    "cross_entropy":         cross_entropy,
    "cce":                   cross_entropy,
}


def get_activation(name: str):
    """Return an activation function by name (case-insensitive)"""
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]


def get_loss(name: str):
    """Return a loss function by name (case-insensitive)"""
    key = name.lower()
    if key not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_LOSSES)}")
    return _LOSSES[key]
