import numpy as np
from typing import Optional, Union, Callable, Set


class Tensor:
    """
    Wrapper around a NumPy array that tracks operations for automatic
    reverse-mode differentiation (autograd).

    Attributes
    ----------
    data          : np.ndarray   the numeric values
    grad          : np.ndarray   accumulated gradient (None until backward)
    requires_grad : bool         whether to track gradients for this tensor
    _backward     : Callable     closure that propagates gradient to parents
    _prev         : set[Tensor]  parent tensors in the computation graph
    _op           : str          name of the op that created this tensor (debug)
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _op: str = "",
    ) -> None:
        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data.astype(np.float64)
        else:
            self.data: np.ndarray = np.array(data, dtype=np.float64)

        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = requires_grad
        self._backward: Callable = lambda: None
        self._prev: Set["Tensor"] = set()
        self._op: str = _op

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Trigger reverse-mode autodiff from this tensor

        grad : optional initial gradient (must match this tensor's shape)
            If None and tensor is scalar, uses ones_like(data)
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.ndim == 0 or self.data.size == 1:
                self.grad = np.ones_like(self.data)
            else:
                raise RuntimeError(
                    "backward() without 'grad' argument only works on scalar tensors. "
                    "Pass an explicit grad array for non-scalar tensors."
                )
        else:
            self.grad = np.array(grad, dtype=np.float64)

        # Build topological order (DFS post-order)
        topo: list["Tensor"] = []
        visited: set["Tensor"] = set()

        def build_topo(t: "Tensor") -> None:
            if t not in visited:
                visited.add(t)
                for parent in t._prev:
                    build_topo(parent)
                topo.append(t)

        build_topo(self)

        # Propagate gradients in reverse topological order
        for t in reversed(topo):
            t._backward()

    def zero_grad(self) -> None:
        """Reset gradient to None."""
        self.grad = None

    def detach(self) -> "Tensor":
        """Return a new Tensor with the same data but no gradient tracking."""
        return Tensor(self.data.copy(), requires_grad=False)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying NumPy array."""
        return self.data.copy()

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}, "
            f"op='{self._op}', requires_grad={self.requires_grad})"
        )

    def __add__(self, other: "TensorLike") -> "Tensor":
        other = _as_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _op="add",
        )
        out._prev = {self, other}

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, _unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other.grad = _accum(other.grad, _unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: "TensorLike") -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: "TensorLike") -> "Tensor":
        other = _as_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _op="sub",
        )
        out._prev = {self, other}

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, _unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other.grad = _accum(other.grad, _unbroadcast(-out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __rsub__(self, other: "TensorLike") -> "Tensor":
        return _as_tensor(other).__sub__(self)

    def __mul__(self, other: "TensorLike") -> "Tensor":
        other = _as_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _op="mul",
        )
        out._prev = {self, other}

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(
                    self.grad, _unbroadcast(out.grad * other.data, self.data.shape)
                )
            if other.requires_grad:
                other.grad = _accum(
                    other.grad, _unbroadcast(out.grad * self.data, other.data.shape)
                )

        out._backward = _backward
        return out

    def __rmul__(self, other: "TensorLike") -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: "TensorLike") -> "Tensor":
        other = _as_tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _op="div",
        )
        out._prev = {self, other}

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(
                    self.grad, _unbroadcast(out.grad / other.data, self.data.shape)
                )
            if other.requires_grad:
                other.grad = _accum(
                    other.grad,
                    _unbroadcast(
                        -out.grad * self.data / (other.data ** 2), other.data.shape
                    ),
                )

        out._backward = _backward
        return out

    def __rtruediv__(self, other: "TensorLike") -> "Tensor":
        return _as_tensor(other).__truediv__(self)

    def __pow__(self, exp: Union[int, float]) -> "Tensor":
        assert isinstance(exp, (int, float)), "Exponent must be a Python int or float"
        out = Tensor(
            self.data ** exp,
            requires_grad=self.requires_grad,
            _op=f"pow({exp})",
        )
        out._prev = {self}

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(
                    self.grad, exp * (self.data ** (exp - 1)) * out.grad
                )

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, _op="neg")
        out._prev = {self}

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, -out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other: "TensorLike") -> "Tensor":
        """
        Matrix multiplication.  Gradient derivation (2-D case):
            out = self @ other
            d(loss)/d(self) = d(loss)/d(out) @ other.T
            d(loss)/d(other) = self.T @ d(loss)/d(out)
        """
        other = _as_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _op="matmul",
        )
        out._prev = {self, other}

        def _backward() -> None:
            if self.requires_grad:
                self.grad = _accum(self.grad, out.grad @ other.data.T)
            if other.requires_grad:
                other.grad = _accum(other.grad, self.data.T @ out.grad)

        out._backward = _backward
        return out

    @staticmethod
    def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> "Tensor":
        """Create a Tensor from an existing NumPy array."""
        return Tensor(arr, requires_grad=requires_grad)

    @staticmethod
    def zeros(shape, requires_grad: bool = False) -> "Tensor":
        """Create a zero-filled Tensor of the given shape."""
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape, requires_grad: bool = False) -> "Tensor":
        """Create a ones-filled Tensor of the given shape."""
        return Tensor(np.ones(shape), requires_grad=requires_grad)

TensorLike = Union[Tensor, np.ndarray, float, int]


def _as_tensor(x: TensorLike) -> Tensor:
    """Convert a scalar / ndarray to a Tensor (no-op if already a Tensor)."""
    if isinstance(x, Tensor):
        return x
    return Tensor(np.array(x, dtype=np.float64))


def _accum(existing: Optional[np.ndarray], grad: np.ndarray) -> np.ndarray:
    """
    Initialises to a copy of grad when existing is None
    """
    if existing is None:
        return np.array(grad, dtype=np.float64)
    return existing + grad


def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Reverse numpy broadcasting by summing grad`over any axes that were
    implicitly added or expanded to reach the current shape
    """
    if grad.shape == target_shape:
        return grad

    if target_shape == ():
        return np.sum(grad).reshape(())

    ndim_diff = grad.ndim - len(target_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)

    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad.reshape(target_shape)
