from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


class Tensor:
    def __init__(
        self,
        data: np.ndarray,
        *,
        requires_grad: bool = False,
        _op: Optional[str] = None,
        _parents: Tuple["Tensor", ...] = (),
        _name: Optional[str] = None,
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.mask = None
        self._op = _op
        self._parents = tuple(_parents)
        self._name = _name
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, grad_shape={self.grad.shape}, op={self._op})"

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    @staticmethod
    def _ensure_tensor(other):
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=np.float64), requires_grad=False)

    @staticmethod
    def _sum_to_shape(grad: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Reduce a broadcasted gradient back to the operand's original shape."""
        grad = np.asarray(grad, dtype=np.float64)
        shape = tuple(shape)

        if grad.shape == shape:
            return grad

        # Extra leading dimensions come from broadcasting and must be summed.
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        # Dimensions of size one were broadcast and must retain their axis.
        for axis, dim in enumerate(shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)

        if grad.shape != shape:
            try:
                grad = grad.reshape(shape)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot reduce gradient with shape {grad.shape} to operand shape {shape}"
                ) from exc
        return grad

    def __add__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="add",
            _parents=(self, other),
            _name="add",
        )

        def _backward():
            if self.requires_grad:
                self.grad += self._sum_to_shape(np.asarray(out.grad, dtype=np.float64), self.data.shape)
            if other.requires_grad:
                other.grad += self._sum_to_shape(np.asarray(out.grad, dtype=np.float64), other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="sub",
            _parents=(self, other),
            _name="sub",
        )

        def _backward():
            if self.requires_grad:
                self.grad += self._sum_to_shape(np.asarray(out.grad, dtype=np.float64), self.data.shape)
            if other.requires_grad:
                other.grad += self._sum_to_shape(-np.asarray(out.grad, dtype=np.float64), other.data.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="mul",
            _parents=(self, other),
            _name="mul",
        )

        def _backward():
            if self.requires_grad:
                self.grad += self._sum_to_shape(np.asarray(out.grad, dtype=np.float64) * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += self._sum_to_shape(np.asarray(out.grad, dtype=np.float64) * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        # NumPy would silently produce inf/nan for zero denominators. Raise a
        # deterministic Python exception instead, before creating a bad graph.
        if np.any(other.data == 0.0):
            raise ZeroDivisionError("Tensor division by zero: denominator contains zero")
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="div",
            _parents=(self, other),
            _name="div",
        )

        def _backward():
            if self.requires_grad:
                self.grad += self._sum_to_shape(np.asarray(out.grad, dtype=np.float64) / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += self._sum_to_shape(
                    -np.asarray(out.grad, dtype=np.float64) * self.data / (other.data ** 2),
                    other.data.shape,
                )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other.__truediv__(self)

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="matmul",
            _parents=(self, other),
            _name="matmul",
        )

        def _backward():
            grad = np.asarray(out.grad, dtype=np.float64)
            left = self.data
            right = other.data

            # NumPy matmul supports vector, matrix, and batched matrix cases.
            # Handle vector operands explicitly, then reduce broadcast batch
            # dimensions back to each operand's original shape.
            if self.requires_grad:
                if left.ndim == 1 and right.ndim == 1:
                    grad_left = grad * right
                elif left.ndim == 1:
                    grad_left = np.matmul(
                        np.expand_dims(grad, axis=-2), np.swapaxes(right, -1, -2)
                    ).squeeze(axis=-2)
                elif right.ndim == 1:
                    grad_left = np.expand_dims(grad, axis=-1) * right
                else:
                    grad_left = np.matmul(grad, np.swapaxes(right, -1, -2))
                self.grad += self._sum_to_shape(grad_left, left.shape)

            if other.requires_grad:
                if left.ndim == 1 and right.ndim == 1:
                    grad_right = grad * left
                elif left.ndim == 1:
                    grad_right = np.expand_dims(left, axis=-1) * np.expand_dims(grad, axis=-2)
                elif right.ndim == 1:
                    grad_right = np.sum(
                        left * np.expand_dims(grad, axis=-1), axis=-2
                    )
                else:
                    grad_right = np.matmul(np.swapaxes(left, -1, -2), grad)
                other.grad += self._sum_to_shape(grad_right, right.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1.0)

    def __pow__(self, exponent):
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _op="pow",
            _parents=(self,),
            _name="pow",
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.asarray(out.grad, dtype=np.float64) * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _op="sum",
            _parents=(self,),
            _name="sum",
        )

        def _backward():
            if not self.requires_grad:
                return
            grad = np.asarray(out.grad, dtype=np.float64)
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                normalized_axes = sorted(a if a >= 0 else self.data.ndim + a for a in axes)
                for a in normalized_axes:
                    grad = np.expand_dims(grad, axis=a)
            grad = np.broadcast_to(grad, self.data.shape).copy()
            self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _op="mean",
            _parents=(self,),
            _name="mean",
        )

        def _backward():
            if not self.requires_grad:
                return
            grad = np.asarray(out.grad, dtype=np.float64)
            if axis is None:
                count = self.data.size
            else:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                normalized_axes = tuple(a if a >= 0 else self.data.ndim + a for a in axes)
                count = np.prod([self.data.shape[a] for a in normalized_axes], dtype=np.float64)
                if not keepdims:
                    for a in sorted(normalized_axes):
                        grad = np.expand_dims(grad, axis=a)
            grad = np.broadcast_to(grad / count, self.data.shape).copy()
            self.grad += grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), requires_grad=self.requires_grad, _op="relu", _parents=(self,), _name="relu")

        def _backward():
            if self.requires_grad:
                self.grad += np.asarray(out.grad, dtype=np.float64) * (self.data > 0.0)

        out._backward = _backward
        return out

    def sigmoid(self):
        val = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(val, requires_grad=self.requires_grad, _op="sigmoid", _parents=(self,), _name="sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += np.asarray(out.grad, dtype=np.float64) * val * (1.0 - val)

        out._backward = _backward
        return out

    def tanh(self):
        val = np.tanh(self.data)
        out = Tensor(val, requires_grad=self.requires_grad, _op="tanh", _parents=(self,), _name="tanh")

        def _backward():
            if self.requires_grad:
                self.grad += np.asarray(out.grad, dtype=np.float64) * (1.0 - val ** 2)

        out._backward = _backward
        return out

    def gelu(self):
        x = self.data
        c = 0.7978845608
        val = 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
        out = Tensor(val, requires_grad=self.requires_grad, _op="gelu", _parents=(self,), _name="gelu")

        def _backward():
            if self.requires_grad:
                tanh_term = np.tanh(c * (x + 0.044715 * x ** 3))
                deriv = 0.5 * (1.0 + tanh_term) + 0.5 * x * (1.0 - tanh_term ** 2) * c * (1.0 + 3.0 * 0.044715 * x ** 2)
                self.grad += np.asarray(out.grad, dtype=np.float64) * deriv

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in getattr(node, "_parents", ()):
                build(parent)
            topo.append(node)

        build(self)

        # Leaf gradients intentionally accumulate across backward calls, but
        # intermediate gradients belong to this traversal and must be cleared.
        for node in topo:
            if node is not self and node._parents:
                node.grad = np.zeros_like(node.data, dtype=np.float64)

        self.grad = np.ones_like(self.data, dtype=np.float64)
        for node in reversed(topo):
            if hasattr(node, "_backward"):
                node._backward()
            mask = getattr(node, "mask", None)
            if mask is not None:
                node.grad = np.where(np.asarray(mask) > 0.0, node.grad, 0.0)


def softmax(logits: Tensor) -> Tensor:
    logits_arr = np.asarray(logits.data, dtype=np.float64)
    shifted = logits_arr - np.max(logits_arr, axis=1, keepdims=True) if logits_arr.ndim > 1 else logits_arr - np.max(logits_arr)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True) if logits_arr.ndim > 1 else exp / np.sum(exp)
    out = Tensor(probs, requires_grad=logits.requires_grad, _op="softmax", _parents=(logits,), _name="softmax")

    def _backward():
        if not logits.requires_grad:
            return
        grad = np.asarray(out.grad, dtype=np.float64)
        if logits_arr.ndim == 1:
            jac = probs[:, None] * (np.eye(probs.size) - probs)
            logits.grad += jac @ grad
        else:
            jac = probs[:, :, None] * (np.eye(probs.shape[1])[None, :, :] - probs[:, None, :])
            logits.grad += np.einsum("bi,bij->bj", grad, jac)

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    logits_arr = np.asarray(logits.data, dtype=np.float64)
    targets_arr = np.asarray(targets.data, dtype=np.float64)
    if targets_arr.ndim == 1:
        targets_arr = np.eye(logits_arr.shape[-1])[targets_arr.astype(int)]
    if logits_arr.ndim == 1:
        logits_arr = logits_arr.reshape(1, -1)
        targets_arr = targets_arr.reshape(1, -1)

    shifted = logits_arr - np.max(logits_arr, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    probs = np.exp(log_probs)
    batch_size = logits_arr.shape[0]
    loss = float(-np.sum(targets_arr * log_probs) / batch_size)
    out = Tensor(
        np.asarray(loss, dtype=np.float64),
        requires_grad=logits.requires_grad or targets.requires_grad,
        _op="cross_entropy",
        _parents=(logits, targets),
        _name="cross_entropy",
    )

    def _backward():
        if logits.requires_grad:
            grad = (probs - targets_arr) / batch_size
            logits.grad += grad.reshape(logits.data.shape)
        if targets.requires_grad:
            targets.grad += (-log_probs / batch_size).reshape(targets.data.shape)

    out._backward = _backward
    return out


def softmax_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    original_logits_shape = logits.data.shape
    original_targets_shape = targets.data.shape
    logits_arr = np.asarray(logits.data, dtype=np.float64)
    targets_arr = np.asarray(targets.data, dtype=np.float64)

    if logits_arr.ndim == 1:
        logits_arr = logits_arr.reshape(1, -1)
        targets_arr = targets_arr.reshape(1, -1)

    shifted = logits_arr - np.max(logits_arr, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    probs = np.exp(log_probs)
    per_sample_loss = -np.sum(targets_arr * log_probs, axis=1)
    loss = float(np.mean(per_sample_loss))

    out = Tensor(
        np.asarray(loss, dtype=np.float64),
        requires_grad=logits.requires_grad or targets.requires_grad,
        _op="softmax_cross_entropy",
        _parents=(logits, targets),
        _name="softmax_cross_entropy",
    )

    def _backward():
        if logits.requires_grad:
            logits.grad += ((probs - targets_arr) / logits_arr.shape[0]).reshape(original_logits_shape)
        if targets.requires_grad:
            targets.grad += (-log_probs / logits_arr.shape[0]).reshape(original_targets_shape)

    out._backward = _backward
    return out


def ones_like(array):
    return Tensor(np.ones_like(np.asarray(array, dtype=np.float64)))


def zeros_like(array):
    return Tensor(np.zeros_like(np.asarray(array, dtype=np.float64)))
