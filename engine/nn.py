from __future__ import annotations

import numpy as np

from .tensor import Tensor


def _validate_inference_input(x: np.ndarray, in_features: int) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != in_features:
        raise ValueError(f"Expected a 2D input with {in_features} features; got {array.shape}")
    return array


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng(0)
        limit = np.sqrt(2.0 / max(1, in_features))
        self.weight = Tensor(rng.normal(0.0, limit, size=(in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((out_features,), dtype=np.float64), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def forward_dense_array(self, x: np.ndarray) -> np.ndarray:
        """Dense NumPy inference path; does not construct an autodiff graph."""
        x = _validate_inference_input(x, self.weight.data.shape[0])
        return x @ self.weight.data + self.bias.data

    def forward_sparse_array(self, x: np.ndarray) -> np.ndarray:
        """Reference sparse-aware inference path that skips exact-zero connections.

        This implementation favors transparency and correctness over speed. It
        iterates over output features and multiplies only by nonzero weights.
        For small or moderately sparse matrices, Python/NumPy overhead can make
        it slower than dense matrix multiplication.
        """
        x = _validate_inference_input(x, self.weight.data.shape[0])
        weights = self.weight.data
        result = np.empty((x.shape[0], weights.shape[1]), dtype=np.float64)
        for output_index in range(weights.shape[1]):
            active = np.flatnonzero(weights[:, output_index] != 0.0)
            if active.size:
                result[:, output_index] = x[:, active] @ weights[active, output_index]
            else:
                result[:, output_index] = 0.0
        result += self.bias.data
        return result


class MLP:
    def __init__(self, layer_sizes, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(Linear(in_size, out_size, rng))

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.weight)
            params.append(layer.bias)
        return params

    def forward(self, x: Tensor):
        out = x
        for layer in self.layers[:-1]:
            out = layer.forward(out).relu()
        if self.layers:
            out = self.layers[-1].forward(out)
        return out

    def forward_dense_array(self, x: np.ndarray) -> np.ndarray:
        """Dense array-only inference path for fair timing against sparse inference."""
        out = np.asarray(x, dtype=np.float64)
        for layer in self.layers[:-1]:
            out = np.maximum(layer.forward_dense_array(out), 0.0)
        if self.layers:
            out = self.layers[-1].forward_dense_array(out)
        return out

    def forward_sparse_array(self, x: np.ndarray) -> np.ndarray:
        """Sparse-aware array-only inference path that skips exact-zero weights."""
        out = np.asarray(x, dtype=np.float64)
        if out.ndim != 2:
            raise ValueError(f"Expected a 2D input array; got shape {out.shape}")
        for layer in self.layers[:-1]:
            out = np.maximum(layer.forward_sparse_array(out), 0.0)
        if self.layers:
            out = self.layers[-1].forward_sparse_array(out)
        return out

    def __call__(self, x: Tensor):
        return self.forward(x)
