from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng(0)
        limit = np.sqrt(2.0 / max(1, in_features))
        self.weight = Tensor(rng.normal(0.0, limit, size=(in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((out_features,), dtype=np.float64), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


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

    def __call__(self, x: Tensor):
        return self.forward(x)
