from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .tensor import Tensor


class Adam:
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params: List[Tensor] = list(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.m = [np.zeros_like(param.data, dtype=np.float64) for param in self.params]
        self.v = [np.zeros_like(param.data, dtype=np.float64) for param in self.params]
        self.t = 0

    @staticmethod
    def _active_mask(param: Tensor) -> np.ndarray:
        """Use an explicit pruning mask; otherwise every value is trainable."""
        mask = getattr(param, "mask", None)
        if mask is None:
            # Zero is a valid parameter value (especially for biases).
            return np.ones_like(param.data, dtype=bool)
        mask = np.asarray(mask)
        if mask.shape != param.data.shape:
            try:
                mask = np.broadcast_to(mask, param.data.shape)
            except ValueError as exc:
                raise ValueError(
                    f"Parameter mask shape {mask.shape} cannot broadcast to {param.data.shape}"
                ) from exc
        return mask > 0.0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if not param.requires_grad:
                continue
            active = self._active_mask(param)
            grad = np.where(active, np.asarray(param.grad, dtype=np.float64), 0.0)
            self.m[i] = np.where(active, self.m[i], 0.0)
            self.v[i] = np.where(active, self.v[i], 0.0)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            updated = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.data = np.where(active, updated, 0.0)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
