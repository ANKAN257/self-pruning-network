from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .tensor import Tensor


class Adam:
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params: List[Tensor] = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(param.data, dtype=np.float64) for param in self.params]
        self.v = [np.zeros_like(param.data, dtype=np.float64) for param in self.params]
        self.t = 0

    @staticmethod
    def _active_mask(param: Tensor) -> np.ndarray:
        if getattr(param, "mask", None) is not None:
            mask = np.asarray(param.mask, dtype=np.float64)
            if mask.shape != param.data.shape:
                mask = np.broadcast_to(mask, param.data.shape)
            return mask > 0.0
        return np.asarray(param.data != 0.0, dtype=bool)

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if not param.requires_grad:
                continue
            grad = np.asarray(param.grad, dtype=np.float64)
            active = self._active_mask(param)
            grad = np.where(active, grad, 0.0)
            self.m[i] = np.where(active, self.m[i], 0.0)
            self.v[i] = np.where(active, self.v[i], 0.0)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
