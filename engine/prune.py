from __future__ import annotations

import numpy as np


def compute_sparsity(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask == 0.0))


def compute_importance(weights: np.ndarray, grads: np.ndarray | None = None) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if grads is None:
        return np.abs(weights)
    grads = np.asarray(grads, dtype=np.float64)
    return np.abs(weights * grads)


def magnitude_score(weights: np.ndarray) -> np.ndarray:
    return np.abs(weights)


def saliency_score(weights: np.ndarray, grads: np.ndarray) -> np.ndarray:
    return np.abs(weights * grads)


def apply_mask(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0.0, weights, 0.0)


def prune(weights: np.ndarray, score: np.ndarray, target_sparsity: float) -> np.ndarray:
    mask = MaskManager().make_mask(np.asarray(score, dtype=np.float64), float(target_sparsity))
    return np.where(mask > 0.0, np.asarray(weights, dtype=np.float64), 0.0)


def prune_parameter(param, score: np.ndarray, target_sparsity: float) -> np.ndarray:
    """Apply a mask to a parameter tensor while zeroing any masked gradient entries.

    This keeps the forward pass honest and preserves the optimizer invariant that
    pruned entries do not accumulate stale momentum or variance updates after they
    are temporarily removed from the graph.
    """
    mask = MaskManager().make_mask(score, target_sparsity)
    param.mask = np.asarray(mask, dtype=np.float64)
    param.data = apply_mask(np.asarray(param.data, dtype=np.float64), mask)
    if param.grad is not None:
        param.grad = np.where(mask > 0.0, np.asarray(param.grad, dtype=np.float64), 0.0)
    return mask


def update_sparsity(current_sparsity: float, target_sparsity: float, step: int, total_steps: int) -> float:
    total_steps = max(1, int(total_steps))
    progress = min(1.0, max(0.0, step / total_steps))
    return float(current_sparsity + (target_sparsity - current_sparsity) * progress)


class PruningScheduler:
    def __init__(self, start: float = 0.0, end: float = 0.9, steps: int = 100):
        self.start = start
        self.end = end
        self.steps = max(1, steps)

    def target(self, step: int) -> float:
        fraction = min(1.0, max(0.0, step / max(1, self.steps)))
        return self.start + (self.end - self.start) * (fraction ** 1.5)


class MaskManager:
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def make_mask(self, score: np.ndarray, target_sparsity: float) -> np.ndarray:
        flat = score.ravel()
        k = max(1, int(np.ceil(target_sparsity * flat.size)))
        if flat.size == 0:
            return np.ones_like(score, dtype=np.float64)
        cutoff = np.partition(flat, k - 1)[k - 1]
        return (score >= cutoff).astype(np.float64)

    def apply(self, weights: np.ndarray, score: np.ndarray, target_sparsity: float) -> np.ndarray:
        mask = self.make_mask(score, target_sparsity)
        return apply_mask(weights, mask)
