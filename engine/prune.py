from __future__ import annotations

import numpy as np


def compute_sparsity(mask: np.ndarray) -> float:
    mask = np.asarray(mask)
    return 0.0 if mask.size == 0 else float(np.mean(mask == 0.0))


def compute_importance(weights: np.ndarray, grads: np.ndarray | None = None) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if grads is None:
        return np.abs(weights)
    grads = np.asarray(grads, dtype=np.float64)
    if grads.shape != weights.shape:
        raise ValueError(f"Gradient shape {grads.shape} must match weight shape {weights.shape}")
    return np.abs(weights * grads)


def magnitude_score(weights: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(weights, dtype=np.float64))


def saliency_score(weights: np.ndarray, grads: np.ndarray) -> np.ndarray:
    return compute_importance(weights, grads)


def apply_mask(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(mask)
    if mask.shape != weights.shape:
        try:
            mask = np.broadcast_to(mask, weights.shape)
        except ValueError as exc:
            raise ValueError(f"Mask shape {mask.shape} cannot broadcast to weights {weights.shape}") from exc
    return np.where(mask > 0.0, weights, 0.0)


def _validate_target(target_sparsity: float) -> float:
    target = float(target_sparsity)
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_sparsity must be between 0.0 and 1.0")
    return target


def _exact_mask_from_scores(score: np.ndarray, target_sparsity: float) -> np.ndarray:
    """Keep floor((1-target)*N) entries; stable sorting makes ties deterministic."""
    score = np.asarray(score, dtype=np.float64)
    target = _validate_target(target_sparsity)
    if score.size == 0:
        return np.ones_like(score, dtype=np.float64)
    if not np.all(np.isfinite(score)):
        raise ValueError("Pruning scores must all be finite")
    flat = score.ravel()
    keep_count = int(np.floor((1.0 - target) * flat.size + 1e-12))
    mask = np.zeros(flat.size, dtype=np.float64)
    if keep_count:
        order = np.argsort(-flat, kind="stable")
        mask[order[:keep_count]] = 1.0
    return mask.reshape(score.shape)


def prune(weights: np.ndarray, score: np.ndarray, target_sparsity: float) -> np.ndarray:
    """Return weights with the lowest-scoring target fraction set to exact zero."""
    weights = np.asarray(weights, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if score.shape != weights.shape:
        raise ValueError(f"Score shape {score.shape} must match weight shape {weights.shape}")
    return apply_mask(weights, _exact_mask_from_scores(score, target_sparsity))


def prune_parameter(param, score: np.ndarray, target_sparsity: float) -> np.ndarray:
    """Apply monotonic prune-only masking and clear gradients for masked entries."""
    score = np.asarray(score, dtype=np.float64)
    if score.shape != param.data.shape:
        raise ValueError(f"Score shape {score.shape} must match parameter shape {param.data.shape}")
    target = _validate_target(target_sparsity)
    old_mask = getattr(param, "mask", None)
    if old_mask is None:
        active = np.ones_like(param.data, dtype=bool)
    else:
        old_mask = np.asarray(old_mask)
        if old_mask.shape != param.data.shape:
            raise ValueError("Existing parameter mask must have the same shape as the parameter")
        active = old_mask > 0.0

    total = param.data.size
    desired_active = int(np.floor((1.0 - target) * total + 1e-12))
    keep_count = min(int(np.count_nonzero(active)), desired_active)
    new_mask = np.zeros(total, dtype=np.float64)
    active_indices = np.flatnonzero(active.ravel())
    if keep_count:
        active_scores = score.ravel()[active_indices]
        if not np.all(np.isfinite(active_scores)):
            raise ValueError("Pruning scores for active parameters must be finite")
        order = np.argsort(-active_scores, kind="stable")
        new_mask[active_indices[order[:keep_count]]] = 1.0
    new_mask = new_mask.reshape(param.data.shape)

    param.mask = new_mask
    param.data = apply_mask(param.data, new_mask)
    if param.grad is not None:
        param.grad = np.where(new_mask > 0.0, np.asarray(param.grad, dtype=np.float64), 0.0)
    return new_mask


def update_sparsity(current_sparsity: float, target_sparsity: float, step: int, total_steps: int) -> float:
    current = _validate_target(current_sparsity)
    target = _validate_target(target_sparsity)
    total_steps = max(1, int(total_steps))
    progress = min(1.0, max(0.0, step / total_steps))
    return float(current + (target - current) * progress)


class PruningScheduler:
    def __init__(self, start: float = 0.0, end: float = 0.9, steps: int = 100):
        self.start = _validate_target(start)
        self.end = _validate_target(end)
        self.steps = max(1, int(steps))

    def target(self, step: int) -> float:
        fraction = min(1.0, max(0.0, step / self.steps))
        return self.start + (self.end - self.start) * (fraction ** 1.5)


class MaskManager:
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def make_mask(self, score: np.ndarray, target_sparsity: float) -> np.ndarray:
        return _exact_mask_from_scores(score, target_sparsity)

    def apply(self, weights: np.ndarray, score: np.ndarray, target_sparsity: float) -> np.ndarray:
        return apply_mask(weights, self.make_mask(score, target_sparsity))
