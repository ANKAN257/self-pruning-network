from __future__ import annotations

import numpy as np


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred_labels = np.argmax(predictions, axis=1)
    target_labels = np.argmax(targets, axis=1)
    return float(np.mean(pred_labels == target_labels))


def active_parameters(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def effective_sparsity(mask: np.ndarray) -> float:
    return float(1.0 - (np.count_nonzero(mask) / max(mask.size, 1)))
