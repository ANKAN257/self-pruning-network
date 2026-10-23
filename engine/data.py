from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_dataset.csv"


def generate_synthetic_dataset(
    n_samples: int = 600,
    n_features: int = 2,
    seed: int = 7,
    path: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, n_features))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0.1).astype(np.int64)

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack([features, labels])
        np.savetxt(output_path, data, delimiter=",", fmt="%.8f")

    return features, labels


def _parse_csv_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.size == 0:
        raise ValueError(f"Dataset file is empty: {path}")

    if data.shape[1] < 2:
        raise ValueError(f"Dataset must contain features and a label column: {path}")

    features = data[:, :-1]
    raw_labels = data[:, -1]
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Dataset must contain only finite numeric values: {path}")
    if not np.all(raw_labels == np.floor(raw_labels)):
        raise ValueError(f"Labels must be integer class indices: {path}")
    labels = raw_labels.astype(np.int64)
    return features, labels


def load_dataset(
    path: Optional[str | Path] = None,
    *,
    seed: int = 7,
    generate_if_missing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH

    if dataset_path.exists():
        return _parse_csv_dataset(dataset_path)

    if not generate_if_missing:
        raise FileNotFoundError(f"Dataset file was not found: {dataset_path}")

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    features, labels = generate_synthetic_dataset(path=dataset_path, seed=seed)
    return features, labels
