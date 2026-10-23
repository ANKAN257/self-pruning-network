"""Training entry point for the self-pruning network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from engine import (
    Adam, DEFAULT_DATASET_PATH, MLP, MaskManager, PruningScheduler, Tensor,
    load_dataset, magnitude_score, prune_parameter, softmax_cross_entropy,
)

RESULTS_PATH = ROOT / "results" / "training_metrics.json"


def load_problem_data(dataset_path: str | Path | None = None):
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    x, labels = load_dataset(path)
    if x.ndim != 2 or labels.ndim != 1 or len(x) != len(labels):
        raise ValueError("Features must be 2D and labels must be a matching 1D array")
    if len(labels) == 0 or labels.min() < 0:
        raise ValueError("Labels must be a non-empty array of non-negative class indices")
    if not np.all(np.isfinite(x)):
        raise ValueError("Features must contain only finite values")
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
        raise ValueError("Labels must be finite integer class indices")
    if len(np.unique(labels)) < 2:
        raise ValueError("Training requires at least two classes")
    targets = np.eye(int(labels.max()) + 1)[labels]
    return x, labels, targets


def apply_pruning_schedule(model, target_sparsity: float, mask_manager: MaskManager) -> None:
    """Apply explicit, monotonic masks to weight matrices (biases are excluded)."""
    for param in model.parameters():
        if param.data.ndim < 2:
            continue
        prune_parameter(param, magnitude_score(param.data), target_sparsity)


def _weight_sparsity(model) -> tuple[float, int, int]:
    total = active = 0
    for param in model.parameters():
        if param.data.ndim < 2:
            continue
        total += param.data.size
        mask = getattr(param, "mask", None)
        active += int(np.count_nonzero(param.data if mask is None else mask))
    return (0.0, 0, 0) if total == 0 else (1.0 - active / total, active, total)


def run_training(
    n_epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    seed: int = 7,
    dataset_path: str | Path | None = None,
):
    if n_epochs < 1 or batch_size < 1:
        raise ValueError("n_epochs and batch_size must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    rng = np.random.default_rng(seed)
    x, labels, targets = load_problem_data(dataset_path)
    model = MLP([x.shape[1], 16, int(labels.max()) + 1], rng=rng)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = PruningScheduler(start=0.0, end=0.8, steps=n_epochs)
    mask_manager = MaskManager()
    history = []

    for epoch in range(n_epochs):
        permutation = rng.permutation(len(x))
        epoch_loss = 0.0
        for start in range(0, len(x), batch_size):
            indices = permutation[start:start + batch_size]
            xb = Tensor(x[indices], requires_grad=False)
            yb = Tensor(targets[indices], requires_grad=False)
            optimizer.zero_grad()
            loss = softmax_cross_entropy(model.forward(xb), yb)
            if not np.all(np.isfinite(loss.data)):
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            if any(not np.all(np.isfinite(param.data)) for param in model.parameters()):
                raise FloatingPointError(f"Non-finite model parameter encountered at epoch {epoch + 1}")
            epoch_loss += float(loss.data) * len(indices)

        target_sparsity = scheduler.target(epoch + 1)
        apply_pruning_schedule(model, target_sparsity, mask_manager)
        logits = model.forward(Tensor(x, requires_grad=False))
        accuracy = float(np.mean(np.argmax(logits.data, axis=1) == labels))
        achieved, active, total = _weight_sparsity(model)
        mean_loss = epoch_loss / len(x)
        if not np.isfinite(mean_loss) or not np.isfinite(accuracy):
            raise FloatingPointError(f"Non-finite training metric encountered at epoch {epoch + 1}")
        history.append({
            "epoch": epoch + 1,
            "loss": mean_loss,
            "accuracy": accuracy,
            "target_sparsity": target_sparsity,
            "achieved_sparsity": achieved,
            "active_weights": active,
            "total_weights": total,
        })

    final_logits = model.forward(Tensor(x, requires_grad=False))
    final_accuracy = float(np.mean(np.argmax(final_logits.data, axis=1) == labels))
    final_loss = float(softmax_cross_entropy(final_logits, Tensor(targets, requires_grad=False)).data)
    if not np.isfinite(final_loss) or not np.isfinite(final_accuracy):
        raise FloatingPointError("Non-finite final training metric encountered")
    achieved, active, total = _weight_sparsity(model)
    results = {
        "seed": seed, "n_epochs": n_epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "final_accuracy": final_accuracy,
        "final_loss": final_loss, "target_sparsity": scheduler.target(n_epochs),
        "achieved_sparsity": achieved, "active_weights": active,
        "total_weights": total, "history": history,
        "evaluation_note": "Training-set metrics only; pruning_analysis.py reports held-out validation metrics.",
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    results = run_training()
    print("Training pipeline complete.")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")
    print(f"Target weight sparsity: {results['target_sparsity']:.2f}")
    print(f"Achieved weight sparsity: {results['achieved_sparsity']:.2f}")
    print(f"Metrics saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
