"""Reproducible held-out benchmark for magnitude and Taylor-saliency pruning.

The active-weight ratio is a structural proxy, not a measured latency speedup:
the current NumPy MLP still uses dense matrix multiplication.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import (
    Adam, MLP, Tensor, load_dataset, magnitude_score, prune_parameter,
    saliency_score, softmax_cross_entropy,
)

RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "pareto_results.csv"
PLOT_PATH = RESULTS_DIR / "pareto_curve.png"
TARGETS = [0.0, 0.5, 0.75, 0.9, 0.95]
SEEDS = [7, 17, 27, 37, 47]
VALIDATION_FRACTION = 0.2


def load_problem_data(dataset_path: str | Path | None = None):
    x, labels = load_dataset(dataset_path)
    if x.ndim != 2 or labels.ndim != 1 or len(x) != len(labels):
        raise ValueError("Expected a 2D feature array and a matching 1D label array")
    if len(labels) < 5:
        raise ValueError("Need at least five samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("Features must contain only finite values")
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
        raise ValueError("Labels must be finite integer class indices")
    labels = labels.astype(np.int64)
    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative class indices")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two classes")
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Class labels must be contiguous integers starting at zero")
    if np.any(counts < 2):
        raise ValueError("Each class needs at least two samples for a train/validation split")
    targets = np.eye(len(classes), dtype=np.float64)[labels]
    return x, labels, targets


def split_data(x, labels, targets, seed: int):
    """Create a deterministic stratified split with each class in both partitions."""
    x = np.asarray(x)
    labels = np.asarray(labels)
    targets = np.asarray(targets)
    if x.ndim != 2 or labels.ndim != 1 or targets.ndim != 2:
        raise ValueError("Expected 2D features/targets and 1D labels")
    if len(x) != len(labels) or len(x) != len(targets):
        raise ValueError("Features, labels, and targets must have matching sample counts")
    if len(labels) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(targets)):
        raise ValueError("Split data must be non-empty and finite")
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
        raise ValueError("Labels must be finite integers")

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) < 2:
            raise ValueError(f"Class {class_id} needs at least two samples for stratification")
        class_indices = rng.permutation(class_indices)
        n_val = min(len(class_indices) - 1, max(1, int(round(len(class_indices) * VALIDATION_FRACTION))))
        val_parts.append(class_indices[:n_val])
        train_parts.append(class_indices[n_val:])

    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    return (
        x[train_idx], labels[train_idx], targets[train_idx],
        x[val_idx], labels[val_idx], targets[val_idx],
    )

def apply_target_mask(model, target_sparsity: float, criterion: str) -> None:
    for param in model.parameters():
        if param.data.ndim < 2:
            continue
        weights = np.asarray(param.data, dtype=np.float64)
        if criterion == "saliency":
            scores = saliency_score(weights, np.asarray(param.grad, dtype=np.float64))
        elif criterion == "magnitude":
            scores = magnitude_score(weights)
        else:
            raise ValueError(f"Unknown pruning criterion: {criterion}")
        prune_parameter(param, scores, target_sparsity)


def evaluate(model, x, labels, targets):
    logits = model.forward(Tensor(x, requires_grad=False))
    accuracy = float(np.mean(np.argmax(logits.data, axis=1) == labels))
    loss = float(softmax_cross_entropy(logits, Tensor(targets, requires_grad=False)).data)
    return accuracy, loss


def weight_counts(model):
    total = active = 0
    for param in model.parameters():
        if param.data.ndim < 2:
            continue
        total += param.data.size
        mask = getattr(param, "mask", None)
        active += int(np.count_nonzero(param.data if mask is None else mask))
    return active, total


def run_trial(
    target_sparsity: float,
    criterion: str,
    seed: int = 7,
    epochs: int = 60,
    batch_size: int = 32,
    dataset_path: str | Path | None = None,
):
    if criterion not in {"magnitude", "saliency"}:
        raise ValueError("criterion must be 'magnitude' or 'saliency'")
    if not np.isfinite(target_sparsity) or not 0.0 <= target_sparsity <= 1.0:
        raise ValueError("target_sparsity must be finite and between 0 and 1")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    rng = np.random.default_rng(seed)
    x, labels, targets = load_problem_data(dataset_path)
    x_train, y_train, t_train, x_val, y_val, t_val = split_data(x, labels, targets, seed)
    model = MLP([x.shape[1], 16, int(labels.max()) + 1], rng=rng)
    optimizer = Adam(model.parameters(), lr=0.05)
    scheduler_steps = max(1, epochs)

    for epoch in range(epochs):
        permutation = rng.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            indices = permutation[start:start + batch_size]
            xb = Tensor(x_train[indices], requires_grad=False)
            yb = Tensor(t_train[indices], requires_grad=False)
            optimizer.zero_grad()
            loss = softmax_cross_entropy(model.forward(xb), yb)
            loss.backward()
            optimizer.step()

        # Estimate saliency from the whole training split, not the last minibatch.
        optimizer.zero_grad()
        full_train_loss = softmax_cross_entropy(
            model.forward(Tensor(x_train, requires_grad=False)),
            Tensor(t_train, requires_grad=False),
        )
        full_train_loss.backward()
        scheduled_target = target_sparsity * ((epoch + 1) / scheduler_steps)
        apply_target_mask(model, min(target_sparsity, scheduled_target), criterion)

    # Recompute validation metrics only after the final mask has been applied.
    val_accuracy, val_loss = evaluate(model, x_val, y_val, t_val)
    active, total = weight_counts(model)
    achieved_sparsity = 1.0 - active / max(total, 1)
    return {
        "seed": seed,
        "criterion": criterion,
        "target_sparsity": float(target_sparsity),
        "achieved_sparsity": float(achieved_sparsity),
        "validation_accuracy": val_accuracy,
        "validation_loss": val_loss,
        "active_weight_parameters": active,
        "total_prunable_weight_parameters": total,
        "active_weight_ratio": active / max(total, 1),
        "cost_metric_note": "Structural active-weight ratio only; dense NumPy matmul means no measured latency speedup.",
    }


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for criterion in sorted({row["criterion"] for row in rows}):
        summary = []
        for target in TARGETS:
            group = [r for r in rows if r["criterion"] == criterion and r["target_sparsity"] == target]
            if group:
                xs = [r["achieved_sparsity"] for r in group]
                ys = [r["validation_accuracy"] for r in group]
                summary.append((float(np.mean(xs)), float(np.mean(ys)), float(np.std(ys))))
        summary.sort(key=lambda item: item[0])
        if summary:
            ax.errorbar(
                [item[0] for item in summary],
                [item[1] for item in summary],
                yerr=[item[2] for item in summary],
                marker="o",
                capsize=3,
                label=f"{criterion} pruning (mean ± std)",
            )
    ax.set_title("Held-out accuracy vs achieved weight sparsity")
    ax.set_xlabel("Achieved weight sparsity")
    ax.set_ylabel("Validation accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    rows = []
    for seed in SEEDS:
        for target in TARGETS:
            for criterion in ("magnitude", "saliency"):
                rows.append(run_trial(target, criterion, seed=seed))
    save_results(rows)
    plot_results(rows)
    print(f"Raw multi-seed validation results saved to: {CSV_PATH}")
    print(f"Mean/std Pareto plot saved to: {PLOT_PATH}")
    print("Active-weight ratio is a structural proxy, not measured latency speedup.")
    for criterion in ("magnitude", "saliency"):
        for target in TARGETS:
            group = [r for r in rows if r["criterion"] == criterion and r["target_sparsity"] == target]
            acc = np.asarray([r["validation_accuracy"] for r in group])
            sparsity = np.asarray([r["achieved_sparsity"] for r in group])
            print(
                f"{criterion:<9} target={target:.2f} achieved={sparsity.mean():.3f} "
                f"val_acc={acc.mean():.4f}±{acc.std():.4f}"
            )


if __name__ == "__main__":
    main()
