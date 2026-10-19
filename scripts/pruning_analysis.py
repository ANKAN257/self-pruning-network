"""Benchmark sparse training quality across target sparsity levels.

This script produces the challenge-style evidence required for the self-pruning
network assignment: a sparsity-accuracy Pareto curve, raw benchmark numbers, and
an honest active-parameter cost metric.
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

from engine import Adam, MLP, Tensor, load_dataset, magnitude_score, prune_parameter, saliency_score, softmax_cross_entropy

RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "pareto_results.csv"
PLOT_PATH = RESULTS_DIR / "pareto_curve.png"
TARGETS = [0.0, 0.5, 0.75, 0.9, 0.95]


def load_problem_data(dataset_path: str | Path | None = None):
    x, labels = load_dataset(dataset_path)
    targets = np.eye(int(labels.max()) + 1 if labels.size else 1)[labels]
    return x, labels, targets


def apply_target_mask(model, target_sparsity: float, criterion: str) -> None:
    for param in model.parameters():
        if param.data.ndim < 2:
            continue
        score = np.asarray(param.data, dtype=np.float64)
        if criterion == "saliency":
            score = saliency_score(np.asarray(param.data, dtype=np.float64), np.asarray(param.grad, dtype=np.float64))
        else:
            score = magnitude_score(np.asarray(param.data, dtype=np.float64))
        prune_parameter(param, score, target_sparsity)


def run_trial(target_sparsity: float, criterion: str, seed: int = 7, epochs: int = 60, batch_size: int = 32):
    rng = np.random.default_rng(seed)
    x, labels, targets = load_problem_data()
    model = MLP([2, 16, 2], rng=rng)
    optimizer = Adam(model.parameters(), lr=0.05)

    for epoch in range(epochs):
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            indices = permutation[start : start + batch_size]
            xb = Tensor(x[indices], requires_grad=False)
            yb = Tensor(targets[indices], requires_grad=False)
            optimizer.zero_grad()
            logits = model.forward(xb)
            loss = softmax_cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        if epoch < epochs - 1:
            apply_target_mask(model, min(0.98, target_sparsity * (epoch + 1) / max(1, epochs)), criterion)

    apply_target_mask(model, target_sparsity, criterion)
    logits = model.forward(Tensor(x, requires_grad=False))
    predictions = np.argmax(logits.data, axis=1)
    accuracy = float(np.mean(predictions == labels))
    loss = float(softmax_cross_entropy(logits, Tensor(targets, requires_grad=False)).data)

    total_params = sum(int(np.prod(param.data.shape)) for param in model.parameters())
    active_params = sum(int(np.count_nonzero(param.data)) for param in model.parameters())
    achieved_sparsity = 1.0 - (active_params / max(total_params, 1))

    return {
        "criterion": criterion,
        "target_sparsity": target_sparsity,
        "achieved_sparsity": achieved_sparsity,
        "accuracy": accuracy,
        "loss": loss,
        "active_parameters": active_params,
        "total_parameters": total_params,
        "work_ratio": active_params / max(total_params, 1),
    }


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "criterion",
        "target_sparsity",
        "achieved_sparsity",
        "accuracy",
        "loss",
        "active_parameters",
        "total_parameters",
        "work_ratio",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for criterion in sorted({row["criterion"] for row in rows}):
        subset = [row for row in rows if row["criterion"] == criterion]
        subset.sort(key=lambda r: r["target_sparsity"])
        ax.plot(
            [row["achieved_sparsity"] for row in subset],
            [row["accuracy"] for row in subset],
            marker="o",
            label=f"{criterion} pruning",
        )
    ax.set_title("Self-pruning network: sparsity vs accuracy")
    ax.set_xlabel("Achieved sparsity")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    rows = []
    for target in TARGETS:
        for criterion in ("magnitude", "saliency"):
            rows.append(run_trial(target, criterion))
    save_results(rows)
    plot_results(rows)

    print(f"Pareto results saved to: {CSV_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")
    for row in rows:
        print(
            f"{row['criterion']:<9} target={row['target_sparsity']:.2f} "
            f"achieved={row['achieved_sparsity']:.3f} acc={row['accuracy']:.4f} "
            f"active={row['active_parameters']}/{row['total_parameters']}"
        )


if __name__ == "__main__":
    main()
