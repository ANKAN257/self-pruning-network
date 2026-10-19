"""Training entry point for the self-pruning network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from engine import Adam, DEFAULT_DATASET_PATH, MLP, MaskManager, PruningScheduler, Tensor, load_dataset, softmax_cross_entropy

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "training_metrics.json"


def load_problem_data(dataset_path: str | Path | None = None):
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    x, labels = load_dataset(path)
    targets = np.eye(int(labels.max()) + 1 if labels.size else 1)[labels]
    return x, labels, targets


def apply_pruning_schedule(model, target_sparsity: float, mask_manager: MaskManager) -> None:
    for param in model.parameters():
        if param.data.ndim < 2:
            continue
        weight = np.asarray(param.data, dtype=np.float64)
        mask = mask_manager.make_mask(np.abs(weight), target_sparsity)
        param.data = np.where(mask > 0.0, weight, 0.0)
        if param.grad is not None:
            param.grad = np.where(mask > 0.0, param.grad, 0.0)


def run_training(
    n_epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    seed: int = 7,
    dataset_path: str | Path | None = None,
):
    rng = np.random.default_rng(seed)
    x, labels, targets = load_problem_data(dataset_path)

    model = MLP([2, 16, 2], rng=rng)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    pruning_scheduler = PruningScheduler(start=0.0, end=0.8, steps=n_epochs)
    mask_manager = MaskManager()

    history = []
    for epoch in range(n_epochs):
        permutation = rng.permutation(len(x))
        epoch_loss = 0.0

        for start in range(0, len(x), batch_size):
            indices = permutation[start : start + batch_size]
            xb = Tensor(x[indices], requires_grad=False)
            yb = Tensor(targets[indices], requires_grad=False)

            optimizer.zero_grad()
            logits = model.forward(xb)
            loss = softmax_cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.data) * len(indices)

        current_sparsity = pruning_scheduler.target(epoch)
        apply_pruning_schedule(model, current_sparsity, mask_manager)

        full_logits = model.forward(Tensor(x, requires_grad=False))
        predictions = np.argmax(full_logits.data, axis=1)
        accuracy = float(np.mean(predictions == labels))
        epoch_loss /= len(x)

        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": accuracy,
                "sparsity": current_sparsity,
            }
        )

    final_logits = model.forward(Tensor(x, requires_grad=False))
    final_predictions = np.argmax(final_logits.data, axis=1)
    final_accuracy = float(np.mean(final_predictions == labels))
    final_loss = float(softmax_cross_entropy(final_logits, Tensor(targets, requires_grad=False)).data)

    results = {
        "seed": seed,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_accuracy": final_accuracy,
        "final_loss": final_loss,
        "final_sparsity": pruning_scheduler.target(n_epochs - 1),
        "history": history,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    results = run_training()
    print("Training pipeline complete.")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")
    print(f"Target sparsity: {results['final_sparsity']:.2f}")
    print(f"Metrics saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
