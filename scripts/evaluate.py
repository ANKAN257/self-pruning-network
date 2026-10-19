"""Evaluation entry point for the self-pruning network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "training_metrics.json"


def main() -> None:
    if not RESULTS_PATH.exists():
        print("No training metrics found. Run `python scripts/train.py` first.")
        return

    with RESULTS_PATH.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    final_accuracy = metrics.get("final_accuracy", 0.0)
    final_loss = metrics.get("final_loss", 0.0)
    final_sparsity = metrics.get("final_sparsity", 0.0)
    last_epoch = metrics["history"][-1] if metrics.get("history") else {}

    print("Evaluation summary")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Final sparsity: {final_sparsity:.2f}")
    if last_epoch:
        print(f"Last epoch accuracy: {last_epoch.get('accuracy', 0.0):.4f}")
    print(f"Metrics file: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
