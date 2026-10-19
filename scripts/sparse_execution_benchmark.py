"""Measure dense versus zero-skipping sparse-aware MLP inference.

This is a microbenchmark, not a promise of speedup. The reference sparse path
uses Python loops and can be slower than dense BLAS, especially for small layers.
Run from the repository root with: python scripts/sparse_execution_benchmark.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import MLP, load_dataset, magnitude_score, prune_parameter

RESULTS_PATH = ROOT / "results" / "sparse_execution_benchmark.csv"
TARGET_SPARSITY = 0.90
BATCH_SIZE = 1024
WARMUP_RUNS = 3
TIMED_RUNS = 15


def median_ms(function, warmups=WARMUP_RUNS, repeats=TIMED_RUNS):
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def main():
    x, labels = load_dataset()
    rng = np.random.default_rng(2026)
    model = MLP([x.shape[1], 16, int(labels.max()) + 1], rng=rng)
    for layer in model.layers:
        prune_parameter(layer.weight, magnitude_score(layer.weight.data), TARGET_SPARSITY)

    repeats = int(np.ceil(BATCH_SIZE / len(x)))
    batch = np.tile(x, (repeats, 1))[:BATCH_SIZE].astype(np.float64, copy=False)
    dense_output = model.forward_dense_array(batch)
    sparse_output = model.forward_sparse_array(batch)
    max_abs_error = float(np.max(np.abs(dense_output - sparse_output)))
    if not np.allclose(dense_output, sparse_output, atol=1e-11, rtol=1e-11):
        raise AssertionError(f"Sparse output differs from dense output (max abs error {max_abs_error})")

    dense_ms = median_ms(lambda: model.forward_dense_array(batch))
    sparse_ms = median_ms(lambda: model.forward_sparse_array(batch))
    total_weights = sum(layer.weight.data.size for layer in model.layers)
    active_weights = sum(np.count_nonzero(layer.weight.data) for layer in model.layers)
    dense_macs = BATCH_SIZE * total_weights
    sparse_macs = BATCH_SIZE * active_weights
    row = {
        "target_sparsity": TARGET_SPARSITY,
        "achieved_sparsity": 1.0 - active_weights / max(total_weights, 1),
        "batch_size": BATCH_SIZE,
        "dense_median_ms": dense_ms,
        "sparse_aware_median_ms": sparse_ms,
        "measured_speedup_dense_over_sparse": dense_ms / sparse_ms if sparse_ms else float("nan"),
        "dense_weight_multiply_count_proxy": dense_macs,
        "sparse_active_weight_multiply_count_proxy": sparse_macs,
        "max_absolute_output_error": max_abs_error,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "note": "Multiply-count proxy excludes bias/activation; timings are environment-specific and sparse path may be slower.",
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Achieved sparsity: {row['achieved_sparsity']:.3f}")
    print(f"Dense median: {dense_ms:.3f} ms")
    print(f"Sparse-aware median: {sparse_ms:.3f} ms")
    print(f"Dense/sparse-aware speed ratio: {row['measured_speedup_dense_over_sparse']:.3f}x")
    print(f"Maximum absolute output error: {max_abs_error:.3e}")
    print("Timing is environment-specific; a ratio below 1.0 means sparse-aware inference was slower.")


if __name__ == "__main__":
    main()
