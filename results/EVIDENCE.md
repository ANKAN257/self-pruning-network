# Part 4 — Evidence and falsifiable claim

This report documents the current pruning experiments and inference benchmark.

## Reproduce the artifacts

```bash
python scripts/pruning_analysis.py
python scripts/sparse_execution_benchmark.py
python -m pytest -q
```

Generated artifacts:
- `pareto_curve.png`: mean held-out validation accuracy versus achieved weight sparsity, with standard-deviation error bars.
- `pareto_results.csv`: per-seed results across five seeds, five target sparsities, and two pruning criteria.
- `sparse_execution_benchmark.csv`: output equivalence, timing, and structural weight-multiply counts.

## Comparison at approximately 90% target sparsity

All five seeds achieved 90.625% weight sparsity, leaving 6 active weights out of 64 prunable weights.

| Seed | Magnitude validation accuracy | Saliency validation accuracy | Difference |
|---:|---:|---:|---:|
| 7 | 95.00% | 95.00% | +0.00 pp |
| 17 | 89.17% | 97.50% | +8.33 pp |
| 27 | 79.17% | 99.17% | +20.00 pp |
| 37 | 98.33% | 98.33% | +0.00 pp |
| 47 | 53.33% | 97.50% | +44.17 pp |
| **Mean** | **83.00%** | **97.50%** | **+14.50 pp** |

The paired sample standard deviation of the per-seed differences is 18.50 percentage points. Saliency matched magnitude on two seeds and outperformed it on three.

**Interpretation:** these results describe one synthetic dataset and five fixed seeds. They do not establish universal superiority or statistical significance.

## Falsifiable claim

> On the current synthetic dataset and fixed five-seed protocol (7, 17, 27, 37, 47), at 90.625% achieved sparsity, mean held-out accuracy is 97.50% for Taylor-style saliency pruning and 83.00% for magnitude pruning, a paired mean difference of +14.50 percentage points.

The claim is falsified if rerunning the same protocol with the same code, data, dependencies, and seeds does not reproduce these values within reasonable numerical tolerance.

## Inference cost measurement

The latest `sparse_execution_benchmark.csv` records:

- Achieved sparsity: 90.625%.
- Structural weight-multiply-count proxy: 65,536 dense versus 6,144 active-weight multiplies for batch size 1,024.
- Maximum absolute output error: 0.0.
- Median dense latency: 0.1052 ms.
- Median sparse-aware latency: 0.3505 ms.
- Dense/sparse-aware speed ratio: 0.300x; sparse-aware inference was approximately 3.33x slower in this environment.

The sparse-aware implementation skips exact-zero connections, but Python loops and indexing overhead outweighed the reduced multiply count. This count is a structural proxy, not a hardware FLOP measurement. Timing is environment-specific. **Do not claim an inference speedup.**

## Division-by-zero edge case

Tensor division raises `ZeroDivisionError` before constructing a computation-graph node when the denominator contains an exact zero. Regression tests cover scalar zero, zero values in denominator tensors, reverse division, and gradients with a broadcast denominator.