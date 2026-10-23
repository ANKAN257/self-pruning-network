# Self-Pruning Neural Network from Scratch

A NumPy-based educational project implementing a custom reverse-mode autodiff engine, Adam optimizer, and mask-based neural-network pruning. It compares magnitude pruning with Taylor-style saliency on held-out data across multiple random seeds.

The project includes a reference sparse-aware inference path that skips exact-zero weights and a microbenchmark comparing it with dense array inference. This path is designed to make the trade-off measurable, not to promise a speedup: Python-loop overhead can outweigh saved multiplies for small layers.

## Features

- Custom NumPy-backed tensor and reverse-mode autodiff engine.
- Arithmetic, matrix multiplication, reductions, activations, and stable softmax cross-entropy.
- MLP and Adam optimizer implemented from scratch.
- Gradual mask-based pruning with magnitude and Taylor-saliency criteria.
- Explicit handling of masked gradients and Adam moments.
- Finite-difference gradient checks for supported operations and broadcasting/reduction cases.
- Multi-seed held-out validation sweep with raw CSV and Pareto-style plot.
- Dense versus sparse-aware inference equivalence tests and measured timing benchmark.
- GitHub Actions workflow that runs pytest on pushes and pull requests.

## Requirements

- Python 3.10 or newer
- Git
- Dependencies listed in `requirements.txt`

The neural-network engine, autodiff, optimizer, and pruning logic use NumPy rather than PyTorch, TensorFlow, JAX, Keras, or an existing autodiff library. Matplotlib is used for plots; pytest runs tests. Scikit-learn is listed as a dependency, although the default data loader uses NumPy.

## Quick start

### 1. Clone and enter the repository

```bash
git clone https://github.com/ANKAN257/self-pruning-network.git
cd self-pruning-network
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run commands using `.venv\Scripts\python.exe` instead.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run training

```bash
python scripts/train.py
```

This trains the model and saves metrics to the `results` folder.

### 5. Run evaluation

```bash
python scripts/evaluate.py
```

This prints a summary of the saved training results.

### 6. Run the test suite

```bash
python -m pytest -q
```

---

## Setup

1. Create a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify NumPy and Matplotlib are available
4. Start implementing the autodiff system and training pipeline

---

## Development Workflow

The expected progression is:

1. Implement tensor nodes and reverse-mode autodiff
2. Add basic arithmetic and activation support
3. Build a simple MLP and softmax cross-entropy loss
4. Train on a small classification dataset
5. Add Adam optimizer and mini-batch training
6. Implement masking and pruning policy
7. Evaluate sparsity settings and generate curves

---

## Expected Outcomes

By the end of the project, the implementation should show:

- A working custom autodiff engine
- A stable training loop for a small classifier
- A dynamic pruning strategy that removes low-importance weights over time
- Honest computational cost reporting for sparse models
- A clear accuracy-vs-sparsity tradeoff analysis

---

## Suggested Evaluation Metrics

At minimum, report:

- Train and validation loss
- Train and validation accuracy
- Active parameter count
- Effective sparsity percentage
- Relative compute reduction
- Comparison of magnitude pruning vs importance-based pruning

---

## Challenge Evidence and Benchmarking

This repository follows the case-study requirements for the self-pruning network challenge:

- The autodiff engine, optimizer, pruning logic, and training loop are implemented from scratch in NumPy.
- The current CSV/synthetic dataset loader uses NumPy; scikit-learn is not used by the model or autodiff engine.
- The project includes a benchmark sweep over sparsity targets to produce a Pareto-style accuracy-vs-sparsity curve.
- The benchmark reports active-weight ratio as a structural proxy only. The current NumPy forward path is dense, so this is not a measured latency or FLOP speedup.
- The masking logic uses explicit masks, exact zeros, zero gradients for inactive entries, and clears Adam moments while entries are inactive. The current policy is prune-only; regrowth is not implemented.

### Benchmark command

```bash
python scripts/pruning_analysis.py
```

Compares magnitude and saliency pruning at target sparsities of 0%, 50%, 75%, 90%, and 95%, across five seeds with a held-out validation split. It writes:

- `results/pareto_results.csv` — raw per-seed results.
- `results/pareto_curve.png` — mean validation accuracy and standard deviation versus achieved sparsity.

### 6. Measure sparse-aware inference

```bash
python scripts/sparse_execution_benchmark.py
```

The benchmark creates a seeded MLP, prunes its weight matrices to approximately 90% sparsity, checks sparse-aware outputs against dense outputs, then reports median timings and a multiply-count proxy for a batch of 1,024 inputs. It writes `results/sparse_execution_benchmark.csv`.

The benchmark is intentionally honest about limitations: it uses a small NumPy model and a transparent output-feature loop, so the sparse path may be slower than dense BLAS. Timings are machine- and environment-specific. The multiply-count proxy counts weight connections only and excludes bias/activation work; it is not a FLOP profiler.

### 7. Run tests

```bash
python -m pytest -q
```

Use `python -m pytest -v` for verbose output. The tests include numerical finite-difference checks and dense/sparse inference equivalence. GitHub Actions also runs the test suite on pushes and pull requests.

## Project structure

```text
self-pruning-network/
├── engine/                         # Tensor engine, MLP, optimizer, data, pruning
├── scripts/
│   ├── train.py                    # Training pipeline
│   ├── pruning_analysis.py         # Multi-seed pruning comparison
│   └── sparse_execution_benchmark.py
├── tests/                          # Regression, gradient-check, and sparse-path tests
├── data/                           # Default synthetic dataset (created if missing)
├── results/                        # Generated metrics, CSV files, and plot
├── .github/workflows/tests.yml     # CI test workflow
├── DESIGN.md
├── README.md
└── requirements.txt
```

Run scripts from the repository root. The default synthetic dataset contains two input features and a class label; the loader creates it if it is missing.

## Pruning and optimizer policy

The benchmark compares:

- **Magnitude:** `|w|`
- **Taylor-style saliency:** `|w * grad(L)|`

The benchmark ramps the requested sparsity during training and reports both target and achieved sparsity. Achieved sparsity is the appropriate measure for comparisons.

The optimizer treats an explicit mask as the source of truth. Masked weights stay exactly zero, masked gradients are suppressed, and Adam's first and second moments are cleared for inactive entries. The current policy is prune-only; regrowth/reactivation is not implemented.

## Interpreting results

- Use the held-out validation metrics from `pruning_analysis.py`; `train.py` reports training metrics.
- `pareto_results.csv` contains individual-seed results; the plot summarizes mean accuracy and standard deviation.
- Active-weight ratio is a structural measure, not proof of lower end-to-end latency or memory use.
- `MLP.forward_sparse_array` skips exact-zero connections, while `MLP.forward_dense_array` uses dense matrix multiplication. The standard autodiff/training forward path remains dense.
- The sparse-aware path is a reference implementation, not a highly optimized sparse kernel. Do not claim a speedup unless the benchmark measurements on the target environment demonstrate one.
- Aggressive pruning can reduce accuracy; use the committed/generated raw data to assess the trade-off.

See [DESIGN.md](DESIGN.md) for autodiff, pruning, evaluation, cost-measurement, and multi-tenant serving notes. The committed [Part 4 evidence report](results/EVIDENCE.md) records the five-seed baseline comparison, a falsifiable quantitative claim, and the measured sparse-aware inference trade-off. Tensor division explicitly raises `ZeroDivisionError` when the denominator contains zero; regression coverage is in `tests/test_division_edges.py`.
