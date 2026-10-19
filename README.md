# Self-Pruning Neural Network from Scratch

A NumPy-based educational project that implements a small neural-network training stack, including a custom reverse-mode automatic differentiation engine, an Adam optimizer, and masking-based weight pruning.

The project compares magnitude pruning with Taylor-style saliency pruning and records held-out validation results across multiple random seeds. It is designed to make gradient flow, masks, optimizer state, and the accuracy–sparsity trade-off explicit.

> **Efficiency note:** The current MLP uses dense NumPy matrix multiplication. The reported active-weight ratio is a structural measure, not a measured inference speedup or FLOP reduction.

## Features

- Custom NumPy-backed tensor and reverse-mode autodiff engine.
- Arithmetic, matrix multiplication, reductions, activation functions, and stable softmax cross-entropy.
- MLP and Adam optimizer implemented in the project.
- Gradual, mask-based weight pruning with magnitude and Taylor-saliency criteria.
- Explicit masked-weight gradient handling and Adam moment clearing.
- Unit tests for tensor operations, gradients, optimizer behavior, and pruning invariants.
- Multi-seed held-out validation benchmark with raw CSV results and a Pareto-style plot.

## Requirements

- Python 3.10 or newer
- Git
- Dependencies from `requirements.txt`

The neural-network engine, autodiff, optimizer, and pruning logic are implemented with NumPy rather than PyTorch, TensorFlow, JAX, Keras, or an existing autodiff library. Matplotlib is used for plotting, and pytest is used for tests. Scikit-learn is listed as a dependency but the current default dataset loader reads/generates data using NumPy.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/ANKAN257/self-pruning-network.git
cd self-pruning-network
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, use the environment's Python directly with `.venv\Scripts\python.exe`.

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

From the repository root:

```bash
python scripts/train.py
```

The script trains the MLP and writes training-set metrics and epoch history to `results/training_metrics.json`. These are training metrics, not held-out validation results.

### 5. Run the pruning benchmark

```bash
python scripts/pruning_analysis.py
```

The benchmark evaluates magnitude and saliency pruning over target sparsities of 0%, 50%, 75%, 90%, and 95%, using five seeds and a held-out validation split. It regenerates:

- `results/pareto_results.csv` — raw per-seed results.
- `results/pareto_curve.png` — mean validation accuracy and standard deviation versus achieved sparsity.

The benchmark prints mean and standard deviation of validation accuracy and achieved sparsity. Results can vary if the code, dataset, dependency versions, or random seeds change.

### 6. Run the tests

```bash
python -m pytest -q
```

For verbose test names:

```bash
python -m pytest -v
```

## Project structure

```text
self-pruning-network/
├── engine/                     # Tensor engine, MLP, optimizer, data, pruning, evaluation
├── scripts/
│   ├── train.py                # Training pipeline
│   └── pruning_analysis.py     # Multi-seed pruning comparison
├── tests/                      # Automated regression and correctness tests
├── data/
│   └── synthetic_dataset.csv   # Default synthetic dataset (created if missing)
├── results/
│   ├── training_metrics.json
│   ├── pareto_results.csv
│   └── pareto_curve.png
├── DESIGN.md                   # Technical design and validation notes
├── README.md
└── requirements.txt
```

The default dataset contains two input features and a class label. If `data/synthetic_dataset.csv` is missing, the loader generates it. The scripts should be run from the repository root.

## How pruning works

Two scoring criteria are compared:

- **Magnitude:** \(|w|\)
- **Taylor-style saliency:** \(|w \cdot \nabla_w L|\)

The benchmark gradually increases the requested sparsity during each trial. It reports both target sparsity and achieved sparsity; achieved sparsity is the appropriate value to use when comparing results.

The optimizer uses each parameter's explicit mask. Inactive weights are kept at exact zero, their gradients are suppressed, and Adam's first- and second-moment state for inactive entries is cleared. The current policy is **prune-only**: connection regrowth/reactivation is not implemented.

## Interpreting the results

- Validation accuracy is measured on a held-out split, separately from the training metrics written by `train.py`.
- The CSV records results for individual seeds; the plot summarizes mean accuracy and standard deviation.
- The active-weight ratio is calculated as:

  ```text
  active_weight_ratio = active_weight_parameters / total_prunable_weight_parameters
  ```

- The active-weight ratio indicates how many prunable weights remain active. It does **not** establish lower latency, lower FLOPs, or lower memory use in execution.
- The current forward path uses dense NumPy matrix multiplication, so zero-valued weights are not automatically skipped by a sparse kernel.
- Very aggressive sparsity can reduce accuracy. Use the CSV to assess the trade-off rather than assuming every target preserves quality.

## Design documentation

See [DESIGN.md](DESIGN.md) for details on the autodiff graph, broadcasting-aware gradients, masked-weight policy, saliency criterion, pruning schedule, validation protocol, efficiency limitations, and multi-tenant serving considerations.

## Reproducing and checking changes

Run the full validation workflow from the repository root:

```bash
python -m pytest -q
python scripts/train.py
python scripts/pruning_analysis.py
git status
```

The scripts regenerate files under `results/`. If you are developing locally, review any changes before committing:

```bash
git diff -- results/
```

## Current limitations

- No sparse-aware matrix multiplication or measured latency benchmark is implemented.
- The pruning policy does not support regrowth.
- The training script reports training-set metrics; use the pruning analysis script for held-out validation comparisons.
- Benchmark results describe this dataset and setup and should not be generalized to larger models without further evaluation.
