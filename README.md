# Self-Pruning Neural Network from Scratch

A NumPy-only project for building a custom neural network training stack, including a reverse-mode autodiff engine and a dynamic self-pruning mechanism that removes unnecessary connections during optimization.

The goal is to train a classifier from first principles while learning a sparse architecture that improves computational efficiency without sacrificing too much accuracy. Unlike post-training pruning, this project focuses on evolving sparsity throughout training in a way that keeps the computation graph valid, optimizer state consistent, and gradients meaningful.

---

## Project Overview

This repository is designed around four core stages:

1. Build a lightweight tensor and autodiff engine
2. Train a multilayer perceptron using only NumPy
3. Add a masking-based pruning system that gradually removes low-value connections
4. Evaluate the tradeoff between sparsity, cost, and accuracy

The final system should demonstrate how a model can become more efficient while preserving predictive performance under targeted sparsity levels such as 50%, 75%, 90%, and 95%.

---

## Why This Project Matters

Many modern pruning methods rely on frameworks that hide the underlying mechanics of gradient flow and sparsity. This project makes those mechanics explicit:

- Pruned weights must contribute exactly zero in forward pass
- Masking must be reflected correctly in backward propagation
- Adam momentum states must remain valid after pruning or reactivation
- Sparse training should produce measurable gains in compute and memory usage

This creates a clean, educational implementation of dynamic sparsification rooted in numerical reasoning rather than framework abstraction.

---

## System Architecture

The overall flow of the project is:

Dataset
  -> Neural Network
  -> Autodiff Engine
  -> Backward Pass
  -> Adam Optimizer
  -> Mask Manager
  -> Pruning Scheduler
  -> Importance Evaluator

At a high level, the pipeline is:

- Data is loaded and prepared for mini-batch training
- Forward pass computes predictions through a custom MLP
- Reverse-mode autodiff accumulates gradients for each operation
- Adam updates parameters using current and historical gradient statistics
- A pruning manager gradually masks low-importance weights
- The model is evaluated at multiple sparsity targets

---

## Six Major Functional Modules

This project is organized around the six core modules that the challenge expects.

### 1. Tensor and Autodiff Engine

Purpose: build a custom reverse-mode automatic differentiation system from scratch.

Core API:
- Tensor()
- backward()
- add, sub, mul, div, matmul, sum, mean
- relu, sigmoid, tanh, gelu
- softmax, cross_entropy, softmax_cross_entropy

Responsibilities:
- Create computation graphs
- Track parent tensors
- Accumulate gradients safely
- Support broadcasting and shaped reductions
- Run reverse-mode backpropagation correctly

### 2. Neural Network Module

Purpose: train a learnable classifier using the custom autodiff engine.

Core API:
- Linear()
- MLP()
- forward()
- parameters()

Responsibilities:
- Layer construction
- Weight initialization
- Forward propagation
- Parameter collection for training

### 3. Loss Function Module

Purpose: convert predictions into a differentiable objective for optimization.

Core API:
- softmax()
- cross_entropy()
- softmax_cross_entropy()

Responsibilities:
- Produce stable class probabilities
- Compute classification loss
- Generate meaningful gradients for backpropagation

### 4. Optimizer Module

Purpose: update model weights using the computed gradients.

Core API:
- Adam.step()
- Adam.zero_grad()

Responsibilities:
- Maintain momentum and variance estimates
- Correct the bias terms
- Update parameters in a numerically stable way

### 5. Self-Pruning Module

Purpose: remove low-importance connections during training while preserving valid optimization.

Core API:
- compute_importance()
- prune()
- apply_mask()
- compute_sparsity()
- update_sparsity()

Responsibilities:
- Calculate connection importance
- Zero pruned weights in the forward pass
- Maintain mask consistency
- Increase sparsity gradually across epochs

### 6. Evaluation and Analysis Module

Purpose: prove the pruning process preserves performance and reduces cost.

Core API:
- accuracy()
- active_parameters()
- effective_sparsity()
- run_trial()
- main benchmark sweep in scripts/pruning_analysis.py

Responsibilities:
- Measure accuracy and loss
- Calculate active parameter count and sparsity
- Produce the Pareto curve for sparsity vs accuracy
- Compare cost reduction against predictive quality

---

## Required Challenge Tests

The project includes the validation checks expected by the problem statement:

- tests/test_gradcheck.py for finite-difference gradient verification
- tests/test_broadcast.py for broadcasting backpropagation
- tests/test_masked_weights.py for masked-weight correctness and optimizer safety

These tests cover selected numerical and pruning invariants. Run the full suite before relying on results; the tests are not a formal proof of every operation.

---

## Core Requirements

### 1. Reverse-Mode Autodiff Engine

Implement a custom automatic differentiation system with:

- NumPy-backed tensor abstraction
- Computation graph tracking
- Gradient storage per tensor
- Reverse-mode differentiation
- Support for the following operations:
  - Addition
  - Subtraction
  - Multiplication
  - Division
  - Matrix multiplication
  - Sum
  - Mean
- Activation functions:
  - ReLU
  - Sigmoid
  - Tanh
  - GELU
- Loss function:
  - Numerically stable softmax cross-entropy
- Backpropagation behavior:
  - Topological graph traversal
  - Gradient accumulation
  - Broadcasting-aware gradients
  - Correct handling of masked parameters
- Validation:
  - Finite-difference gradient checking
  - Operator-level unit tests

### 2. Neural Network Training

Train a classifier using only the custom autodiff engine and NumPy.

Required components:

- Linear layers
- Activation layers
- Multi-layer perceptron (MLP)
- Adam optimizer implemented from scratch
- Mini-batch training
- Stable convergence with a fixed random seed
- No NaN propagation

Expected outputs:

- Loss curve over training epochs
- Accuracy curve over training epochs

### 3. Self-Pruning Mechanism

Design a dynamic pruning mechanism that removes low-value connections during training instead of only at the end.

Key ideas:

- Importance metric per connection can be based on:
  - Weight magnitude
  - Gradient sensitivity
  - Taylor saliency
- Pruning schedule should increase sparsity gradually through stages, such as:
  - 0% -> 50% -> 75% -> 90%
- Masking must be honest and explicit:
  - Pruned connections contribute zero to forward propagation
  - They are excluded from effective computation
  - Their gradients are handled explicitly
- Optimizer correctness must be preserved:
  - Adam first moments remain valid
  - Adam second moments remain valid
  - Future updates continue to behave properly after pruning or reactivation

### 4. Evaluation and Tradeoff Analysis

Evaluate the sparse model under several sparsity targets:

- 0%
- 50%
- 75%
- 90%
- 95%

The project should compare:

- Accuracy vs sparsity
- Active parameter count or FLOP reduction
- Sparse inference cost or speedup
- Plain pruning baseline vs importance-based pruning

The final result should be a Pareto-style tradeoff analysis that shows whether higher sparsity can be achieved with acceptable accuracy loss.

---

## Constraints

This project intentionally avoids the use of:

- PyTorch
- TensorFlow
- JAX
- Keras
- Hugging Face
- Existing autodiff libraries

Allowed tools:

- NumPy
- Matplotlib
- Scikit-learn for dataset loading only

---

## Initial Phase Essentials

At the beginning of this project, the critical requirement is to build the core technical foundation correctly, not to finalize a folder layout. The minimum essential items to define and implement in phase 1 are:

- A custom NumPy-backed tensor abstraction
- Reverse-mode autodiff with graph tracking and gradient accumulation
- Core differentiable operations and activation functions
- A numerically stable softmax cross-entropy loss
- A simple MLP training loop with mini-batches
- An Adam optimizer implemented from scratch
- A masking mechanism for pruned weights that preserves correct gradients
- A pruning scheduler and importance metric
- Evaluation scripts to measure sparsity, accuracy, and compute cost
- Reproducible experiments with fixed random seeds

This is the actual project backbone needed to solve the challenge. Once the core logic is stable, the repository structure can be organized cleanly around those components.

---

## Quick Start for a Remote Clone

Anyone can run this project after cloning the repository from GitHub.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd self-pruning-network
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the training pipeline

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

This writes:

- `results/pareto_results.csv` — raw sweep results
- `results/pareto_curve.png` — accuracy vs sparsity Pareto plot

### Honest cost metric

The project measures cost as the fraction of active parameters remaining after pruning:

```text
active_ratio = active_parameters / total_parameters
```

This is a structural active-weight ratio, not a measured runtime metric. The current implementation still runs dense NumPy matrix multiplication, so it does not prove an inference speedup.

### Falsifiable claim

The repository supports a claim such as:

Use the regenerated multi-seed validation CSV to evaluate a falsifiable claim. For example: “At a specified achieved weight sparsity, saliency pruning improves mean held-out accuracy over magnitude pruning by at least X percentage points across the five fixed seeds.” Fill in X only after the new benchmark has been run; do not treat the existing committed results as validation evidence.

---

## Summary

This project combines the fundamentals of neural-network training, automatic differentiation, and model sparsification into a single, from-scratch educational system. The key focus is not only to prune a model, but to do so in a mathematically correct, computationally honest, and practically useful way.

The final deliverable is a working framework and analysis pipeline that demonstrates how self-pruning can reduce model cost while preserving accuracy under realistic sparsity targets.


## Updated validation workflow

After pulling a correctness update, run:

```bash
python -m pytest -q
python scripts/train.py
python scripts/pruning_analysis.py
```

The analysis script writes raw per-seed validation results and a mean/standard-deviation Pareto plot. It reports achieved weight sparsity separately from the target. The active-weight ratio remains a structural proxy until a sparse-aware forward implementation and measured latency comparison are added.
