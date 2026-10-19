# Design Notes and Defense Strategy

## 1. Goal

Build a small neural network with a custom reverse-mode autodiff engine and self-pruning weights. The implementation makes gradient accumulation, mask lifecycle, optimizer state, accuracy trade-offs, and the limits of sparse execution explicit.

## 2. Autodiff design

Each NumPy-backed `Tensor` stores data, a `requires_grad` flag, a gradient array, parent tensors, and a backward closure. Backpropagation traverses the graph in reverse topological order and accumulates gradients in place, which is necessary when a parameter influences an output through multiple paths. Broadcasting-aware backward rules reduce gradients to the original operand shape.

The test suite includes central finite-difference checks for arithmetic with broadcasting, matrix multiplication, powers, reductions including multi-axis reductions, activations, softmax, and softmax-cross-entropy. Finite-difference checks are a practical way to catch derivative mistakes that simple expected-value tests may miss.

## 3. Masked-weight correctness

The optimizer uses an explicit parameter mask as the source of truth. A zero-valued parameter remains trainable if it has no mask (important for zero-initialized biases). Inactive weights are forced to exact zero; their gradients and Adam first/second moments are cleared. Pruning is monotonic and prune-only. Regrowth is not implemented; a future regrowth policy must initialize the revived weight and reset its optimizer state before marking it active.

## 4. Importance criterion

The saliency score is:

$$
\text{saliency} = |w \cdot \nabla_w L|
$$

This first-order Taylor-style score estimates how sensitive the loss is to removing a connection. Magnitude pruning, based on `|w|`, is the baseline. The multi-seed benchmark evaluates both criteria using the same seeded train/validation split for each seed.

## 5. Pruning schedule

The scheduler ramps target sparsity nonlinearly from the starting value to the target. The benchmark estimates saliency from the full training split rather than only the final minibatch and reports achieved sparsity rather than substituting the requested target.

## 6. Cost and sparse-aware inference

The active-weight ratio is:

$$
\text{active-weight ratio} = \frac{\text{active weight entries}}{\text{total prunable weight entries}}
$$

This ratio is structural and is not a latency or FLOP speedup claim.

The ordinary autodiff/training forward path uses dense NumPy matrix multiplication. A separate inference-only path, `MLP.forward_sparse_array`, iterates over output features and multiplies only by nonzero weights. It is a correctness-oriented reference implementation; Python and array-indexing overhead can make it slower than dense BLAS. `scripts/sparse_execution_benchmark.py` checks numerical equivalence and measures median wall-clock latency for dense and sparse-aware array paths on the same model and batch. It also reports a weight multiply-count proxy. The proxy excludes bias and activation work and is not a hardware FLOP counter.

Run the benchmark in the target environment before making a performance claim. Record the Python/NumPy version and machine details alongside any final numbers; a microbenchmark result on a small synthetic model does not establish a general speedup.

## 7. Validation protocol

`scripts/pruning_analysis.py` evaluates target sparsities 0%, 50%, 75%, 90%, and 95% across five seeds, with held-out validation data. It saves individual trial rows to `results/pareto_results.csv` and plots mean accuracy with standard deviation in `results/pareto_curve.png`. Regenerate the results after changing code, data, or dependencies.

Run:

```bash
python -m pytest -q
python scripts/train.py
python scripts/pruning_analysis.py
python scripts/sparse_execution_benchmark.py
```

## 8. Multi-tenant serving considerations

Treat each deployed model version and its mask as an immutable artifact. Build and validate sparse execution representations offline and publish them atomically. Batch requests by compatible model version and shape; enforce per-tenant concurrency and queue quotas; isolate tenant-specific state and metrics; and apply admission control against latency and memory budgets. Use canaries and roll back by switching the model-version pointer. Never mutate shared weights or masks in place for one tenant while another tenant is serving from the same artifact.
