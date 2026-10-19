# Design Notes and Defense Strategy

## 1. Goal

The core challenge is to build a neural network that prunes itself while training, without relying on a framework-level autodiff engine. The finished system must keep the computation graph honest when weights are masked, preserve optimizer correctness under pruning and reactivation, and produce evidence that the sparse model is genuinely cheaper without an unacceptable accuracy drop.

## 2. Autodiff design

The project uses a custom `Tensor` object backed by NumPy arrays. Each tensor stores:

- a floating-point array
- a `requires_grad` flag
- a gradient array
- parent tensors for reverse-mode traversal
- a backward closure for local gradient propagation

The backward pass performs a topological walk from outputs back to inputs and accumulates gradients in-place. This is essential because multiple paths can contribute to the same parameter. The implementation explicitly handles broadcasting by reducing gradient tensors back to the original parameter shape, which is crucial for the masking case and for operations like `(N, D) + (D,)`.

## 3. Masked-weight correctness

The most important bug pattern in pruning implementations is silently corrupting gradient state after a parameter is masked. A quick implementation often keeps the weight at zero but allows momentum or variance to accumulate with stale gradients. That can produce a non-physical optimizer state after the connection is revived.

The optimizer uses an explicit per-parameter mask as the source of truth. Unmasked parameters are fully trainable even when their numeric value is zero (important for zero-initialized biases). Masked entries are forced to exact zero, their gradients are zeroed during backpropagation, and Adam clears their first and second moments on each step. The current policy is monotonic prune-only: it does not support connection regrowth. A future regrowth implementation must explicitly choose a new value and reset optimizer state before marking a connection active.

## 4. Importance criterion

The pruning criterion uses a saliency-style score based on the product of weight magnitude and gradient magnitude:

$$
\text{saliency} = |w \cdot \nabla_w L|
$$

This approximates the first-order change in loss caused by removing a connection. A magnitude-only approach is a weaker baseline because it ignores how sensitive the loss is to each parameter. Using saliency captures both weight strength and sensitivity, which is a more defensible criterion for pruning under a strict compute budget.

## 5. Pruning schedule

The scheduler uses a gradual ramp from 0% to the target sparsity, with a nonlinear curve to keep early training stable and later epochs more aggressive. The schedule is not a one-shot “prune-at-the-end” solution. Instead, it preserves optimization stability while gradually pushing the network toward a sparse structure. This is a more realistic engineering choice for a self-pruning system and matches the real-world pressure to reduce cost during training instead of after the fact.

## 6. Cost metric

The benchmark reports the active-weight ratio:

$
\text{active-weight ratio} = \frac{\text{active weight entries}}{\text{total prunable weight entries}}
$

This is a structural proxy, not a measured latency or FLOP speedup. The current NumPy MLP still uses dense matrix multiplication, so zero-valued entries are still part of dense kernel execution. A real speedup claim requires a sparse-aware forward path and a controlled latency/FLOP measurement against the dense baseline.

## 7. Benchmark strategy

The repository contains a pruning sweep over several target sparsity levels, including 0%, 50%, 75%, 90%, and 95%. For each target, the project records:

- achieved sparsity
- active parameter count
- accuracy
- loss
- work ratio

These results are stored in `results/pareto_results.csv` and visualized in `results/pareto_curve.png`.

## 8. Defense angle

In a live defense, the key explanations are:

1. Why the backward pass must accumulate gradients correctly even when broadcasting occurs.
2. Why masking a parameter is not merely setting it to zero; the gradient and optimizer state must be treated consistently.
3. Why saliency-based pruning is more defensible than magnitude-only pruning.
4. Why the reported cost metric is honest and what it does not claim.
5. Why the project is reproducible and evidence-based rather than relying on hand-wavy claims.

The repo is designed so those points can be defended directly from code and committed results.


## 9. Validation protocol

The benchmark uses a seeded held-out split and multiple independent seeds. Magnitude and saliency use the same split for each seed. Saliency is estimated from the training split rather than only the final minibatch. Report the mean and standard deviation of validation accuracy and achieved weight sparsity; do not substitute the scheduled target for measured sparsity. Regenerate the committed CSV and plot by running `python scripts/pruning_analysis.py` after code changes.

## 10. Serving at multi-tenant scale

Treat each deployed model version and its mask as an immutable, versioned artifact. Build and validate the sparse execution representation offline, then publish it atomically so in-flight requests cannot observe a partially updated mask. Batch requests by compatible model version and shape, enforce per-tenant concurrency/queue quotas, and isolate tenant-specific model state and metrics. Apply admission control against latency and memory budgets; use bounded queues and cancellation for overload. Canary a new artifact, compare error/latency/quality metrics, and roll back by switching the model-version pointer. Never mutate shared weights or masks in place for one tenant while another tenant is serving from the same artifact.
