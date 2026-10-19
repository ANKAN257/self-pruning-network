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

This project fixes that by zeroing the gradient of pruned entries before the optimizer step and by masking the weights themselves in the forward pass. In practice, a pruned connection is treated as a hard zero for the forward computation and receives no effective gradient contribution while it is inactive. The optimizer state is therefore not allowed to accumulate stale gradients for a weight that is temporarily off. This preserves Adam’s invariants and makes revival stable.

## 4. Importance criterion

The pruning criterion uses a saliency-style score based on the product of weight magnitude and gradient magnitude:

$$
\text{saliency} = |w \cdot \nabla_w L|
$$

This approximates the first-order change in loss caused by removing a connection. A magnitude-only approach is a weaker baseline because it ignores how sensitive the loss is to each parameter. Using saliency captures both weight strength and sensitivity, which is a more defensible criterion for pruning under a strict compute budget.

## 5. Pruning schedule

The scheduler uses a gradual ramp from 0% to the target sparsity, with a nonlinear curve to keep early training stable and later epochs more aggressive. The schedule is not a one-shot “prune-at-the-end” solution. Instead, it preserves optimization stability while gradually pushing the network toward a sparse structure. This is a more realistic engineering choice for a self-pruning system and matches the real-world pressure to reduce cost during training instead of after the fact.

## 6. Cost metric

The project reports cost honestly using active parameter count instead of the dense multiply-by-zero shortcut. The relevant metric is:

$$
\text{work ratio} = \frac{\text{active parameters}}{\text{total parameters}}
$$

This is a genuine sparse-work metric. It reflects real savings in the surviving computation graph and avoids overstating the benefit by claiming the dense forward pass is efficient when zeros are still being multiplied.

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
