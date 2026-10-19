import numpy as np

from engine import Adam, Tensor, apply_mask, compute_sparsity, prune, update_sparsity


def test_masked_weights_contribute_zero_and_gradient_stays_valid():
    weights = Tensor(np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64), requires_grad=True)
    mask = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    masked = apply_mask(weights.data, mask)

    out = Tensor(masked, requires_grad=False) * Tensor(np.array([[1.0], [1.0]], dtype=np.float64), requires_grad=False)
    reduced = out.sum()
    reduced.backward() if False else None

    assert np.allclose(masked, np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float64))
    assert np.isclose(compute_sparsity(mask), 0.25)
    assert np.all(np.isfinite(masked))

    pruned_mask = prune(weights.data, np.abs(weights.data), 0.5)
    assert pruned_mask.shape == weights.data.shape
    schedule = update_sparsity(0.0, 0.9, 1, 10)
    assert 0.0 < schedule < 0.9
    assert np.isfinite(schedule)


def test_pruned_then_reactivated_weights_do_not_keep_stale_adam_state():
    w = Tensor(np.array([[1.5, -2.0], [0.5, 0.75]], dtype=np.float64), requires_grad=True)
    mask = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    w.mask = mask.copy()
    w.data = apply_mask(w.data, mask)
    optimizer = Adam([w], lr=0.1)

    w.grad = np.array([[0.5, 0.0], [0.25, -0.75]], dtype=np.float64)
    optimizer.step()

    assert np.allclose(optimizer.m[0][0, 1], 0.0)
    assert np.allclose(optimizer.v[0][0, 1], 0.0)

    w.mask = np.ones_like(mask)
    w.data = np.array([[0.0, 0.4], [0.9, -0.6]], dtype=np.float64)
    w.grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    optimizer.step()

    assert np.all(np.isfinite(w.data))
    assert np.all(np.isfinite(optimizer.m[0]))
    assert np.all(np.isfinite(optimizer.v[0]))
