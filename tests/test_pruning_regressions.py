import numpy as np
import pytest

from engine import Adam, MaskManager, Tensor, compute_sparsity, prune_parameter


def test_adam_updates_zero_initialized_bias_without_explicit_mask():
    bias = Tensor(np.zeros(3, dtype=np.float64), requires_grad=True)
    optimizer = Adam([bias], lr=0.1)
    bias.grad = np.array([1.0, -2.0, 0.5])
    optimizer.step()
    assert np.all(np.abs(bias.data) > 0.0)


@pytest.mark.parametrize(
    ("scores", "target", "expected_sparsity"),
    [
        (np.array([1.0, 1.0, 1.0, 1.0]), 0.0, 0.0),
        (np.array([1.0, 1.0, 1.0, 1.0]), 0.5, 0.5),
        (np.array([1.0, 1.0, 1.0, 1.0]), 0.75, 0.75),
        (np.array([1.0, 1.0, 1.0, 1.0]), 1.0, 1.0),
    ],
)
def test_mask_manager_hits_exact_sparsity_with_tied_scores(scores, target, expected_sparsity):
    mask = MaskManager().make_mask(scores, target)
    assert compute_sparsity(mask) == expected_sparsity


def test_pruning_is_monotonic_and_keeps_masked_entries_zero():
    param = Tensor(np.array([4.0, 3.0, 2.0, 1.0]), requires_grad=True)
    first = prune_parameter(param, np.abs(param.data), 0.5)
    assert compute_sparsity(first) == 0.5
    assert np.all(param.data[first == 0.0] == 0.0)
    second = prune_parameter(param, np.array([100.0, 90.0, 80.0, 70.0]), 0.25)
    assert np.all(second <= first)
    assert compute_sparsity(second) == 0.5
    assert np.all(param.data[second == 0.0] == 0.0)


def test_adam_clears_moments_for_pruned_entries_and_keeps_exact_zero():
    param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    optimizer = Adam([param], lr=0.1)
    param.grad = np.array([1.0, 1.0])
    optimizer.step()
    assert optimizer.m[0][1] != 0.0
    param.mask = np.array([1.0, 0.0])
    param.data[1] = 0.0
    param.grad = np.array([0.5, 100.0])
    optimizer.step()
    assert param.data[1] == 0.0
    assert optimizer.m[0][1] == 0.0
    assert optimizer.v[0][1] == 0.0


def test_pruned_mask_is_used_by_optimizer_even_if_masked_gradient_is_nonzero():
    param = Tensor(np.array([1.0, 0.0]), requires_grad=True)
    param.mask = np.array([1.0, 0.0])
    optimizer = Adam([param], lr=0.1)
    param.grad = np.array([0.0, 10.0])
    optimizer.step()
    assert param.data[1] == 0.0
    assert optimizer.m[0][1] == 0.0
    assert optimizer.v[0][1] == 0.0
