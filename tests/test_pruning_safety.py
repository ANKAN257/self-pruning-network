import numpy as np
import pytest

from engine import PruningScheduler, Tensor, compute_sparsity, prune_parameter


def make_parameter(values):
    parameter = Tensor(np.asarray(values, dtype=np.float64), requires_grad=True)
    parameter.grad = np.ones_like(parameter.data)
    return parameter


def test_prune_parameter_hits_exact_target_for_first_prune():
    parameter = make_parameter([[0.1, 0.9], [0.2, 0.8]])

    mask = prune_parameter(parameter, np.abs(parameter.data), target_sparsity=0.5)

    assert np.count_nonzero(mask) == 2
    assert compute_sparsity(mask) == pytest.approx(0.5)
    assert np.all(parameter.data[mask == 0.0] == 0.0)


def test_pruning_is_monotonic_and_does_not_regrow_removed_weights():
    parameter = make_parameter([[0.1, 0.9], [0.2, 0.8]])
    first_mask = prune_parameter(parameter, np.abs(parameter.data), 0.5)
    # Give the removed entries artificially large scores; they must stay removed.
    scores = np.array([[100.0, 1.0], [90.0, 2.0]])
    second_mask = prune_parameter(parameter, scores, 0.25)

    assert np.all(second_mask <= first_mask)
    assert np.count_nonzero(second_mask) == 2
    assert np.all(parameter.data[second_mask == 0.0] == 0.0)
    assert np.all(parameter.grad[second_mask == 0.0] == 0.0)


@pytest.mark.parametrize("target", [-0.01, 1.01, float("nan"), float("inf")])
def test_pruning_rejects_invalid_target_sparsity(target):
    parameter = make_parameter([[1.0, 2.0]])

    with pytest.raises(ValueError, match="target_sparsity"):
        prune_parameter(parameter, np.abs(parameter.data), target)


def test_pruning_rejects_non_finite_scores_without_mutating_parameter():
    parameter = make_parameter([[1.0, 2.0]])
    original = parameter.data.copy()

    with pytest.raises(ValueError, match="finite"):
        prune_parameter(parameter, np.array([[1.0, np.nan]]), 0.5)

    np.testing.assert_array_equal(parameter.data, original)
    assert parameter.mask is None


def test_pruning_scheduler_is_bounded_and_monotonic():
    scheduler = PruningScheduler(start=0.1, end=0.9, steps=10)
    values = [scheduler.target(step) for step in range(-2, 13)]

    assert values[0] == pytest.approx(0.1)
    assert values[-1] == pytest.approx(0.9)
    assert all(0.1 <= value <= 0.9 for value in values)
    assert all(left <= right for left, right in zip(values, values[1:]))
