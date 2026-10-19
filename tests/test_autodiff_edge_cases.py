import numpy as np
import pytest

from engine import Tensor


def _finite_difference_check(function, values, *, eps=1e-6, atol=3e-5, rtol=3e-4):
    leaves = [
        Tensor(np.array(value, dtype=np.float64, copy=True), requires_grad=True)
        for value in values
    ]
    output = function(*leaves)
    if output.data.size != 1:
        output = output.sum()
    output.backward()

    for input_index, value in enumerate(values):
        value = np.asarray(value, dtype=np.float64)
        numerical = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            plus = [np.array(v, dtype=np.float64, copy=True) for v in values]
            minus = [np.array(v, dtype=np.float64, copy=True) for v in values]
            plus[input_index][index] += eps
            minus[input_index][index] -= eps
            plus_loss = np.sum(function(*[Tensor(v) for v in plus]).data)
            minus_loss = np.sum(function(*[Tensor(v) for v in minus]).data)
            numerical[index] = (plus_loss - minus_loss) / (2.0 * eps)

        assert np.allclose(leaves[input_index].grad, numerical, atol=atol, rtol=rtol), (
            f"matmul gradient mismatch for input {input_index}: "
            f"analytic={leaves[input_index].grad}, numerical={numerical}"
        )


@pytest.mark.parametrize(
    "left,right,function",
    [
        (
            np.array([0.2, -0.4, 0.7]),
            np.array([0.5, 0.8, -0.3]),
            lambda a, b: a @ b,
        ),
        (
            np.array([0.2, -0.4, 0.7]),
            np.array([[0.5, -0.3], [0.8, 0.6], [-0.7, 0.9]]),
            lambda a, b: ((a @ b) ** 2).sum(),
        ),
        (
            np.array([[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]]),
            np.array([0.5, 0.8, -0.3]),
            lambda a, b: ((a @ b) ** 2).sum(),
        ),
    ],
)
def test_vector_and_matrix_matmul_gradients_match_finite_difference(left, right, function):
    _finite_difference_check(function, [left, right])


def test_batched_matmul_with_broadcasted_right_operand_matches_finite_difference():
    rng = np.random.default_rng(14)
    left = rng.normal(size=(2, 3, 4))
    right = rng.normal(size=(4, 5))
    _finite_difference_check(lambda a, b: ((a @ b) ** 2).mean(), [left, right])


def test_batched_matmul_reduces_broadcasted_left_operand_gradient():
    rng = np.random.default_rng(15)
    left = rng.normal(size=(1, 3, 4))
    right = rng.normal(size=(2, 4, 5))
    _finite_difference_check(lambda a, b: ((a @ b) ** 2).mean(), [left, right])


def test_repeated_backward_resets_intermediate_grads_but_accumulates_leaf_grads():
    x = Tensor(np.array([2.0]), requires_grad=True)
    squared = x * x
    loss = (squared * squared).sum()

    loss.backward()
    first_grad = x.grad.copy()
    loss.backward()

    assert np.allclose(first_grad, np.array([32.0]))
    assert np.allclose(x.grad, np.array([64.0]))
