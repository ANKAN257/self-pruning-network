import numpy as np
import pytest

from engine import Tensor, softmax, softmax_cross_entropy


def assert_finite_difference_matches_autodiff(function, values, *, eps=1e-6, atol=2e-5, rtol=2e-4):
    """Compare reverse-mode gradients with central finite differences."""
    leaves = [Tensor(np.array(value, dtype=np.float64, copy=True), requires_grad=True) for value in values]
    output = function(*leaves)
    if output.data.size != 1:
        output = output.sum()
    output.backward()

    for tensor_index, value in enumerate(values):
        value = np.array(value, dtype=np.float64, copy=True)
        numerical = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            plus = [np.array(v, dtype=np.float64, copy=True) for v in values]
            minus = [np.array(v, dtype=np.float64, copy=True) for v in values]
            plus[tensor_index][index] += eps
            minus[tensor_index][index] -= eps
            plus_out = function(*[Tensor(v) for v in plus]).data
            minus_out = function(*[Tensor(v) for v in minus]).data
            numerical[index] = (float(np.sum(plus_out)) - float(np.sum(minus_out))) / (2.0 * eps)
        assert np.allclose(leaves[tensor_index].grad, numerical, atol=atol, rtol=rtol), (
            f"gradient mismatch for input {tensor_index}:\n"
            f"analytic={leaves[tensor_index].grad}\nnumeric={numerical}"
        )


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_broadcast_binary_operation_gradients_match_finite_difference(operation):
    x = np.array([[0.8], [1.4]])
    y = np.array([[1.2, 2.0, 2.8]])
    def function(a, b):
        result = {"add": lambda: a + b, "sub": lambda: a - b,
                  "mul": lambda: a * b, "div": lambda: a / b}[operation]()
        return (result ** 2).sum()
    assert_finite_difference_matches_autodiff(function, [x, y])


def test_matmul_gradients_match_finite_difference():
    x = np.array([[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]])
    w = np.array([[0.5, -0.3], [0.8, 0.6], [-0.7, 0.9]])
    assert_finite_difference_matches_autodiff(lambda a, b: ((a @ b) ** 2).mean(), [x, w])


def test_power_gradient_matches_finite_difference():
    x = np.array([0.4, 0.9, 1.7])
    assert_finite_difference_matches_autodiff(lambda a: (a ** 3.2).sum(), [x])


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu"])
def test_activation_gradients_match_finite_difference(activation):
    # Avoid ReLU's nondifferentiable point at zero.
    x = np.array([-1.3, -0.6, 0.25, 0.8, 1.4])
    upstream = np.array([0.7, -0.3, 1.2, 0.5, -0.9])
    def function(a):
        return (getattr(a, activation)() * upstream).sum()
    assert_finite_difference_matches_autodiff(function, [x], atol=5e-5, rtol=5e-4)


@pytest.mark.parametrize("operation", ["sum", "mean"])
@pytest.mark.parametrize("axis", [(0, 2), (2, 0), (-1, 0), (1,)])
def test_multi_axis_reduction_gradients_match_finite_difference(operation, axis):
    x = np.arange(1.0, 25.0).reshape(2, 3, 4) / 7.0
    weights = np.arange(1.0, 4.0) if len(axis) == 2 and set(a % 3 for a in axis) == {0, 2} else None
    def function(a):
        reduced = getattr(a, operation)(axis=axis)
        return (reduced ** 2).sum()
    assert_finite_difference_matches_autodiff(function, [x])


def test_softmax_gradient_matches_finite_difference():
    x = np.array([[0.2, -0.5, 0.9], [1.1, 0.4, -0.8]])
    upstream = np.array([[0.3, -0.7, 0.5], [0.2, 0.8, -0.4]])
    assert_finite_difference_matches_autodiff(lambda a: (softmax(a) * upstream).sum(), [x])


def test_softmax_cross_entropy_logits_and_target_gradients_match_finite_difference():
    logits = np.array([[0.4, -0.2, 1.0], [-0.7, 0.8, 0.1]])
    targets = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    assert_finite_difference_matches_autodiff(
        lambda a, b: softmax_cross_entropy(a, b), [logits, targets], atol=3e-5, rtol=3e-4
    )
