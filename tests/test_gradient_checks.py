import numpy as np
import pytest

from engine import Tensor, softmax, softmax_cross_entropy


def _finite_difference(fn, values, eps=1e-6):
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += eps
        minus[index] -= eps
        result[index] = (fn(plus) - fn(minus)) / (2.0 * eps)
    return result


def _check_unary_gradient(operation, values):
    x = Tensor(np.asarray(values, dtype=np.float64), requires_grad=True)
    operation(x).sum().backward()
    numeric = _finite_difference(
        lambda candidate: float(np.sum(operation(Tensor(candidate)).data)), values
    )
    assert np.allclose(x.grad, numeric, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "operation,values",
    [
        (lambda x: x ** 3, np.array([0.7, 1.4, 2.1])),
        (lambda x: x.relu(), np.array([-1.3, 0.4, 1.7])),
        (lambda x: x.sigmoid(), np.array([-1.2, 0.3, 1.5])),
        (lambda x: x.tanh(), np.array([-1.1, 0.2, 1.4])),
        (lambda x: x.gelu(), np.array([-1.0, 0.25, 1.3])),
    ],
)
def test_unary_operation_gradients_match_finite_differences(operation, values):
    _check_unary_gradient(operation, values)


@pytest.mark.parametrize(
    "operation",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ],
)
def test_elementwise_binary_gradients_match_finite_differences(operation):
    a_values = np.array([[0.7, 1.2], [1.8, 2.4]])
    b_values = np.array([1.3, 2.1])

    a = Tensor(a_values.copy(), requires_grad=True)
    b = Tensor(b_values.copy(), requires_grad=True)
    operation(a, b).sum().backward()

    numeric_a = _finite_difference(
        lambda candidate: float(np.sum(operation(Tensor(candidate), Tensor(b_values)).data)),
        a_values,
    )
    numeric_b = _finite_difference(
        lambda candidate: float(np.sum(operation(Tensor(a_values), Tensor(candidate)).data)),
        b_values,
    )
    assert np.allclose(a.grad, numeric_a, rtol=2e-4, atol=2e-5)
    assert np.allclose(b.grad, numeric_b, rtol=2e-4, atol=2e-5)


def test_matmul_gradients_match_finite_differences():
    x_values = np.array([[0.2, -0.4, 0.8], [1.1, 0.3, -0.5]])
    w_values = np.array([[0.5, -0.7], [0.9, 0.2], [-0.3, 0.6]])

    x = Tensor(x_values.copy(), requires_grad=True)
    w = Tensor(w_values.copy(), requires_grad=True)
    (x @ w).sum().backward()

    numeric_x = _finite_difference(
        lambda candidate: float(np.sum(candidate @ w_values)), x_values
    )
    numeric_w = _finite_difference(
        lambda candidate: float(np.sum(x_values @ candidate)), w_values
    )
    assert np.allclose(x.grad, numeric_x, rtol=2e-4, atol=2e-5)
    assert np.allclose(w.grad, numeric_w, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("operation", ["sum", "mean"])
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, False), (0, True), (1, True)])
def test_reduction_gradients_match_finite_differences(operation, axis, keepdims):
    values = np.arange(1.0, 7.0).reshape(2, 3) / 3.0
    x = Tensor(values.copy(), requires_grad=True)
    getattr(x, operation)(axis=axis, keepdims=keepdims).sum().backward()
    numeric = _finite_difference(
        lambda candidate: float(
            np.sum(getattr(Tensor(candidate), operation)(axis=axis, keepdims=keepdims).data)
        ),
        values,
    )
    assert np.allclose(x.grad, numeric, rtol=2e-4, atol=2e-5)


def test_softmax_gradient_matches_finite_differences():
    values = np.array([[0.2, -0.4, 0.7], [1.0, 0.1, -0.3]])
    upstream = np.array([[0.5, -0.2, 0.8], [-0.1, 0.4, 0.3]])
    x = Tensor(values.copy(), requires_grad=True)
    (softmax(x) * Tensor(upstream)).sum().backward()
    numeric = _finite_difference(
        lambda candidate: float(np.sum(softmax(Tensor(candidate)).data * upstream)),
        values,
    )
    assert np.allclose(x.grad, numeric, rtol=2e-4, atol=2e-5)


def test_softmax_cross_entropy_logits_gradient_matches_finite_differences():
    logits_values = np.array([[0.2, -0.4, 0.7], [1.0, 0.1, -0.3]])
    targets = Tensor(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    logits = Tensor(logits_values.copy(), requires_grad=True)
    softmax_cross_entropy(logits, targets).backward()
    numeric = _finite_difference(
        lambda candidate: float(softmax_cross_entropy(Tensor(candidate), targets).data),
        logits_values,
    )
    assert np.allclose(logits.grad, numeric, rtol=2e-4, atol=2e-5)
