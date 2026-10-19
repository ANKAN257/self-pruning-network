import numpy as np
import pytest

from engine import Tensor, softmax_cross_entropy


def test_power_operation_backpropagates():
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    (x ** 3).sum().backward()
    assert np.allclose(x.grad, np.array([12.0, 27.0]))


@pytest.mark.parametrize("operation", ["sum", "mean"])
@pytest.mark.parametrize("axis,keepdims", [(0, True), (1, True), (0, False), (1, False)])
def test_reduction_backward_supports_axis_and_keepdims(operation, axis, keepdims):
    x = Tensor(np.arange(1.0, 7.0).reshape(2, 3), requires_grad=True)
    result = getattr(x, operation)(axis=axis, keepdims=keepdims)
    result.sum().backward()
    expected = np.ones_like(x.data)
    if operation == "mean":
        expected /= x.data.shape[axis]
    assert np.allclose(x.grad, expected)


def test_softmax_cross_entropy_is_stable_for_extreme_logits():
    logits = Tensor(np.array([[1000.0, -1000.0], [-1000.0, 1000.0]]), requires_grad=True)
    targets = Tensor(np.array([[1.0, 0.0], [1.0, 0.0]]))
    loss = softmax_cross_entropy(logits, targets)
    loss.backward()
    assert np.isfinite(loss.data)
    assert np.all(np.isfinite(logits.grad))


def test_softmax_cross_entropy_target_gradient_matches_log_probability():
    logits = Tensor(np.array([[1.0, 2.0, -0.5]]), requires_grad=False)
    targets = Tensor(np.array([[0.2, 0.3, 0.5]]), requires_grad=True)
    loss = softmax_cross_entropy(logits, targets)
    loss.backward()
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    assert np.allclose(targets.grad, -log_probs)
