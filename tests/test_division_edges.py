import numpy as np
import pytest

from engine import Tensor


def test_tensor_division_backpropagates_for_broadcast_denominator():
    numerator = Tensor(np.array([[2.0, 4.0], [6.0, 8.0]]), requires_grad=True)
    denominator = Tensor(np.array([2.0, 4.0]), requires_grad=True)

    (numerator / denominator).sum().backward()

    assert np.allclose(numerator.grad, np.array([[0.5, 0.25], [0.5, 0.25]]))
    assert np.allclose(denominator.grad, np.array([-2.0, -0.75]))


def test_tensor_division_rejects_scalar_zero_denominator():
    numerator = Tensor(np.array([1.0, 2.0]), requires_grad=True)

    with pytest.raises(ZeroDivisionError, match="denominator contains zero"):
        _ = numerator / 0.0


def test_tensor_division_rejects_zero_inside_tensor_denominator():
    numerator = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    denominator = Tensor(np.array([1.0, 0.0]), requires_grad=True)

    with pytest.raises(ZeroDivisionError, match="denominator contains zero"):
        _ = numerator / denominator


def test_reverse_tensor_division_rejects_zero_tensor_denominator():
    denominator = Tensor(np.array([0.0, 2.0]), requires_grad=True)

    with pytest.raises(ZeroDivisionError, match="denominator contains zero"):
        _ = 1.0 / denominator
