import numpy as np
import pytest

from engine import Tensor
from engine.optim import Adam


def test_adam_keeps_pruned_parameters_zero_and_clears_masked_moments():
    param = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    param.mask = np.array([1.0, 0.0, 1.0])
    param.grad = np.array([0.2, 50.0, -0.3])
    optimizer = Adam([param], lr=0.05)

    optimizer.step()

    assert param.data[1] == 0.0
    assert optimizer.m[0][1] == 0.0
    assert optimizer.v[0][1] == 0.0
    assert param.data[0] != 1.0
    assert param.data[2] != 3.0


def test_adam_without_mask_can_update_a_zero_valued_parameter():
    param = Tensor(np.array([0.0, 1.0]), requires_grad=True)
    param.grad = np.array([0.5, 0.5])
    optimizer = Adam([param], lr=0.01)

    optimizer.step()

    assert param.data[0] != 0.0


def test_adam_rejects_non_broadcastable_mask():
    param = Tensor(np.ones((2, 3)), requires_grad=True)
    param.mask = np.ones((4,))
    optimizer = Adam([param])

    with pytest.raises(ValueError, match="cannot broadcast"):
        optimizer.step()
