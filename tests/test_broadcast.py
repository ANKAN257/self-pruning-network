import numpy as np

from engine import Tensor, compute_importance


def test_broadcast_addition_backpropagates_correctly():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), requires_grad=True)
    bias = Tensor(np.array([1.0, -1.0], dtype=np.float64), requires_grad=True)
    out = (x + bias).sum()
    out.backward()

    assert x.grad.shape == x.data.shape
    assert bias.grad.shape == bias.data.shape
    assert np.allclose(x.grad, np.ones_like(x.data))
    assert np.allclose(bias.grad, np.array([2.0, 2.0], dtype=np.float64))

    importance = compute_importance(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                                   np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
    assert importance.shape == (2, 2)
    assert np.all(importance >= 0.0)
