import numpy as np

from engine import Tensor, softmax


def test_gradcheck_matches_finite_differences():
    x = Tensor(np.array([1.5, -0.5], dtype=np.float64), requires_grad=True)
    y = (x * x).sum()
    y.backward()

    eps = 1e-6
    numerical = np.empty_like(x.data)
    for i in range(x.data.size):
        xp = x.data.copy()
        xm = x.data.copy()
        xp[i] += eps
        xm[i] -= eps
        numerical[i] = (np.sum(xp * xp) - np.sum(xm * xm)) / (2.0 * eps)

    assert np.allclose(x.grad, numerical, atol=1e-4, rtol=1e-4)

    probs = softmax(Tensor(np.array([1.0, 2.0], dtype=np.float64), requires_grad=False))
    assert probs.data.shape == (2,)
    assert np.isclose(probs.data.sum(), 1.0, atol=1e-8)
