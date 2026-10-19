import numpy as np

from engine import MLP, Tensor, prune_parameter, magnitude_score


def test_sparse_aware_forward_matches_dense_forward_after_pruning():
    rng = np.random.default_rng(42)
    model = MLP([7, 12, 5], rng=rng)
    for layer in model.layers:
        prune_parameter(layer.weight, magnitude_score(layer.weight.data), 0.75)
    x = rng.normal(size=(31, 7))

    dense = model.forward_dense_array(x)
    sparse = model.forward_sparse_array(x)
    autodiff = model.forward(Tensor(x)).data

    assert np.allclose(sparse, dense, atol=1e-12, rtol=1e-12)
    assert np.allclose(dense, autodiff, atol=1e-12, rtol=1e-12)


def test_sparse_forward_handles_all_zero_output_column():
    model = MLP([3, 2], rng=np.random.default_rng(3))
    model.layers[0].weight.data[:, 0] = 0.0
    x = np.ones((4, 3))
    output = model.forward_sparse_array(x)
    assert np.allclose(output[:, 0], model.layers[0].bias.data[0])
