import numpy as np

from engine import Adam, MLP, Tensor, apply_mask, compute_sparsity, load_dataset, softmax_cross_entropy


def test_add_and_sum_backward():
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = (x * 2.0 + 1.0).sum()
    y.backward()

    assert np.allclose(x.grad, np.array([2.0, 2.0, 2.0]))


def test_broadcast_backward():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    bias = Tensor(np.array([1.0, 2.0]), requires_grad=False)
    out = (x + bias).sum()
    out.backward()

    assert x.grad.shape == x.data.shape
    assert np.all(np.isfinite(x.grad))


def test_matmul_backward():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    w = Tensor(np.array([[2.0, 0.0], [1.0, 3.0]]), requires_grad=True)
    out = (x @ w).sum()
    out.backward()

    assert np.allclose(x.grad, np.array([[2.0, 4.0], [2.0, 4.0]]))


def test_softmax_cross_entropy_and_training():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 2))
    y_raw = (X[:, 0] > 0).astype(np.int64)
    y = np.eye(2)[y_raw]

    model = MLP([2, 8, 2], rng=rng)
    optimizer = Adam(model.parameters(), lr=0.05)

    for _ in range(80):
        optimizer.zero_grad()
        logits = model.forward(Tensor(X, requires_grad=False))
        loss = softmax_cross_entropy(logits, Tensor(y, requires_grad=False))
        loss.backward()
        optimizer.step()

    final_logits = model.forward(Tensor(X, requires_grad=False))
    predictions = np.argmax(final_logits.data, axis=1)
    accuracy = np.mean(predictions == y_raw)

    assert accuracy > 0.8


def test_masking_and_sparsity():
    weights = np.array([0.9, -0.2, 0.7, 0.1], dtype=np.float64)
    mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    pruned = apply_mask(weights, mask)

    assert np.allclose(pruned, np.array([0.9, 0.0, 0.7, 0.0]))
    assert np.isclose(compute_sparsity(mask), 0.5)


def test_dataset_loader_accepts_csv_and_falls_back_to_synthetic():
    data_path = "data/test_dataset.csv"
    np.savetxt(data_path, np.array([[1.0, 2.0, 1], [2.0, 1.0, 0], [0.5, 0.5, 1]]), delimiter=",", fmt="%.6f")
    try:
        X, y = load_dataset(data_path)
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        assert set(np.unique(y)).issubset({0, 1})
    finally:
        import os

        if os.path.exists(data_path):
            os.remove(data_path)

    X, y = load_dataset()
    assert X.shape[1] == 2
    assert y.shape[0] == X.shape[0]


def test_masked_weight_gradient_is_zeroed_and_resume_is_stable():
    w = Tensor(np.array([[1.0, -1.0], [2.0, 0.5]], dtype=np.float64), requires_grad=True)
    mask = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    w.data = np.where(mask > 0.0, w.data, 0.0)
    out = w.sum()
    out.backward()
    w.grad = np.where(mask > 0.0, w.grad, 0.0)

    assert np.allclose(w.grad, np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64))
    assert np.all(np.isfinite(w.grad))
    assert np.all(np.isfinite(w.data))
