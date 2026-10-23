import numpy as np
import pytest

from engine.data import load_dataset
from scripts.pruning_analysis import load_problem_data, run_trial, split_data


def test_split_data_is_reproducible_and_preserves_each_class():
    labels = np.array([0] * 20 + [1] * 10 + [2] * 5)
    x = np.column_stack([np.arange(len(labels)), labels]).astype(float)
    targets = np.eye(3)[labels]

    first = split_data(x, labels, targets, seed=41)
    second = split_data(x, labels, targets, seed=41)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)

    x_train, y_train, _, x_val, y_val, _ = first
    assert set(np.unique(y_train)) == {0, 1, 2}
    assert set(np.unique(y_val)) == {0, 1, 2}
    assert len(x_train) + len(x_val) == len(x)
    assert set(map(tuple, x_train)).isdisjoint(set(map(tuple, x_val)))


def test_split_data_rejects_class_with_only_one_sample():
    labels = np.array([0, 0, 0, 1])
    x = np.arange(8, dtype=float).reshape(4, 2)
    targets = np.eye(2)[labels]

    with pytest.raises(ValueError, match="at least two samples"):
        split_data(x, labels, targets, seed=1)


@pytest.mark.parametrize(
    ("target", "epochs", "batch_size", "criterion", "message"),
    [
        (float("nan"), 2, 4, "magnitude", "target_sparsity"),
        (1.1, 2, 4, "magnitude", "target_sparsity"),
        (0.5, 0, 4, "magnitude", "epochs"),
        (0.5, 2, 0, "magnitude", "batch_size"),
        (0.5, 2, 4, "random", "criterion"),
    ],
)
def test_run_trial_rejects_invalid_settings(target, epochs, batch_size, criterion, message):
    with pytest.raises(ValueError, match=message):
        run_trial(target, criterion, epochs=epochs, batch_size=batch_size)


def test_dataset_loader_rejects_non_integer_labels(tmp_path):
    path = tmp_path / "fractional_labels.csv"
    np.savetxt(
        path,
        np.array([[0.0, 1.0, 0.0], [1.0, -1.0, 1.5], [0.5, 0.5, 0.0]]),
        delimiter=",",
    )

    with pytest.raises(ValueError, match="integer class indices"):
        load_dataset(path)


def test_dataset_loader_rejects_non_finite_values(tmp_path):
    path = tmp_path / "non_finite.csv"
    np.savetxt(
        path,
        np.array([[0.0, 1.0, 0.0], [np.nan, -1.0, 1.0], [0.5, 0.5, 0.0]]),
        delimiter=",",
    )

    with pytest.raises(ValueError, match="finite"):
        load_dataset(path)


def test_analysis_data_requires_at_least_two_classes(tmp_path):
    path = tmp_path / "one_class.csv"
    np.savetxt(
        path,
        np.array([[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.0], [0.4, 0.1, 0.0]]),
        delimiter=",",
    )

    with pytest.raises(ValueError, match="at least two classes"):
        load_problem_data(path)
