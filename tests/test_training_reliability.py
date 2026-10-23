import json

import numpy as np
import pytest

from scripts.train import load_problem_data, run_training


def test_training_is_reproducible_for_fixed_seed():
    first = run_training(n_epochs=3, batch_size=64, seed=123)
    second = run_training(n_epochs=3, batch_size=64, seed=123)

    assert first["final_loss"] == pytest.approx(second["final_loss"], abs=1e-12)
    assert first["final_accuracy"] == second["final_accuracy"]
    assert first["achieved_sparsity"] == second["achieved_sparsity"]
    assert [row["loss"] for row in first["history"]] == pytest.approx(
        [row["loss"] for row in second["history"]], abs=1e-12
    )
    assert all(np.isfinite(row["loss"]) and np.isfinite(row["accuracy"]) for row in first["history"])


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, float("nan"), float("inf")])
def test_training_rejects_invalid_learning_rate(learning_rate):
    with pytest.raises(ValueError, match="learning_rate"):
        run_training(n_epochs=1, learning_rate=learning_rate)


def test_training_rejects_non_finite_dataset_features(tmp_path):
    dataset = tmp_path / "invalid.csv"
    np.savetxt(
        dataset,
        np.array([[0.0, 1.0, 0.0], [float("nan"), -1.0, 1.0], [1.0, 0.5, 0.0]]),
        delimiter=",",
    )

    with pytest.raises(ValueError, match="finite"):
        load_problem_data(dataset)


def test_training_rejects_single_class_dataset(tmp_path):
    dataset = tmp_path / "single_class.csv"
    np.savetxt(
        dataset,
        np.array([[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.5, 0.5, 0.0]]),
        delimiter=",",
    )

    with pytest.raises(ValueError, match="two classes"):
        load_problem_data(dataset)
