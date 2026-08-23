from types import SimpleNamespace

import numpy as np
import pytest

from infcontrol.inference_control import InferenceControl
from infcontrol.operators import Formula


def test_recall_at_k_ranks_candidates_per_goal() -> None:
    goals = np.array(
        [
            Formula(["goal", "a"]),
            Formula(["goal", "a"]),
            Formula(["goal", "a"]),
            Formula(["goal", "b"]),
            Formula(["goal", "b"]),
        ],
        dtype=object,
    )
    controller = SimpleNamespace(
        features=np.column_stack((goals, goals)),
        labels=np.array([True, False, True, False, True]),
        fitter=SimpleNamespace(
            train_indices=np.arange(5),
            validation_indices=np.arange(5),
            test_indices=np.arange(5),
            probabilities=np.array([0.9, 0.8, 0.1, 0.7, 0.6]),
        ),
    )

    assert InferenceControl.recall_at_k(controller, "train", 1) == pytest.approx(0.25)
    assert InferenceControl.recall_at_k(controller, "train", 5) == pytest.approx(1.0)


def test_print_metrics_contains_recall_table(capsys) -> None:
    controller = SimpleNamespace(
        features=np.empty((2, 2), dtype=object),
        fitter=SimpleNamespace(
            features=[np.array([1.0, 2.0])],
            train_indices=np.array([0, 1]),
            validation_indices=np.array([0, 1]),
            test_indices=np.array([0, 1]),
            probabilities=np.array([0.2, 0.8]),
        ),
        labels=np.array([0.0, 1.0]),
        log_loss=lambda split: 0.5,
        accuracy=lambda split: 0.5,
        recall_at_k=lambda split, k: 0.5,
    )
    controller.features[:, 0] = [Formula(["goal", "a"])] * 2

    InferenceControl.print_metrics(controller)
    output = capsys.readouterr().out
    assert "recall@1" in output
    assert "recall@5" in output
    assert "recall@10" in output
