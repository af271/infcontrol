import csv
import json

import numpy as np

from infcontrol import InferenceControl, PredictionModel


def _write_csv(csv_path, num_goals=10):
    rows = []
    for index in range(num_goals):
        goal = ["->", f"g{index}", "b"]
        rows.extend(
            [
                {"goal": goal, "candidate": goal, "used": "True"},
                {"goal": goal, "candidate": ["->", f"c{index}", "d"], "used": "False"},
            ]
        )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["goal", "candidate", "used"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def test_save_and_load_model_predicts_like_the_trained_controller(tmp_path):
    csv_path = tmp_path / "features.csv"
    _write_csv(csv_path)

    controller = InferenceControl(csv_path, max_popped=1000)
    list(controller.run())
    assert controller.accepted_graphs

    model_dir = controller.save(tmp_path / "models" / "my_model")
    assert (model_dir / "model.json").exists()

    loaded = PredictionModel.load(model_dir)
    assert len(loaded.graphs) == len(controller.accepted_graphs)

    goals = controller.features[:, 0]
    candidates = controller.features[:, 1]
    expected = controller.predict(goals, candidates)
    actual = loaded.predict(goals, candidates)

    np.testing.assert_allclose(actual, expected)


def test_load_model_with_no_accepted_features_returns_prior(tmp_path):
    csv_path = tmp_path / "features.csv"
    _write_csv(csv_path)

    controller = InferenceControl(csv_path, max_popped=1)
    model_dir = controller.save(tmp_path / "models" / "empty_model")

    loaded = PredictionModel.load(model_dir)
    assert loaded.graphs == []

    goals = controller.features[:, 0]
    candidates = controller.features[:, 1]
    probabilities = loaded.predict(goals, candidates)
    assert np.allclose(probabilities, probabilities[0])
