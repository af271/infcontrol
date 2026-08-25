import csv
import json

import numpy as np

from infcontrol import InferenceControl


def test_inference_control_runs_wunderbaum_features(tmp_path):
    csv_path = tmp_path / "features.csv"
    rows = []
    for index in range(10):
        goal = ["->", f"g{index}", "b"]
        rows.extend(
            [
                {"goal": goal, "candidate": goal, "used": "True"},
                {
                    "goal": goal,
                    "candidate": ["->", f"c{index}", "d"],
                    "used": "False",
                },
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

    controller = InferenceControl(csv_path, max_popped=1000)
    assert controller.targets[-1].name == "one"
    assert controller.targets[-1].value == 1
    split_groups = [
        {str(controller.features[index, 0]) for index in indices}
        for indices in (
            controller.fitter.train_indices,
            controller.fitter.validation_indices,
            controller.fitter.test_indices,
        )
    ]
    assert not split_groups[0] & split_groups[1]
    assert not split_groups[0] & split_groups[2]
    assert not split_groups[1] & split_groups[2]

    accepted = list(controller.run())

    assert accepted
    assert len(controller.fitter.coefficients) == len(accepted)
    assert controller.fitter.log_loss() < np.log(2.0)


def test_inference_control_predicts_equal_length_formula_arrays(tmp_path):
    csv_path = tmp_path / "features.csv"
    rows = []
    for index in range(10):
        goal = ["->", f"g{index}", "b"]
        rows.extend(
            [
                {"goal": goal, "candidate": goal, "used": "True"},
                {
                    "goal": goal,
                    "candidate": ["->", f"c{index}", "d"],
                    "used": "False",
                },
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

    controller = InferenceControl(csv_path, max_popped=1000)
    list(controller.run())
    probabilities = controller.predict(
        controller.features[:, 0], controller.features[:, 1]
    )

    assert probabilities.shape == (len(rows),)
    np.testing.assert_allclose(probabilities, controller.fitter.probabilities)
