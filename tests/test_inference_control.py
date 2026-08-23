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

    output_dir = tmp_path / "graphs"
    controller = InferenceControl(csv_path, output_dir=output_dir, max_popped=1000)
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
    assert len(controller.saved_graphs) == len(accepted)
    assert all(path.exists() for path in controller.saved_graphs)
