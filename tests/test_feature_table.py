import csv
import json
from pathlib import Path

from infcontrol.feature_table import write_feature_table


def _write_dataset(path: Path) -> None:
    records = [
        {
            "label": "th1",
            "goal_formula": ["→", "θ", "ψ"],
            "theory": ["th2", "ax1"],
            "smallest": ["th2"],
        },
        {
            "label": "th2",
            "goal_formula": ["→", "ψ", "θ"],
            "theory": ["ax2"],
            "smallest": [],
        },
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


def test_write_feature_table(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    output_path = tmp_path / "features.csv"
    _write_dataset(dataset_path)

    written = write_feature_table(output_path=output_path, dataset_path=dataset_path)
    assert written == output_path

    with output_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert list(rows[0].keys()) == ["goal", "candidate", "used", "candidate_known"]

    known = rows[0]
    unknown = rows[1]

    assert known["candidate_known"] == "True"
    assert known["used"] == "True"
    assert known["candidate"] != ""

    assert unknown["candidate_known"] == "False"
    assert unknown["candidate"] == ""
