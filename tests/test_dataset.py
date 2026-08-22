import json
from pathlib import Path

from infcontrol.dataset import Dataset


def _write_dataset(path: Path) -> None:
    records = [
        {
            "label": "th1",
            "goal_formula": ["→", "θ", "ψ"],
            "theory": ["ax1", "th2"],
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


def test_dataset_accessors(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)

    ds = Dataset(dataset_path)

    assert len(ds) == 2
    assert ds.get("th1")["label"] == "th1"
    assert ds.formula("th1") == ["→", "θ", "ψ"]
    assert ds.goal("th2") == ["→", "ψ", "θ"]
    assert ds.theory("th1") == ["ax1", "th2"]
    assert ds.smallest("th1") == ["th2"]
    assert ds.has_formula("th2")
    assert not ds.has_formula("ax1")
    assert ds.formula("unknown") is None
