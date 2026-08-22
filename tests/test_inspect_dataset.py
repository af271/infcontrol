import json
from pathlib import Path

from infcontrol.dataset import Dataset
from infcontrol.inspect_dataset import build_report, dataset_summary, lookup_candidates


def _write_dataset(path: Path) -> None:
    records = [
        {
            "label": "th1",
            "goal_formula": ["→", "θ", "ψ"],
            "theory": ["th2", "th2", "ax1"],
            "smallest": ["th2"],
        },
        {
            "label": "th2",
            "goal_formula": ["→", "ψ", "θ"],
            "theory": ["th1", "ax2"],
            "smallest": ["th1"],
        },
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


def test_lookup_candidates_is_unique(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    ds = Dataset(dataset_path)

    found = lookup_candidates(ds, limit=3)
    labels = [label for label, _ in found]

    assert labels == ["th2", "th1"]


def test_summary_and_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    ds = Dataset(dataset_path)

    summary = dataset_summary(ds)
    assert summary["theorems"] == 2
    assert summary["candidate_pairs"] == 5
    assert summary["positive_labels"] == 2
    assert summary["known_candidates"] == 3

    report = build_report(ds, n_examples=1, lookup_limit=2, seed=7)
    assert "DATASET SUMMARY" in report
    assert "LOOKUP TEST" in report
    assert "th1" in report
    assert "th2" in report
