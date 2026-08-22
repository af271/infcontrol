import json
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import TypedDict

from infcontrol.canonicalize import ExprNode

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_FILE = PROJECT_ROOT / "data" / "trace_dataset.json"


class DatasetRecord(TypedDict):
    label: str
    goal_formula: ExprNode
    theory: list[str]
    smallest: list[str]


class Dataset:
    """Read-only view over the serialized MeTTa theorem dataset."""

    def __init__(self, filename: str | PathLike[str] | Path = DATASET_FILE):
        path = Path(filename)
        with path.open(encoding="utf-8") as f:
            self.records: list[DatasetRecord] = json.load(f)

        self.by_label: dict[str, DatasetRecord] = {record["label"]: record for record in self.records}
        self.formula_map: dict[str, ExprNode] = {record["label"]: record["goal_formula"] for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.records)

    def get(self, label: str) -> DatasetRecord:
        return self.by_label[label]

    def has_formula(self, label: str) -> bool:
        return label in self.formula_map

    def formula(self, label: str) -> ExprNode | None:
        return self.formula_map.get(label)

    def theory(self, label: str) -> list[str]:
        return self.by_label[label]["theory"]

    def smallest(self, label: str) -> list[str]:
        return self.by_label[label]["smallest"]

    def goal(self, label: str) -> ExprNode:
        return self.by_label[label]["goal_formula"]


__all__ = ["DATASET_FILE", "Dataset", "DatasetRecord"]
