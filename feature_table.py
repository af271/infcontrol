"""Build a candidate-level feature table from the Hyperon theorem dataset.

The exported CSV contains one row per (goal, candidate) pair with columns:

- goal: Formula
- candidate: Formula
- used: bool
- candidate_known: bool
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from infcontrol.dataset import DATASET_FILE, Dataset
from infcontrol.operators import Formula

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "feature_table.csv"


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One candidate-level training row before CSV serialization."""

    goal: Formula
    candidate: Formula | None
    used: bool
    candidate_known: bool


def _node_to_list(node: object) -> object:
    """Convert nested tuples from Formula internals to JSON-compatible lists."""
    if isinstance(node, tuple):
        return [_node_to_list(child) for child in node]
    return node


def _formula_to_json(formula: Formula) -> str:
    """Serialize a Formula into stable JSON for CSV cells."""
    as_list = _node_to_list(formula.canonical_tree)
    return json.dumps(as_list, ensure_ascii=False)


def iter_feature_rows(dataset: Dataset) -> Iterator[FeatureRow]:
    """Yield one FeatureRow for each candidate in each theorem theory list."""
    for record in dataset:
        goal_formula = Formula(record["goal_formula"])
        used_set = set(record["smallest"])

        for candidate_label in record["theory"]:
            candidate_known = dataset.has_formula(candidate_label)
            candidate_formula = dataset.formula(candidate_label) if candidate_known else None
            candidate = Formula(candidate_formula) if candidate_formula is not None else None

            yield FeatureRow(
                goal=goal_formula,
                candidate=candidate,
                used=candidate_label in used_set,
                candidate_known=candidate_known,
            )


def write_feature_table(
    output_path: str | PathLike[str] | Path = DEFAULT_OUTPUT,
    dataset_path: str | PathLike[str] | Path = DATASET_FILE,
) -> Path:
    """Generate the feature CSV at ``output_path`` and return the written path."""
    dataset = Dataset(dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["goal", "candidate", "used", "candidate_known"])
        writer.writeheader()

        for row in iter_feature_rows(dataset):
            writer.writerow(
                {
                    "goal": _formula_to_json(row.goal),
                    "candidate": _formula_to_json(row.candidate) if row.candidate is not None else "",
                    "used": row.used,
                    "candidate_known": row.candidate_known,
                }
            )

    return output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Hyperon candidate feature table CSV.")
    parser.add_argument("--dataset", default=str(DATASET_FILE), help="Input trace dataset JSON path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    return parser


def main() -> None:
    """CLI entry point for creating a feature CSV."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    written = write_feature_table(output_path=args.output, dataset_path=args.dataset)
    print(f"Written {written}")


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_OUTPUT", "FeatureRow", "iter_feature_rows", "write_feature_table", "main"]
