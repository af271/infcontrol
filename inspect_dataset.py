import random
from typing import Any

from infcontrol.canonicalize import canonicalize
from infcontrol.dataset import Dataset, DatasetRecord


def format_formula(formula: Any, indent: int = 0) -> str:
    """Render a nested list formula as a multi-line indented string."""
    lines: list[str] = []
    _append_formula_lines(formula, lines, indent)
    return "\n".join(lines)


def _append_formula_lines(formula: Any, lines: list[str], indent: int) -> None:
    pad = " " * indent

    if not isinstance(formula, list):
        lines.append(pad + str(formula))
        return

    lines.append(pad + "(")

    for x in formula:
        if isinstance(x, list):
            _append_formula_lines(x, lines, indent + 2)
        else:
            lines.append(" " * (indent + 2) + str(x))

    lines.append(pad + ")")


def dataset_summary(ds: Dataset) -> dict[str, int | float]:
    """Return aggregate statistics for the theorem dataset."""
    total_candidates = sum(len(r["theory"]) for r in ds)
    total_smallest = sum(len(r["smallest"]) for r in ds)
    known_candidates = sum(1 for r in ds for c in r["theory"] if ds.has_formula(c))

    p_used = total_smallest / total_candidates if total_candidates else 0.0

    return {
        "theorems": len(ds),
        "candidate_pairs": total_candidates,
        "positive_labels": total_smallest,
        "p_used": p_used,
        "known_candidates": known_candidates,
    }


def sample_records(ds: Dataset, n: int = 5, seed: int | None = None) -> list[DatasetRecord]:
    """Sample up to ``n`` random records from the dataset."""
    rng = random.Random(seed)
    return rng.sample(ds.records, min(n, len(ds.records)))


def lookup_candidates(ds: Dataset, limit: int = 3) -> list[tuple[str, Any]]:
    """Return up to ``limit`` unique known candidate formulas."""
    seen: set[str] = set()
    found: list[tuple[str, Any]] = []

    for record in ds:
        for candidate in record["theory"]:
            if candidate in seen:
                continue
            if not ds.has_formula(candidate):
                continue

            seen.add(candidate)
            formula = ds.formula(candidate)
            if formula is None:
                continue

            found.append((candidate, formula))
            if len(found) >= limit:
                return found

    return found


def build_report(ds: Dataset, n_examples: int = 5, lookup_limit: int = 3, seed: int | None = None) -> str:
    """Build a text report equivalent to the previous script output."""
    lines: list[str] = []
    summary = dataset_summary(ds)

    lines.append("=" * 80)
    lines.append("DATASET SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Theorems              : {summary['theorems']}")
    lines.append(f"Candidate pairs       : {summary['candidate_pairs']}")
    lines.append(f"Positive labels       : {summary['positive_labels']}")
    lines.append(f"P(used)              : {summary['p_used']:.5f}")
    lines.append("")
    lines.append(f"Known candidate formulas: {summary['known_candidates']}/{summary['candidate_pairs']}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("RANDOM EXAMPLES")
    lines.append("=" * 80)

    for record in sample_records(ds, n=n_examples, seed=seed):
        lines.append("")
        lines.append("-" * 80)
        lines.append("LABEL")
        lines.append(record["label"])
        lines.append("")
        lines.append("GOAL")
        lines.append(format_formula(record["goal_formula"]))
        lines.append("")
        lines.append("CANONICAL GOAL")
        lines.append(format_formula(canonicalize(record["goal_formula"])))
        lines.append("")
        lines.append(f"{len(record['theory'])} CANDIDATES")

        for candidate in record["theory"]:
            used = candidate in record["smallest"]
            marker = "*" if used else " "
            status = "known" if ds.has_formula(candidate) else "unknown"
            lines.append(f"{marker} {candidate:20} {status}")

        lines.append("")
        lines.append("USED")
        lines.append(str(record["smallest"]))

    lines.append("")
    lines.append("=" * 80)
    lines.append("LOOKUP TEST")
    lines.append("=" * 80)

    for candidate, formula in lookup_candidates(ds, limit=lookup_limit):
        lines.append("")
        lines.append(candidate)
        lines.append(format_formula(formula))

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for inspecting the serialized dataset."""
    print(build_report(Dataset()), end="")


if __name__ == "__main__":
    main()

__all__ = [
    "format_formula",
    "dataset_summary",
    "sample_records",
    "lookup_candidates",
    "build_report",
    "main",
]
