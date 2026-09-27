"""Build strictly online chronological candidate-experience features."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infcontrol.operators import Formula


@dataclass(frozen=True)
class TraceRecord:
    theorem_index: int
    label: str
    theory: tuple[str, ...]
    goal: Any
    smallest: frozenset[str]


CONTEXT_COLUMNS = (
    "candidate_age",
    "candidate_fresh_1",
    "candidate_fresh_2",
    "candidate_fresh_5",
    "candidate_used_count",
    "candidate_exposure_count",
    "candidate_prior_alpha1",
    "candidate_prior_alpha5",
    "candidate_prior_logodds_alpha5",
    "recent_used_count_1",
    "recent_used_count_2",
    "recent_used_count_5",
    "recent_used_count_10",
    "recent_used_count_20",
    "ema_used_h5",
    "ema_used_h10",
    "ema_used_h20",
    "theorem_index",
)


def _tokenize(text: str):
    token = ""
    for char in text:
        if char in "()":
            if token:
                yield token
                token = ""
            yield char
        elif char.isspace():
            if token:
                yield token
                token = ""
        else:
            token += char
    if token:
        yield token


def _parse(tokens):
    stack = []
    current = None
    for token in tokens:
        if token == "(":
            nested = []
            if current is not None:
                stack.append(current)
            current = nested
        elif token == ")":
            if stack:
                finished = current
                current = stack.pop()
                current.append(finished)
            else:
                yield current
                current = None
        elif current is not None:
            current.append(token)


def read_trace(path: str | Path) -> list[TraceRecord]:
    source = Path(path).read_text(encoding="utf-8")
    source = "\n".join(line.split(";;", 1)[0] for line in source.splitlines())
    records = []
    for theorem_index, expression in enumerate(_parse(_tokenize(source))):
        if not expression or expression[0] != "TraceFullSmallest":
            continue
        if len(expression) != 5:
            raise ValueError(
                f"unexpected TraceFullSmallest record at index {theorem_index}"
            )
        _, label, theory, goal, smallest = expression
        records.append(
            TraceRecord(
                theorem_index=len(records),
                label=label,
                theory=tuple(theory),
                goal=goal,
                smallest=frozenset(smallest),
            )
        )
    return records


def _online_context(record: TraceRecord, state: dict[str, Any]) -> dict[str, float]:
    candidate = state["candidate"]
    current = record.theorem_index
    first_seen = state["first_seen"].get(candidate, current)
    age = float(current - first_seen)
    used_count = float(state["used_count"].get(candidate, 0))
    exposure_count = float(state["exposure_count"].get(candidate, 0))
    global_p = (
        state["total_used"] / state["total_exposure"]
        if state["total_exposure"]
        else 0.5
    )

    def recent(window: int) -> float:
        return float(sum(candidate in used for used in state["recent_used"][-window:]))

    def ema(half_life: float) -> float:
        return float(
            sum(
                2.0 ** (-(current - index) / half_life)
                for index in state["use_times"].get(candidate, ())
            )
        )

    prior = (used_count + global_p) / (exposure_count + 1.0)
    prior_alpha5 = (used_count + 5.0 * global_p) / (exposure_count + 5.0)
    return {
        "candidate_age": age,
        "candidate_fresh_1": float(age <= 1),
        "candidate_fresh_2": float(age <= 2),
        "candidate_fresh_5": float(age <= 5),
        "candidate_used_count": used_count,
        "candidate_exposure_count": exposure_count,
        "candidate_prior_alpha1": prior,
        "candidate_prior_alpha5": prior_alpha5,
        "candidate_prior_logodds_alpha5": math.log(prior_alpha5 / (1.0 - prior_alpha5)),
        "recent_used_count_1": recent(1),
        "recent_used_count_2": recent(2),
        "recent_used_count_5": recent(5),
        "recent_used_count_10": recent(10),
        "recent_used_count_20": recent(20),
        "ema_used_h5": ema(5),
        "ema_used_h10": ema(10),
        "ema_used_h20": ema(20),
        "theorem_index": float(current),
    }


class OnlineContextTracker:
    """Maintain the strictly chronological feature state used at inference."""

    def __init__(self) -> None:
        self.theorem_index = 0
        self.state: dict[str, Any] = {
            "first_seen": {},
            "used_count": defaultdict(int),
            "exposure_count": defaultdict(int),
            "use_times": defaultdict(list),
            "recent_used": [],
            "total_used": 0,
            "total_exposure": 0,
        }

    def features(self, candidate_labels: list[str]) -> dict[str, list[float]]:
        columns = {name: [] for name in CONTEXT_COLUMNS if name != "theorem_index"}
        for candidate in candidate_labels:
            values = _online_context(
                TraceRecord(self.theorem_index, "", (), None, frozenset()),
                {**self.state, "candidate": candidate},
            )
            for name, column in columns.items():
                column.append(values[name])
        return columns

    def observe(self, theory_labels: list[str], used_labels: list[str]) -> None:
        for candidate in theory_labels:
            self.state["first_seen"].setdefault(candidate, self.theorem_index)
            self.state["exposure_count"][candidate] += 1
            self.state["total_exposure"] += 1
        used = set(used_labels)
        for candidate in used:
            self.state["used_count"][candidate] += 1
            self.state["use_times"][candidate].append(self.theorem_index)
        self.state["total_used"] += len(used)
        self.state["recent_used"].append(used)
        self.theorem_index += 1


def iter_context_rows(path: str | Path):
    records = read_trace(path)
    state: dict[str, Any] = {
        "first_seen": {},
        "used_count": defaultdict(int),
        "exposure_count": defaultdict(int),
        "use_times": defaultdict(list),
        "recent_used": [],
        "total_used": 0,
        "total_exposure": 0,
    }
    formulas = {record.label: Formula(record.goal) for record in records}
    for record in records:
        for candidate_index, candidate in enumerate(record.theory):
            if candidate not in formulas:
                continue
            values = _online_context(
                TraceRecord(
                    record.theorem_index,
                    record.label,
                    record.theory,
                    record.goal,
                    record.smallest,
                ),
                {**state, "candidate": candidate},
            )
            yield {
                "goal": json.dumps(record.goal, ensure_ascii=False),
                "candidate": json.dumps(
                    formulas[candidate].canonical_tree, ensure_ascii=False
                ),
                "candidate_label": candidate,
                "candidate_index": candidate_index,
                "used": str(candidate in record.smallest),
                **values,
            }
        for candidate in record.theory:
            state["first_seen"].setdefault(candidate, record.theorem_index)
            state["exposure_count"][candidate] += 1
            state["total_exposure"] += 1
        used = set(record.smallest)
        for candidate in used:
            state["used_count"][candidate] += 1
            state["use_times"][candidate].append(record.theorem_index)
        state["total_used"] += len(used)
        state["recent_used"].append(used)


def write_context_feature_table(
    input_path: str | Path, output_path: str | Path
) -> Path:
    rows = iter_context_rows(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "goal",
        "candidate",
        "candidate_label",
        "candidate_index",
        "used",
        *CONTEXT_COLUMNS,
    ]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


__all__ = [
    "CONTEXT_COLUMNS",
    "OnlineContextTracker",
    "TraceRecord",
    "iter_context_rows",
    "read_trace",
    "write_context_feature_table",
]
