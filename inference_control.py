"""End-to-end Hyperon feature enumeration and logistic fitting."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from william.library.base import Value
from william.library.types import Array
from william.propagation_py import RandomPropagation as RandomPropagationPy
from william.structures import Graph
from william.wunderbaum import Wunderbaum

from infcontrol.logit_fitter import LogitFitter
from infcontrol.operators import (
    Formula,
    FormulaArray,
    FormulaEqual,
    ImplicationChainLength,
    NumOperators,
    NumVariables,
    OperatorHistogram,
    RootOperator,
    SharedOperators,
    SharedSymbols,
    SharedVariables,
    SymbolHistogram,
    TreeDepth,
    TreeSize,
)

DEFAULT_OPERATORS = (
    TreeSize,
    TreeDepth,
    NumVariables,
    NumOperators,
    RootOperator,
    OperatorHistogram,
    SymbolHistogram,
    ImplicationChainLength,
    FormulaEqual,
    SharedVariables,
    SharedOperators,
    SharedSymbols,
)


class InferenceControl:
    """Enumerate Hyperon features and greedily fit them to CSV labels."""

    def __init__(
        self,
        csv_path: str | Path,
        operators: Iterable[type] = DEFAULT_OPERATORS,
        output_dir: str | Path | None = None,
        max_popped: int | None = 1000,
        level: int = 0,
        fitter_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.max_popped = max_popped
        self.rows = self._read_rows(self.csv_path)
        if not self.rows:
            raise ValueError("CSV must contain at least one row with two formulas")

        self.features, self.labels = self._arrays_from_rows(self.rows)
        self.targets = [
            Value(
                FormulaArray(self.features[:, column]),
                spec=Array[Formula],
                permeable=False,
                name=name,
            )
            for column, name in enumerate(("goal", "candidate"))
        ]
        self.fitter = LogitFitter(self.labels, **(fitter_kwargs or {}))
        self.enum = Wunderbaum(
            [(operator(dl=8), 0) for operator in operators], level=level
        )
        self.prop_class = RandomPropagationPy(partial=False, cache=False)
        self.saved_graphs: list[Path] = []

    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open(encoding="utf-8", newline="") as file:
            return [
                row
                for row in csv.DictReader(file)
                if row.get("goal") and row.get("candidate")
            ]

    @staticmethod
    def _arrays_from_rows(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        formulas = np.empty((len(rows), 2), dtype=object)
        labels = np.empty(len(rows), dtype=float)
        for index, row in enumerate(rows):
            formulas[index, 0] = Formula(json.loads(row["goal"]))
            formulas[index, 1] = Formula(json.loads(row["candidate"]))
            labels[index] = 1.0 if row.get("used", "False").lower() == "true" else 0.0
        return formulas, labels

    def run(self) -> Iterator[tuple[Graph, float]]:
        """Enumerate, evaluate, and save accepted feature graphs."""
        for graph, _dl, mem, _bottleneck in self.enum.iterate(
            self.prop_class,
            self.targets,
            max_popped=self.max_popped,
            enum_only=True,
        ):
            roots = list(graph.roots())
            if not roots or roots[0] not in mem:
                continue
            feature = mem[roots[0]].val.value
            if (
                not isinstance(feature, np.ndarray)
                or feature.shape != self.labels.shape
            ):
                continue
            if feature.dtype.kind not in "iufb":
                continue

            coefficient = self.fitter.evaluate(feature)
            if coefficient is None:
                continue

            graph_path = self._save_graph(graph, mem, len(self.saved_graphs))
            if graph_path is not None:
                self.saved_graphs.append(graph_path)
            yield graph, coefficient

    def _save_graph(self, graph: Graph, mem: Any, index: int) -> Path | None:
        if self.output_dir is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"feature_{index}"
        graph.render(filename=path, mem=mem, create_file="dot", view=False)
        return path.with_suffix(".dot")


__all__ = ["DEFAULT_OPERATORS", "InferenceControl"]
