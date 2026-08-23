"""End-to-end Hyperon feature enumeration and logistic fitting."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from william.library.base import Value
from william.library.basic_ops import Equal, Len, LessThan
from william.library.complex_ops import Abs, Cos, Exp, Log, Mean, Sin, Sqrt, Sum, Tanh
from william.library.types import Array
from william.propagation_py import RandomPropagation as RandomPropagationPy
from william.structures import Graph
from william.wunderbaum import Wunderbaum

from infcontrol.logit_fitter import LogitFitter
from infcontrol.operators import (
    Formula,
    FormulaArray,
    FormulaEqual,
    AbsDifference,
    ImplicationChainLength,
    MaxSharedSubtreeSize,
    NumOperators,
    NumVariables,
    OperatorCosineSimilarity,
    OperatorL1Distance,
    OperatorHistogram,
    PremiseCount,
    Ratio,
    RootOperator,
    SkeletonizeFormula,
    SharedOperators,
    SharedSymbols,
    SharedVariables,
    SubtreeJaccard,
    SubtreeSet,
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
    Abs,
    Sum,
    Mean,
    #Exp,
    #Log,
    Sqrt,
    #Sin,
    #Cos,
    #Tanh,
    Len,
    LessThan,
    Equal,
    OperatorL1Distance,
    OperatorCosineSimilarity,
    PremiseCount,
    AbsDifference,
    Ratio,
    SkeletonizeFormula,
    SubtreeSet,
    SubtreeJaccard,
    MaxSharedSubtreeSize,
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
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        split_seed: int = 42,
        data_fraction: float = 1.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.max_popped = max_popped
        if not 0 < data_fraction <= 1:
            raise ValueError("data_fraction must be greater than 0 and at most 1")
        self.rows = self._read_rows(self.csv_path)
        if not self.rows:
            raise ValueError("CSV must contain at least one row with two formulas")
        self.rows = self._sample_rows_by_goal(self.rows, data_fraction, split_seed)
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
        self.targets.append(Value(1.0, spec=float, permeable=True, name="one"))
        self.fitter = LogitFitter(
            self.labels,
            groups=self.features[:, 0],
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
            **(fitter_kwargs or {}),
        )
        self.enum = Wunderbaum(
            [(operator(dl=8), 0) for operator in operators], level=level
        )
        self.prop_class = RandomPropagationPy(partial=False, cache=False)
        self.saved_graphs: list[Path] = []
        self.last_mem: Any | None = None

    @staticmethod
    def _sample_rows_by_goal(
        rows: list[dict[str, str]], fraction: float, seed: int
    ) -> list[dict[str, str]]:
        if fraction == 1:
            return rows

        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["goal"], []).append(row)

        group_names = list(grouped)
        np.random.default_rng(seed).shuffle(group_names)
        target_size = int(np.ceil(len(rows) * fraction))
        selected: list[dict[str, str]] = []
        for group_name in group_names:
            selected.extend(grouped[group_name])
            if len(selected) >= target_size:
                break
        return selected

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
            if not roots or roots[0] not in mem or len(roots) != 1:
                continue
            feature = mem[roots[0]].val.value
            if (
                not isinstance(feature, np.ndarray)
                or feature.shape != self.labels.shape
            ):
                continue
            if feature.dtype.kind not in "iufb":
                continue
            if not np.all(np.isfinite(feature)):
                continue

            coefficient = self.fitter.evaluate(feature)
            if coefficient is None:
                continue

            graph_path = self._save_graph(graph, mem, len(self.saved_graphs))
            if graph_path is not None:
                self.saved_graphs.append(graph_path)
            self.last_mem = mem
            yield graph, coefficient

    def log_loss(self, split: str = "train") -> float:
        """Return the current model's mean log-loss on a data split."""
        return self.fitter.log_loss(split)

    def accuracy(self, split: str) -> float:
        indices = {
            "train": self.fitter.train_indices,
            "validation": self.fitter.validation_indices,
            "test": self.fitter.test_indices,
        }[split]
        predictions = self.fitter.probabilities[indices] >= 0.5
        return float(np.mean(predictions == self.labels[indices]))

    def recall_at_k(self, split: str, k: int) -> float:
        if k <= 0:
            raise ValueError("k must be positive")
        indices = {
            "train": self.fitter.train_indices,
            "validation": self.fitter.validation_indices,
            "test": self.fitter.test_indices,
        }[split]
        probabilities = self.fitter.probabilities[indices]
        goals = self.features[indices, 0]
        labels = self.labels[indices]

        grouped: dict[str, list[tuple[float, float]]] = {}
        for goal, probability, label in zip(goals, probabilities, labels):
            grouped.setdefault(str(goal), []).append((probability, label))

        recalls = []
        for candidates in grouped.values():
            total_used = sum(label == 1.0 for _, label in candidates)
            if total_used == 0:
                continue
            ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
            used_in_top_k = sum(label == 1.0 for _, label in ranked[:k])
            recalls.append(used_in_top_k / total_used)
        return float(np.mean(recalls)) if recalls else float("nan")

    def k_at_recall(self, split: str, recall_target: float = 0.8) -> float:
        """Return the mean smallest k needed to reach ``recall_target`` per goal."""
        if not 0 < recall_target <= 1:
            raise ValueError("recall_target must be greater than 0 and at most 1")

        indices = {
            "train": self.fitter.train_indices,
            "validation": self.fitter.validation_indices,
            "test": self.fitter.test_indices,
        }[split]
        probabilities = self.fitter.probabilities[indices]
        goals = self.features[indices, 0]
        labels = self.labels[indices]

        grouped: dict[str, list[tuple[float, float]]] = {}
        for goal, probability, label in zip(goals, probabilities, labels):
            grouped.setdefault(str(goal), []).append((probability, label))

        required_ks = []
        for candidates in grouped.values():
            total_used = sum(label == 1.0 for _, label in candidates)
            if total_used == 0:
                continue
            required_hits = int(np.ceil(recall_target * total_used))
            ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
            hits = 0
            for rank, (_, label) in enumerate(ranked, start=1):
                hits += label == 1.0
                if hits >= required_hits:
                    required_ks.append(rank)
                    break
        return float(np.mean(required_ks)) if required_ks else float("nan")

    def print_metrics(
        self,
        recall_ks: tuple[int, ...] = (5, 50, 200),
        recall_target: float = 0.8,
    ) -> None:
        headers = [
            "split",
            "log_loss",
            "geo_mean_p",
            # "accuracy",
            f"k@recall>={recall_target:.0%}",
        ] + [f"recall@{k}" for k in recall_ks]
        rows = []
        for split in ("train", "validation", "test"):
            loss = self.log_loss(split)
            rows.append(
                [
                    split,
                    f"{loss:.6f}",
                    f"{np.exp(-loss):.6f}",
                    # f"{self.accuracy(split):.6f}",
                    f"{self.k_at_recall(split, recall_target):.2f}",
                    *[f"{self.recall_at_k(split, k):.6f}" for k in recall_ks],
                ]
            )

        widths = [
            max(len(str(value)) for value in [header, *(row[i] for row in rows)])
            for i, header in enumerate(headers)
        ]
        print(f"features: {len(self.fitter.features)}")
        print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
        print("-+-".join("-" * width for width in widths))
        for row in rows:
            print(
                " | ".join(str(value).ljust(width) for value, width in zip(row, widths))
            )

    def _save_graph(self, graph: Graph, mem: Any, index: int) -> Path | None:
        if self.output_dir is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"feature_{index}"
        graph.render(filename=path, mem=mem, create_file="dot", view=False)
        return path.with_suffix(".dot")


__all__ = ["DEFAULT_OPERATORS", "InferenceControl"]
