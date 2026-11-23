"""End-to-end Hyperon feature enumeration and logistic fitting."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from time import perf_counter, time
from typing import Any

import numpy as np

from infcontrol.context_features import CONTEXT_COLUMNS
from infcontrol.logit_fitter import LogitFitter
from infcontrol.model import evaluate_graph, formula_array, save_model
from infcontrol.operators import (
    AbsDifference,
    Formula,
    FormulaArray,
    FormulaEqual,
    ImplicationChainLength,
    MaxSharedSubtreeSize,
    NumOperators,
    NumVariables,
    OperatorCosineSimilarity,
    OperatorHistogram,
    OperatorL1Distance,
    PremiseCount,
    Ratio,
    RootOperator,
    SharedOperators,
    SharedSymbols,
    SharedVariables,
    SkeletonizeFormula,
    SubtreeJaccard,
    SubtreeSet,
    SymbolHistogram,
    TreeDepth,
    TreeSize,
)
from infcontrol.template_memory import feature_record, load_templates, primitive_templates, program_info
from william.library.base import Value
from william.library.basic_ops import Equal, Len, LessThan
from william.library.complex_ops import Abs, Mean, Sqrt, Sum
from william.library.types import Array
from william.mork_memory import MorkTemplateMemory
from william.propagation_py import RandomPropagation as RandomPropagationPy
from william.structures import Graph
from william.wunderbaum import Wunderbaum

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
    # Exp,
    # Log,
    Sqrt,
    # Sin,
    # Cos,
    # Tanh,
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

GENERAL_OPERATORS = (
    TreeSize,
    TreeDepth,
    NumVariables,
    NumOperators,
    ImplicationChainLength,
    Abs,
    LessThan,
    AbsDifference,
    Ratio,
)


class InferenceControl:
    """Enumerate Hyperon features and greedily fit them to CSV labels."""

    def __init__(
        self,
        csv_path: str | Path,
        operators: Iterable[type] = DEFAULT_OPERATORS,
        max_popped: int | None = 1000,
        level: int = 0,
        fitter_kwargs: dict[str, Any] | None = None,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        split_seed: int = 42,
        data_fraction: float = 1.0,
        use_context: bool = True,
        mork_url: str | None = None,
        memory_namespace: str = "hyperon-features-v1",
        erase_memory: bool = False,
        max_seconds: float | None = None,
        min_template_operators: int = 1,
        primitive_prior: bool = False,
        audit_candidates: bool = False,
        allow_partial_features: bool = False,
        template_strategy: str = "dl",
        remember_templates: bool = True,
    ) -> None:
        self.started_at = perf_counter()
        self.deadline_ts = None if max_seconds is None else time() + max_seconds
        if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive.")
        if min_template_operators < 1:
            raise ValueError("min_template_operators must be positive.")
        if template_strategy not in ("dl", "eager"):
            raise ValueError("template_strategy must be 'dl' or 'eager'.")
        if primitive_prior and mork_url is not None:
            raise ValueError("The static primitive prior must not use learned MORK templates.")
        self.max_seconds = max_seconds
        self.template_strategy = template_strategy
        self.remember_templates = remember_templates
        self.audit_candidates = audit_candidates
        self.allow_partial_features = allow_partial_features
        self.candidate_programs: dict[str, dict[str, Any]] = {}
        self.accepted_programs: list[dict[str, Any]] = []
        self.csv_path = Path(csv_path)
        self.max_popped = max_popped
        if max_popped is not None and max_popped < 0:
            raise ValueError("max_popped must be non-negative.")
        if not 0 < data_fraction <= 1:
            raise ValueError("data_fraction must be greater than 0 and at most 1")
        self.rows = self._read_rows(self.csv_path)
        if not self.rows:
            raise ValueError("CSV must contain at least one row with two formulas")
        self.rows = self._sample_rows_by_goal(self.rows, data_fraction, split_seed)
        self.features, self.labels = self._arrays_from_rows(self.rows)
        self.group_ids = np.asarray(
            [row.get("theorem_index") or row["goal"] for row in self.rows],
            dtype=object,
        )
        self.context_feature_names = (
            [name for name in CONTEXT_COLUMNS if name in self.rows[0] and name != "theorem_index"]
            if use_context
            else []
        )
        self.context_features = (
            np.asarray(
                [[float(row[name]) for name in self.context_feature_names] for row in self.rows],
                dtype=float,
            )
            if self.context_feature_names
            else np.empty((len(self.rows), 0), dtype=float)
        )
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
        fitter_options = {
            "objective": "pairwise_rank",
            "acceptance_metric": "k_at_recall",
            "recall_target": 1.0,
            **(fitter_kwargs or {}),
        }
        self.fitter = LogitFitter(
            self.labels,
            groups=self.group_ids,
            group_order=(
                np.asarray([float(row["theorem_index"]) for row in self.rows], dtype=float)
                if "theorem_index" in self.rows[0]
                else None
            ),
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
            **fitter_options,
        )
        primitive_ops = [operator(dl=8) for operator in operators]
        self.operator_names = [operator.name for operator in primitive_ops]
        self.template_memory = (
            MorkTemplateMemory(mork_url, memory_namespace, erase=erase_memory) if mork_url is not None else None
        )
        self.loaded_templates = (
            load_templates(self.template_memory, primitive_ops, min_template_operators)
            if self.template_memory is not None
            else []
        )
        self.initial_templates = (
            primitive_templates(primitive_ops)
            if primitive_prior
            else self.loaded_templates
            if template_strategy == "eager"
            else []
        )
        self.enum = Wunderbaum(
            [(operator, 0) for operator in primitive_ops]
            + [(template.operator, template.count) for template in self.loaded_templates],
            level=level,
        )
        self.accepted_template_sources: list[list[str]] = []
        self.template_attempts = 0
        self._templates_evaluated = False
        self.prop_class = RandomPropagationPy(partial=False, cache=False)
        self.last_mem: Any | None = None
        self.accepted_graphs: list[Graph] = []
        self.accepted_context_features: list[np.ndarray] = []
        self.accepted_context_names: list[str] = []
        self._context_evaluated = False

    @staticmethod
    def _sample_rows_by_goal(rows: list[dict[str, str]], fraction: float, seed: int) -> list[dict[str, str]]:
        if fraction == 1:
            return rows

        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            group_id = row.get("theorem_index") or row["goal"]
            grouped.setdefault(group_id, []).append(row)

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
            return [row for row in csv.DictReader(file) if row.get("goal") and row.get("candidate")]

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
        if not self._context_evaluated:
            for column, name in enumerate(self.context_feature_names):
                if self._time_expired():
                    return
                values = self.context_features[:, column]
                if (
                    np.all(np.isfinite(values))
                    and self.fitter.evaluate(values, deadline_ts=self.deadline_ts) is not None
                ):
                    self.accepted_context_features.append(values.copy())
                    self.accepted_context_names.append(name)
            self._context_evaluated = True
        if not self._templates_evaluated:
            self._templates_evaluated = True
            for template in self.initial_templates:
                if self._time_expired():
                    return
                if template.graph.nodes[0].output.spec not in (Array[int], Array[float], Array[bool]):
                    continue
                if self.max_popped is not None and self.template_attempts >= self.max_popped:
                    break
                self.template_attempts += 1
                values = evaluate_graph(template.graph, self.prop_class, self.features[:, 0], self.features[:, 1])
                if self._time_expired():
                    return
                self._audit_graph(template.graph)
                if self._time_expired():
                    return
                coefficient = self.fitter.evaluate(values, deadline_ts=self.deadline_ts)
                if coefficient is None:
                    continue
                self.accepted_graphs.append(template.graph.clone(copy=True))
                self.accepted_template_sources.append(
                    [template.record] if any(template is item for item in self.loaded_templates) else []
                )
                self._audit_graph(template.graph, accepted=True)
                self._remember_graph(template.graph)
                yield template.graph, coefficient
        remaining = None if self.max_popped is None else self.max_popped - self.template_attempts
        if self._time_expired():
            return
        for graph, _dl, mem, _bottleneck in self.enum.iterate(
            self.prop_class,
            self.targets,
            max_popped=remaining,
            enum_only=True,
            deadline_ts=self.deadline_ts,
        ):
            if self._time_expired():
                return
            roots = list(graph.roots())
            if self.allow_partial_features:
                roots = [root for root in roots if root.options]
            if not roots or roots[0] not in mem or len(roots) != 1:
                continue
            if self.allow_partial_features:
                graph = Graph([roots[0]])
            feature = mem[roots[0]].val.value
            if not isinstance(feature, np.ndarray) or feature.shape != self.labels.shape:
                continue
            if feature.dtype.kind not in "iufb":
                continue
            if not np.all(np.isfinite(feature)):
                continue

            self._audit_graph(graph)
            if self._time_expired():
                return
            coefficient = self.fitter.evaluate(feature, deadline_ts=self.deadline_ts)
            if coefficient is None:
                continue

            self.last_mem = mem
            self.accepted_graphs.append(graph.clone(copy=True))
            sources = [
                template.record
                for template in self.loaded_templates
                if any(node.op is template.operator for node in graph.walk(val_nodes=False))
            ]
            self.accepted_template_sources.append(sources)
            self._audit_graph(graph, accepted=True)
            self._remember_graph(graph)
            yield graph, coefficient

    @property
    def candidate_attempts(self) -> int:
        return self.template_attempts + self.enum.num_popped

    def _time_expired(self) -> bool:
        return self.deadline_ts is not None and time() >= self.deadline_ts

    def _audit_graph(self, graph: Graph, *, accepted: bool = False) -> None:
        if not self.audit_candidates:
            return
        elapsed_seconds = perf_counter() - self.started_at
        info = {**program_info(graph), "elapsed_seconds": elapsed_seconds}
        if accepted:
            info["within_time_budget"] = self.max_seconds is None or info["elapsed_seconds"] <= self.max_seconds
            self.accepted_programs.append(info)
        else:
            self.candidate_programs.setdefault(info["record"], info)

    def _remember_graph(self, graph: Graph) -> None:
        if self.template_memory is None or not self.remember_templates:
            return
        root = list(graph.roots())[0].clone_and_unpack()
        records = [feature_record(node) for node in root.walk(above=False, op_nodes=False) if node.options]
        self.template_memory.remember(records)

    def predict(
        self,
        goal_array: Any,
        candidate_array: Any,
        context_features: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict probabilities for equally sized Goal and candidate arrays."""
        goals = formula_array(goal_array)
        candidates = formula_array(candidate_array)
        if goals.shape != candidates.shape:
            raise ValueError("goal_array and candidate_array must have the same shape")
        context_columns = []
        if self.accepted_context_names:
            if context_features is None:
                raise ValueError("context_features are required for this model")
            context_columns = [np.asarray(context_features[name], dtype=float) for name in self.accepted_context_names]
            if any(column.shape != goals.shape for column in context_columns):
                raise ValueError("context feature arrays must match the formula array shape")
        feature_columns = context_columns + [
            evaluate_graph(graph, self.prop_class, goals, candidates) for graph in self.accepted_graphs
        ]
        features = np.column_stack(feature_columns) if feature_columns else np.empty((goals.size, 0))
        return self.fitter.predict(features)

    def log_loss(self, split: str = "train") -> float:
        """Return the current model's mean log-loss on a data split."""
        return self.fitter.log_loss(split)

    def save(self, directory: str | Path) -> Path:
        """Persist the accepted feature graphs and fitted coefficients.

        The resulting directory can be loaded standalone with
        :meth:`infcontrol.model.PredictionModel.load`, without the training
        CSV or this ``InferenceControl`` instance.
        """
        saved = save_model(
            directory,
            self.accepted_graphs,
            self.fitter.initial_strength,
            self.fitter.coefficients,
            self.accepted_context_names,
        )
        if self.template_memory is not None:
            report = {
                "template_strategy": self.template_strategy,
                "remember_templates": self.remember_templates,
                "namespace": self.template_memory.namespace,
                "generation": self.template_memory.generation,
                "candidate_attempts": self.candidate_attempts,
                "template_attempts": self.template_attempts,
                "loaded_templates": [
                    {"record": template.record, "count": template.count} for template in self.loaded_templates
                ],
                "features": [
                    {"graph": f"feature_{index:04d}.dot", "source_templates": sources}
                    for index, sources in enumerate(self.accepted_template_sources)
                ],
            }
            (saved / "template_memory.json").write_text(json.dumps(report, indent=2))
        return saved

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
        groups = self.group_ids[indices]
        labels = self.labels[indices]

        grouped: dict[str, list[tuple[float, float]]] = {}
        for group, probability, label in zip(groups, probabilities, labels):
            grouped.setdefault(str(group), []).append((probability, label))

        recalls = []
        for candidates in grouped.values():
            total_used = sum(label == 1.0 for _, label in candidates)
            if total_used == 0:
                continue
            ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
            used_in_top_k = sum(label == 1.0 for _, label in ranked[:k])
            recalls.append(used_in_top_k / total_used)
        return float(np.mean(recalls)) if recalls else float("nan")

    def all_at_k(self, split: str, k: int) -> float:
        """Return the fraction of theorem instances with every positive in top k."""
        if k <= 0:
            raise ValueError("k must be positive")
        indices = {
            "train": self.fitter.train_indices,
            "validation": self.fitter.validation_indices,
            "test": self.fitter.test_indices,
        }[split]
        probabilities = self.fitter.probabilities[indices]
        groups = self.group_ids[indices]
        labels = self.labels[indices]

        grouped: dict[str, list[tuple[float, float]]] = {}
        for group, probability, label in zip(groups, probabilities, labels):
            grouped.setdefault(str(group), []).append((probability, label))

        complete = []
        for candidates in grouped.values():
            total_used = sum(label == 1.0 for _, label in candidates)
            if total_used == 0:
                continue
            ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
            used_in_top_k = sum(label == 1.0 for _, label in ranked[:k])
            complete.append(used_in_top_k == total_used)
        return float(np.mean(complete)) if complete else float("nan")

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
        groups = self.group_ids[indices]
        labels = self.labels[indices]

        grouped: dict[str, list[tuple[float, float]]] = {}
        for group, probability, label in zip(groups, probabilities, labels):
            grouped.setdefault(str(group), []).append((probability, label))

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
        recall_target: float = 1.0,
    ) -> None:
        headers = (
            [
                "split",
                "log_loss",
                "geo_mean_p",
                # "accuracy",
                f"k@recall>={recall_target:.0%}",
            ]
            + [f"recall@{k}" for k in recall_ks]
            + [f"all@{k}" for k in recall_ks]
        )
        rows = []
        for split in ("train", "validation", "test"):
            loss = self.fitter.log_loss(split)
            rows.append(
                [
                    split,
                    f"{loss:.6f}",
                    f"{np.exp(-loss):.6f}",
                    # f"{self.accuracy(split):.6f}",
                    f"{self.k_at_recall(split, recall_target):.2f}",
                    *[f"{self.recall_at_k(split, k):.6f}" for k in recall_ks],
                    *[f"{self.all_at_k(split, k):.6f}" for k in recall_ks],
                ]
            )

        widths = [
            max(len(str(value)) for value in [header, *(row[i] for row in rows)]) for i, header in enumerate(headers)
        ]
        print(f"features: {len(self.fitter.features)}")
        print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
        print("-+-".join("-" * width for width in widths))
        for row in rows:
            print(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)))


__all__ = ["DEFAULT_OPERATORS", "InferenceControl"]
