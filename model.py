"""Persist and load trained inference-control models for prediction.

A saved model is a directory containing one ``.dot`` file per accepted
feature graph plus a ``model.json`` with the fitted logit coefficients. It
has no dependency on the training CSV or :class:`~infcontrol.logit_fitter.LogitFitter`,
so it can be loaded standalone by :class:`PredictionModel` (e.g. from the
``infcontrol-predict`` CLI used by other languages).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from william.library.base import Value
from william.nvmap import MapEntry, NodeValueMap
from william.propagation_py import RandomPropagation as RandomPropagationPy
from william.structures import Graph, dot_to_graph

from infcontrol.operators import Formula, FormulaArray

MODEL_FILE = "model.json"

# william's dot-file (de)serializer abbreviates "infcontrol.operators.Formula"
# to "F" in type specs (see william.library.types.spec_str); register the
# matching name so specs like "Array[F]" can be parsed back on load.
dot_to_graph.EVAL_LOCALS.setdefault("F", Formula)


def _default_operators() -> tuple[type, ...]:
    # imported lazily to avoid a circular import with inference_control.py
    from infcontrol.inference_control import DEFAULT_OPERATORS

    return DEFAULT_OPERATORS


def formula_array(values: Any) -> np.ndarray:
    """Coerce a sequence of formulas (or raw parsed trees) into a 1-D object array."""
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError("formula arrays must be one-dimensional")
    items = list(values)
    array = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        array[index] = item if isinstance(item, Formula) else Formula(item)
    return array


def _template_graph(graph: Graph) -> Graph:
    """Clone `graph` with cached value-node contents cleared.

    Accepted feature graphs carry the concrete training data (and any
    intermediate results) in their value nodes, which cannot be represented
    as parseable Python literals in the dot text format. Only the name/spec
    of each value node is needed to reload and re-evaluate the graph, since
    :func:`evaluate_graph` always supplies fresh leaf values via ``mem``.
    """
    template = graph.clone(copy=True)
    for node in template.walk(op_nodes=False, val_nodes=True):
        output = node.output
        node.output = Value(
            None, name=output.name, spec=output.spec, permeable=output.permeable
        )
    return template


def evaluate_graph(
    graph: Graph,
    prop_class: RandomPropagationPy,
    goals: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Propagate ``goals``/``candidates`` through an accepted feature graph."""
    cloned = graph.clone(copy=True)
    mem = NodeValueMap()
    values = {
        "goal": FormulaArray(goals),
        "candidate": FormulaArray(candidates),
        "one": 1.0,
    }
    for leaf in cloned.leaves():
        if leaf.output.name not in values:
            raise RuntimeError(f"unknown target leaf {leaf.output.name!r}")
        mem[leaf] = MapEntry(
            same=False,
            val=Value(values[leaf.output.name], spec=leaf.output.spec, permeable=False),
        )
    roots = list(cloned.roots())
    if len(roots) != 1:
        raise RuntimeError("accepted feature graph must have exactly one root")
    propagated = next(prop_class.propagate(roots[0], mem), None)
    if propagated is None or roots[0] not in propagated:
        raise RuntimeError("accepted feature graph could not be evaluated")
    result = propagated[roots[0]].val.value
    if not isinstance(result, np.ndarray) or result.shape != goals.shape:
        raise RuntimeError("accepted graph produced an invalid feature array")
    return np.asarray(result, dtype=float)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=float)
    positive = logits >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    return probabilities


def save_model(
    directory: str | Path,
    graphs: Iterable[Graph],
    initial_strength: float,
    coefficients: Iterable[float],
    context_feature_names: Iterable[str] = (),
) -> Path:
    """Persist accepted feature graphs and logit coefficients to ``directory``."""
    # william's Graph.save() resolves relative paths against its own package
    # directory, not the caller's cwd, so make sure we always land in `directory`.
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    coefficients = list(coefficients)
    context_feature_names = list(context_feature_names)
    graph_coefficients = coefficients[len(context_feature_names) :]
    graph_names = []
    for index, graph in enumerate(graphs):
        name = f"feature_{index:04d}.dot"
        _template_graph(graph).save(directory / name)
        graph_names.append(name)
    if len(graph_names) != len(graph_coefficients):
        raise ValueError("graphs and graph coefficients must have the same length")
    metadata = {
        "initial_strength": initial_strength,
        "context_feature_names": list(context_feature_names),
        "context_coefficients": coefficients[: len(context_feature_names)],
        "features": [
            {"graph": graph_name, "coefficient": coefficient}
            for graph_name, coefficient in zip(graph_names, graph_coefficients)
        ],
    }
    (directory / MODEL_FILE).write_text(json.dumps(metadata, indent=2))
    return directory


class PredictionModel:
    """A loaded inference-control model, ready to score goal/candidate pairs."""

    def __init__(
        self,
        graphs: list[Graph],
        coefficients: list[float],
        initial_strength: float,
        context_feature_names: list[str] | None = None,
    ) -> None:
        self.graphs = graphs
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.initial_strength = initial_strength
        self.context_feature_names = context_feature_names or []
        self.prop_class = RandomPropagationPy(partial=False, cache=False)

    @classmethod
    def load(
        cls, directory: str | Path, operators: Iterable[type] | None = None
    ) -> PredictionModel:
        """Load a model previously written by :func:`save_model`."""
        # resolve for the same reason as in save_model: william resolves
        # relative dot-file paths against its own package directory.
        directory = Path(directory).resolve()
        metadata = json.loads((directory / MODEL_FILE).read_text())
        ops = [operator(dl=8) for operator in (operators or _default_operators())]
        graphs = []
        coefficients = []
        for feature in metadata["features"]:
            graphs.append(Graph.load(directory / feature["graph"], ops=ops))
            coefficients.append(feature["coefficient"])
        coefficients = metadata.get("context_coefficients", []) + coefficients
        return cls(
            graphs,
            coefficients,
            metadata["initial_strength"],
            metadata.get("context_feature_names", []),
        )

    def predict(
        self,
        goal_array: Any,
        candidate_array: Any,
        context_features: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Predict use-probabilities for equally sized goal/candidate arrays."""
        goals = formula_array(goal_array)
        candidates = formula_array(candidate_array)
        if goals.shape != candidates.shape:
            raise ValueError("goal_array and candidate_array must have the same shape")
        if self.context_feature_names:
            if context_features is None:
                raise ValueError("context_features are required for this model")
            context_columns = [
                np.asarray(context_features[name], dtype=float)
                for name in self.context_feature_names
            ]
            if any(column.shape != goals.shape for column in context_columns):
                raise ValueError(
                    "context feature arrays must match the formula array shape"
                )
        else:
            context_columns = []
        feature_columns = context_columns + [
            evaluate_graph(graph, self.prop_class, goals, candidates)
            for graph in self.graphs
        ]
        if not feature_columns:
            return sigmoid(np.full(goals.shape, self.initial_strength, dtype=float))
        features = np.column_stack(feature_columns)
        logits = self.initial_strength + features @ self.coefficients
        return sigmoid(logits)


__all__ = [
    "PredictionModel",
    "evaluate_graph",
    "formula_array",
    "save_model",
    "sigmoid",
]
