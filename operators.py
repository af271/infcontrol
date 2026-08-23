#!/usr/bin/env python3

"""Symbolic formula features and WILLIAM-style operators.

This module provides:

- :class:`Formula`: immutable wrapper around a parsed formula tree with cached
  structural metadata.
- Unary feature operators (subclasses of ``UnOp`` / ``UnaryOp``).
- Binary relation operators (subclasses of ``BinOp``).

Design goals:

- Parse/canonicalize/traverse only once per formula construction.
- Keep operator calls O(1) or O(min(n, m)) set operations on cached fields.
- Stay compatible with WILLIAM's operator framework (`_raw_specs`, `_call`,
  `_suitable`, `check`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from infcontrol.canonicalize import canonicalize, is_variable
from william.library.base import BinOp, UnaryOp
from william.library.description import desc_len
from william.library.types import Array

Tree = list["TreeNode"]
TreeNode = str | int | float | bool | None | Tree
FrozenTree = tuple["FrozenNode", ...]
FrozenNode = str | int | float | bool | None | FrozenTree

IMPLICATION_SYMBOLS = {"->", "→"}


@dataclass(frozen=True, slots=True)
class Formula:
    """Immutable symbolic formula with cached structural features.

    Parameters
    ----------
    tree:
        Parsed MeTTa-like syntax tree (nested Python lists with atoms).

    Notes
    -----
    The constructor canonicalizes variables once, then computes and caches all
    requested structural quantities in a single recursive traversal.
    """

    raw_tree: FrozenNode
    canonical_tree: FrozenNode
    size: int
    depth: int
    variables: frozenset[str]
    operators: frozenset[str]
    symbols: frozenset[str]
    operator_histogram: dict[str, int]
    symbol_histogram: dict[str, int]
    root_operator: str | None
    implication_chain_length: int

    def __init__(self, tree: TreeNode) -> None:
        frozen_raw = _freeze(tree)
        canonical = canonicalize(_thaw(frozen_raw))
        frozen_canonical = _freeze(canonical)

        stats = _compute_stats(frozen_canonical)

        object.__setattr__(self, "raw_tree", frozen_raw)
        object.__setattr__(self, "canonical_tree", frozen_canonical)
        object.__setattr__(self, "size", stats.size)
        object.__setattr__(self, "depth", stats.depth)
        object.__setattr__(self, "variables", frozenset(stats.variables))
        object.__setattr__(self, "operators", frozenset(stats.operators))
        object.__setattr__(self, "symbols", frozenset(stats.symbol_histogram.keys()))
        object.__setattr__(self, "operator_histogram", dict(stats.operator_histogram))
        object.__setattr__(self, "symbol_histogram", dict(stats.symbol_histogram))
        object.__setattr__(self, "root_operator", stats.root_operator)
        object.__setattr__(
            self, "implication_chain_length", stats.implication_chain_length
        )

    def desc_len(self, mode: str = "use_gaussian") -> float:
        return desc_len(self.canonical_tree, mode=mode)

    def __str__(self) -> str:
        def format_node(node: FrozenNode) -> str:
            if isinstance(node, tuple):
                return f"({' '.join(format_node(child) for child in node)})"
            return str(node)

        return format_node(self.canonical_tree)


class FormulaArray(np.ndarray):
    """One-dimensional Formula array with William-compatible rendering."""

    def __new__(cls, values: Any) -> Self:
        array = np.asarray(values, dtype=object).view(cls)
        if array.ndim != 1 or not all(isinstance(value, Formula) for value in array):
            raise ValueError("FormulaArray must contain one-dimensional Formula values")
        return array

    def __array_finalize__(self, obj: Any) -> None:
        pass

    def __str__(self) -> str:
        return "[" + ", ".join(str(value) for value in self) + "]"

    def __repr__(self) -> str:
        return str(self)

    def desc_len(self, mode: str = "use_gaussian") -> float:
        return float(sum(value.desc_len(mode=mode) for value in self))


def _subtrees(tree: FrozenNode) -> set[FrozenNode]:
    if not isinstance(tree, tuple):
        return set()
    result = {tree}
    for child in tree:
        result.update(_subtrees(child))
    return result


def _skeleton(tree: FrozenNode) -> FrozenNode:
    if isinstance(tree, tuple):
        return tuple(_skeleton(child) for child in tree)
    if isinstance(tree, str) and _is_var_symbol(tree):
        return "_"
    return tree


def _histogram_vector(left: Formula, right: Formula) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(left.operators | right.operators)
    return (
        np.asarray([left.operator_histogram.get(key, 0) for key in keys], dtype=float),
        np.asarray([right.operator_histogram.get(key, 0) for key in keys], dtype=float),
    )


@dataclass(slots=True)
class _Stats:
    size: int
    depth: int
    variables: set[str]
    operators: set[str]
    operator_histogram: Counter[str]
    symbol_histogram: Counter[str]
    root_operator: str | None
    implication_chain_length: int


def _freeze(node: TreeNode) -> FrozenNode:
    """Recursively convert nested lists into nested tuples."""
    if isinstance(node, list):
        return tuple(_freeze(child) for child in node)
    if isinstance(node, tuple):
        return tuple(_freeze(child) for child in node)
    return node


def _thaw(node: FrozenNode) -> TreeNode:
    """Recursively convert nested tuples into nested lists."""
    if isinstance(node, tuple):
        return [_thaw(child) for child in node]
    return node


def _is_canonical_variable(symbol: str) -> bool:
    if not symbol.startswith("v") or len(symbol) < 2:
        return False
    return symbol[1:].isdigit()


def _is_var_symbol(symbol: str) -> bool:
    # Support both pre-canonical symbols (Greek) and canonical vN names.
    return is_variable(symbol) or _is_canonical_variable(symbol)


def _compute_stats(tree: FrozenNode) -> _Stats:
    """Compute all cached formula statistics in one traversal."""

    operator_hist = Counter[str]()
    symbol_hist = Counter[str]()
    variables: set[str] = set()
    operators: set[str] = set()

    def visit(node: FrozenNode) -> tuple[int, int]:
        # Returns (size, depth).
        if isinstance(node, tuple):
            if not node:
                return 1, 1

            local_size = 1
            child_depths: list[int] = []

            head = node[0]
            if isinstance(head, str):
                local_size += 1
                operators.add(head)
                operator_hist[head] += 1
                symbol_hist[head] += 1
                if _is_var_symbol(head):
                    variables.add(head)
            else:
                head_size, head_depth = visit(head)
                local_size += head_size
                child_depths.append(head_depth)

            for child in node[1:]:
                child_size, child_depth = visit(child)
                local_size += child_size
                child_depths.append(child_depth)

            depth = 1 + (max(child_depths) if child_depths else 0)
            return local_size, depth

        if isinstance(node, str):
            symbol_hist[node] += 1
            if _is_var_symbol(node):
                variables.add(node)

        return 1, 0

    size, depth = visit(tree)

    root_operator: str | None = None
    if isinstance(tree, tuple) and tree and isinstance(tree[0], str):
        root_operator = tree[0]

    implication_chain_length = _implication_chain_length(tree)

    return _Stats(
        size=size,
        depth=depth,
        variables=variables,
        operators=operators,
        operator_histogram=operator_hist,
        symbol_histogram=symbol_hist,
        root_operator=root_operator,
        implication_chain_length=implication_chain_length,
    )


def _implication_chain_length(node: FrozenNode) -> int:
    """Return implication-chain length by following right-associative tails.

    For implication nodes we count one implication edge per implication link.
    The traversal follows the last argument recursively when it is itself an
    implication expression.
    """

    if not isinstance(node, tuple) or len(node) < 3:
        return 0

    head = node[0]
    if not isinstance(head, str) or head not in IMPLICATION_SYMBOLS:
        return 0

    links = max(1, len(node) - 2)
    tail = node[-1]
    if isinstance(tail, tuple):
        return links + _implication_chain_length(tail)
    return links


def _as_formula_batch(value: Any) -> np.ndarray | None:
    """Return a Formula object array when ``value`` is a valid batch input."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=object)
    else:
        return None

    if arr.dtype != object:
        arr = arr.astype(object)
    if all(isinstance(item, Formula) for item in arr.flat):
        return arr
    return None


def _apply_unary_batch(value: Any, fn, dtype: np.dtype | type) -> np.ndarray | None:
    arr = _as_formula_batch(value)
    if arr is None:
        return None

    out = np.empty(arr.shape, dtype=dtype)
    for idx, item in np.ndenumerate(arr):
        out[idx] = fn(item)
    return out


def _apply_binary_batch(
    left: Any, right: Any, fn, dtype: np.dtype | type
) -> np.ndarray | None:
    left_arr = _as_formula_batch(left)
    right_arr = _as_formula_batch(right)
    if left_arr is None and right_arr is None:
        return None

    if left_arr is None:
        if not isinstance(left, Formula):
            raise TypeError("left must be Formula when right is batched")
        left_arr = np.full(right_arr.shape, left, dtype=object)
    if right_arr is None:
        if not isinstance(right, Formula):
            raise TypeError("right must be Formula when left is batched")
        right_arr = np.full(left_arr.shape, right, dtype=object)

    if left_arr.shape != right_arr.shape:
        raise ValueError(f"batch shape mismatch: {left_arr.shape} != {right_arr.shape}")

    out = np.empty(left_arr.shape, dtype=dtype)
    for idx, left_item in np.ndenumerate(left_arr):
        out[idx] = fn(left_item, right_arr[idx])
    return out


# Keep naming aligned with user request while staying framework-compatible.
UnOp = UnaryOp


class TreeSize(UnOp):
    _raw_specs = [
        (tuple[Formula], int),
        (tuple[list[Formula]], Array[float]),
        (tuple[tuple[Formula]], Array[float]),
        (tuple[Array[Formula]], Array[float]),
    ]

    def _call(
        self, formula: Formula | list[Formula] | tuple[Formula] | np.ndarray
    ) -> int | np.ndarray:
        batch = _apply_unary_batch(formula, lambda f: f.size, np.float64)
        if batch is not None:
            return batch
        return formula.size

    @staticmethod
    def _suitable(output: Any) -> bool:
        if isinstance(output, np.ndarray):
            return output.dtype.kind in ("i", "u", "f") and np.all(output >= 1)
        return isinstance(output, int) and output >= 1

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and output == inp.size


class TreeDepth(UnOp):
    _raw_specs = [
        (tuple[Formula], int),
        (tuple[list[Formula]], Array[float]),
        (tuple[tuple[Formula]], Array[float]),
        (tuple[Array[Formula]], Array[float]),
    ]

    def _call(
        self, formula: Formula | list[Formula] | tuple[Formula] | np.ndarray
    ) -> int | np.ndarray:
        batch = _apply_unary_batch(formula, lambda f: f.depth, np.float64)
        if batch is not None:
            return batch
        return formula.depth

    @staticmethod
    def _suitable(output: Any) -> bool:
        if isinstance(output, np.ndarray):
            return output.dtype.kind in ("i", "u", "f") and np.all(output >= 0)
        return isinstance(output, int) and output >= 0

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and output == inp.depth


class Histogram(dict[str, int]):
    def desc_len(self, mode: str = "use_gaussian") -> float:
        return desc_len(tuple(self.items()), mode=mode)


class NumVariables(UnOp):
    _raw_specs = [
        (tuple[Formula], int),
        (tuple[list[Formula]], Array[float]),
        (tuple[tuple[Formula]], Array[float]),
        (tuple[Array[Formula]], Array[float]),
    ]

    def _call(
        self, formula: Formula | list[Formula] | tuple[Formula] | np.ndarray
    ) -> int | np.ndarray:
        batch = _apply_unary_batch(formula, lambda f: len(f.variables), np.float64)
        if batch is not None:
            return batch
        return len(formula.variables)

    @staticmethod
    def _suitable(output: Any) -> bool:
        if isinstance(output, np.ndarray):
            return output.dtype.kind in ("i", "u", "f") and np.all(output >= 0)
        return isinstance(output, int) and output >= 0

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and output == len(inp.variables)


class NumOperators(UnOp):
    _raw_specs = [
        (tuple[Formula], int),
        (tuple[list[Formula]], Array[float]),
        (tuple[tuple[Formula]], Array[float]),
        (tuple[Array[Formula]], Array[float]),
    ]

    def _call(
        self, formula: Formula | list[Formula] | tuple[Formula] | np.ndarray
    ) -> int | np.ndarray:
        batch = _apply_unary_batch(
            formula, lambda f: sum(f.operator_histogram.values()), np.float64
        )
        if batch is not None:
            return batch
        return sum(formula.operator_histogram.values())

    @staticmethod
    def _suitable(output: Any) -> bool:
        if isinstance(output, np.ndarray):
            return output.dtype.kind in ("i", "u", "f") and np.all(output >= 0)
        return isinstance(output, int) and output >= 0

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and output == sum(
            inp.operator_histogram.values()
        )


class RootOperator(UnOp):
    _raw_specs = [
        (tuple[Formula], str),
        (tuple[list[Formula]], Array[str]),
        (tuple[tuple[Formula]], Array[str]),
        (tuple[Array[Formula]], Array[str]),
    ]

    def _call(
        self, formula: Formula | list[Formula] | tuple[Formula] | np.ndarray
    ) -> str | np.ndarray:
        batch = _apply_unary_batch(formula, lambda f: f.root_operator, object)
        if batch is not None:
            if any(item is None for item in batch.flat):
                raise ValueError("formula has no root operator")
            return batch.astype(str)
        if formula.root_operator is None:
            raise ValueError("formula has no root operator")
        return formula.root_operator

    @staticmethod
    def _suitable(output: Any) -> bool:
        return isinstance(output, str) and len(output) > 0

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and output == inp.root_operator


class OperatorHistogram(UnOp):
    _raw_specs = [
        (tuple[Formula], dict[str, int]),
        (tuple[list[Formula]], Array[dict[str, int]]),
        (tuple[tuple[Formula]], Array[dict[str, int]]),
        (tuple[Array[Formula]], Array[dict[str, int]]),
    ]

    def _call(
        self, formula: Formula | list[Formula] | tuple[Formula] | np.ndarray
    ) -> dict[str, int] | np.ndarray:
        batch = _apply_unary_batch(
            formula, lambda f: dict(f.operator_histogram), object
        )
        if batch is not None:
            return batch
        return Histogram(formula.operator_histogram)

    @staticmethod
    def _properties(*args):
        return {"spec": Histogram}

    @staticmethod
    def _suitable(output: Any) -> bool:
        return isinstance(output, dict) and all(
            isinstance(k, str) and isinstance(v, int) and v >= 0
            for k, v in output.items()
        )

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and output == inp.operator_histogram


class SymbolHistogram(UnOp):
    _raw_specs = [
        (tuple[Formula], dict[str, int]),
        (tuple[list[Formula]], Array[dict[str, int]]),
        (tuple[tuple[Formula]], Array[dict[str, int]]),
        (tuple[Array[Formula]], Array[dict[str, int]]),
    ]

    def _call(
        self, formula: Formula | list[Formula] | tuple[Formula] | np.ndarray
    ) -> dict[str, int] | np.ndarray:
        batch = _apply_unary_batch(formula, lambda f: dict(f.symbol_histogram), object)
        if batch is not None:
            return batch
        return Histogram(formula.symbol_histogram)

    @staticmethod
    def _properties(*args):
        return {"spec": Histogram}

    @staticmethod
    def _suitable(output: Any) -> bool:
        return isinstance(output, dict) and all(
            isinstance(k, str) and isinstance(v, int) and v >= 0
            for k, v in output.items()
        )

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and output == inp.symbol_histogram


class ImplicationChainLength(UnOp):
    _raw_specs = [
        (tuple[Formula], int),
        (tuple[list[Formula]], Array[float]),
        (tuple[tuple[Formula]], Array[float]),
        (tuple[Array[Formula]], Array[float]),
    ]

    def _call(
        self, formula: Formula | list[Formula] | tuple[Formula] | np.ndarray
    ) -> int | np.ndarray:
        batch = _apply_unary_batch(
            formula, lambda f: f.implication_chain_length, np.float64
        )
        if batch is not None:
            return batch
        return formula.implication_chain_length

    @staticmethod
    def _suitable(output: Any) -> bool:
        return isinstance(output, int) and output >= 0

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and output == inp.implication_chain_length


class FormulaEqual(BinOp):
    _raw_specs = [
        (tuple[Formula, Formula], bool),
        (tuple[list[Formula], list[Formula]], Array[bool]),
        (tuple[tuple[Formula], tuple[Formula]], Array[bool]),
        (tuple[Array[Formula], Array[Formula]], Array[bool]),
        (tuple[Formula, list[Formula]], Array[bool]),
        (tuple[list[Formula], Formula], Array[bool]),
        (tuple[Formula, Array[Formula]], Array[bool]),
        (tuple[Array[Formula], Formula], Array[bool]),
    ]
    commutative = True

    def _call(
        self,
        left: Formula | list[Formula] | tuple[Formula] | np.ndarray,
        right: Formula | list[Formula] | tuple[Formula] | np.ndarray,
    ) -> bool | np.ndarray:
        batch = _apply_binary_batch(
            left,
            right,
            lambda left_formula, right_formula: (
                left_formula.canonical_tree == right_formula.canonical_tree
            ),
            np.bool_,
        )
        if batch is not None:
            return batch
        return left.canonical_tree == right.canonical_tree

    @staticmethod
    def _suitable(output: Any) -> bool:
        return isinstance(output, bool)

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and isinstance(output, bool)


class OperatorL1Distance(BinOp):
    _raw_specs = [
        (tuple[Formula, Formula], float),
        (tuple[Array[Formula], Array[Formula]], Array[float]),
    ]

    def _call(self, left: Any, right: Any) -> float | np.ndarray:
        batch = _apply_binary_batch(
            left,
            right,
            lambda a, b: np.abs(
                _histogram_vector(a, b)[0] - _histogram_vector(a, b)[1]
            ).sum(),
            np.float64,
        )
        if batch is not None:
            return batch
        return float(
            np.abs(
                _histogram_vector(left, right)[0] - _histogram_vector(left, right)[1]
            ).sum()
        )


class OperatorCosineSimilarity(BinOp):
    _raw_specs = [
        (tuple[Formula, Formula], float),
        (tuple[Array[Formula], Array[Formula]], Array[float]),
    ]

    @staticmethod
    def _similarity(left: Formula, right: Formula) -> float:
        left_vector, right_vector = _histogram_vector(left, right)
        denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
        return (
            float(np.dot(left_vector, right_vector) / denominator)
            if denominator
            else 0.0
        )

    def _call(self, left: Any, right: Any) -> float | np.ndarray:
        batch = _apply_binary_batch(left, right, self._similarity, np.float64)
        return self._similarity(left, right) if batch is None else batch


class PremiseCount(UnOp):
    _raw_specs = [
        (tuple[Formula], float),
        (tuple[Array[Formula]], Array[float]),
    ]

    @staticmethod
    def _count(formula: Formula) -> float:
        tree = formula.canonical_tree
        if isinstance(tree, tuple) and tree and tree[0] in IMPLICATION_SYMBOLS:
            return float(max(0, len(tree) - 2))
        return 0.0

    def _call(self, formula: Any) -> float | np.ndarray:
        batch = _apply_unary_batch(formula, self._count, np.float64)
        return self._count(formula) if batch is None else batch


class AbsDifference(BinOp):
    _raw_specs = [
        (tuple[int, int], float),
        (tuple[float, float], float),
        (tuple[Array[float], Array[float]], Array[float]),
    ]

    def _call(self, left: Any, right: Any) -> float | np.ndarray:
        batch = _apply_binary_batch(left, right, lambda a, b: abs(a - b), np.float64)
        return abs(left - right) if batch is None else batch


class Ratio(BinOp):
    _raw_specs = [
        (tuple[int, int], float),
        (tuple[float, float], float),
        (tuple[Array[float], Array[float]], Array[float]),
    ]

    @staticmethod
    def _ratio(left: Any, right: Any) -> Any:
        if np.isscalar(right):
            return 0.0 if right == 0 else left / right
        return np.divide(
            left, right, out=np.zeros_like(left, dtype=float), where=right != 0
        )

    def _call(self, left: Any, right: Any) -> float | np.ndarray:
        batch = _apply_binary_batch(left, right, self._ratio, np.float64)
        return self._ratio(left, right) if batch is None else batch


class SkeletonizeFormula(UnOp):
    _raw_specs = [
        (tuple[Formula], Formula),
        (tuple[Array[Formula]], Array[Formula]),
    ]

    def _call(self, formula: Any) -> Formula | np.ndarray:
        if isinstance(formula, np.ndarray):
            return FormulaArray(
                [Formula(_skeleton(item.canonical_tree)) for item in formula]
            )
        return Formula(_skeleton(formula.canonical_tree))


class SubtreeSet(UnOp):
    _raw_specs = [
        (tuple[Formula], set[str]),
        (tuple[Array[Formula]], Array[str]),
    ]

    def _call(self, formula: Any) -> set[str] | np.ndarray:
        if isinstance(formula, np.ndarray):
            return np.asarray(
                [str(sorted(_subtrees(item.canonical_tree))) for item in formula],
                dtype=str,
            )
        return {str(tree) for tree in _subtrees(formula.canonical_tree)}


class SubtreeJaccard(BinOp):
    _raw_specs = [
        (tuple[Formula, Formula], float),
        (tuple[Array[Formula], Array[Formula]], Array[float]),
    ]

    @staticmethod
    def _similarity(left: Formula, right: Formula) -> float:
        left_subtrees = _subtrees(left.canonical_tree)
        right_subtrees = _subtrees(right.canonical_tree)
        union = left_subtrees | right_subtrees
        return float(len(left_subtrees & right_subtrees) / len(union)) if union else 1.0

    def _call(self, left: Any, right: Any) -> float | np.ndarray:
        batch = _apply_binary_batch(left, right, self._similarity, np.float64)
        return self._similarity(left, right) if batch is None else batch


class MaxSharedSubtreeSize(BinOp):
    _raw_specs = [
        (tuple[Formula, Formula], float),
        (tuple[Array[Formula], Array[Formula]], Array[float]),
    ]

    @staticmethod
    def _size(tree: FrozenNode) -> int:
        if not isinstance(tree, tuple):
            return 1
        return 1 + sum(MaxSharedSubtreeSize._size(child) for child in tree)

    @classmethod
    def _similarity(cls, left: Formula, right: Formula) -> float:
        shared = _subtrees(left.canonical_tree) & _subtrees(right.canonical_tree)
        return float(max((cls._size(tree) for tree in shared), default=0))

    def _call(self, left: Any, right: Any) -> float | np.ndarray:
        batch = _apply_binary_batch(left, right, self._similarity, np.float64)
        return self._similarity(left, right) if batch is None else batch


class SharedVariables(BinOp):
    _raw_specs = [
        (tuple[Formula, Formula], int),
        (tuple[list[Formula], list[Formula]], Array[float]),
        (tuple[tuple[Formula], tuple[Formula]], Array[float]),
        (tuple[Array[Formula], Array[Formula]], Array[float]),
        (tuple[Formula, list[Formula]], Array[float]),
        (tuple[list[Formula], Formula], Array[float]),
        (tuple[Formula, Array[Formula]], Array[float]),
        (tuple[Array[Formula], Formula], Array[float]),
    ]
    commutative = True

    def _call(
        self,
        left: Formula | list[Formula] | tuple[Formula] | np.ndarray,
        right: Formula | list[Formula] | tuple[Formula] | np.ndarray,
    ) -> int | np.ndarray:
        batch = _apply_binary_batch(
            left,
            right,
            lambda left_formula, right_formula: len(
                left_formula.variables & right_formula.variables
            ),
            np.float64,
        )
        if batch is not None:
            return batch
        return len(left.variables & right.variables)

    @staticmethod
    def _suitable(output: Any) -> bool:
        return isinstance(output, int) and output >= 0

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and isinstance(output, int) and output >= 0


class SharedOperators(BinOp):
    _raw_specs = [
        (tuple[Formula, Formula], int),
        (tuple[list[Formula], list[Formula]], Array[float]),
        (tuple[tuple[Formula], tuple[Formula]], Array[float]),
        (tuple[Array[Formula], Array[Formula]], Array[float]),
        (tuple[Formula, list[Formula]], Array[float]),
        (tuple[list[Formula], Formula], Array[float]),
        (tuple[Formula, Array[Formula]], Array[float]),
        (tuple[Array[Formula], Formula], Array[float]),
    ]
    commutative = True

    def _call(
        self,
        left: Formula | list[Formula] | tuple[Formula] | np.ndarray,
        right: Formula | list[Formula] | tuple[Formula] | np.ndarray,
    ) -> int | np.ndarray:
        batch = _apply_binary_batch(
            left,
            right,
            lambda left_formula, right_formula: len(
                left_formula.operators & right_formula.operators
            ),
            np.float64,
        )
        if batch is not None:
            return batch
        return len(left.operators & right.operators)

    @staticmethod
    def _suitable(output: Any) -> bool:
        return isinstance(output, int) and output >= 0

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and isinstance(output, int) and output >= 0


class SharedSymbols(BinOp):
    _raw_specs = [
        (tuple[Formula, Formula], int),
        (tuple[list[Formula], list[Formula]], Array[float]),
        (tuple[tuple[Formula], tuple[Formula]], Array[float]),
        (tuple[Array[Formula], Array[Formula]], Array[float]),
        (tuple[Formula, list[Formula]], Array[float]),
        (tuple[list[Formula], Formula], Array[float]),
        (tuple[Formula, Array[Formula]], Array[float]),
        (tuple[Array[Formula], Formula], Array[float]),
    ]
    commutative = True

    def _call(
        self,
        left: Formula | list[Formula] | tuple[Formula] | np.ndarray,
        right: Formula | list[Formula] | tuple[Formula] | np.ndarray,
    ) -> int | np.ndarray:
        batch = _apply_binary_batch(
            left,
            right,
            lambda left_formula, right_formula: len(
                left_formula.symbols & right_formula.symbols
            ),
            np.float64,
        )
        if batch is not None:
            return batch
        return len(left.symbols & right.symbols)

    @staticmethod
    def _suitable(output: Any) -> bool:
        return isinstance(output, int) and output >= 0

    @staticmethod
    def check(inp: Formula, output: Any) -> bool:
        return isinstance(inp, Formula) and isinstance(output, int) and output >= 0


__all__ = [
    "Formula",
    "FormulaArray",
    "OperatorL1Distance",
    "OperatorCosineSimilarity",
    "PremiseCount",
    "AbsDifference",
    "Ratio",
    "SkeletonizeFormula",
    "SubtreeSet",
    "SubtreeJaccard",
    "MaxSharedSubtreeSize",
    "UnOp",
    "TreeSize",
    "TreeDepth",
    "NumVariables",
    "NumOperators",
    "RootOperator",
    "OperatorHistogram",
    "SymbolHistogram",
    "ImplicationChainLength",
    "FormulaEqual",
    "SharedVariables",
    "SharedOperators",
    "SharedSymbols",
]
