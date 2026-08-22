"""Utility functions for symbolic MeTTa formulas."""

from __future__ import annotations

Expr = list["ExprNode"]
ExprNode = str | int | float | bool | None | Expr

# Current dataset uses Greek letters as variables.
# If future datasets introduce different variable naming conventions,
# this table is the only place that needs adaptation.
GREEK_CHARS = set("αβγδεζηθικλμνξοπρστυφχψω𝛼𝛽𝛾𝛿𝜀𝜁𝜂𝜃𝜄𝜅𝜆𝜇𝜈𝜉𝜊𝜋𝜌𝜎𝜏𝜐𝜑𝜒𝜓𝜔𝛢𝛣𝛤𝛥𝛦𝛧𝛨𝛩𝛪𝛫𝛬𝛭𝛮𝛯𝛰𝛱𝛲𝛴𝛵𝛶𝛷𝛸𝛹𝛺")


def is_variable(symbol: object) -> bool:
    """Return ``True`` iff ``symbol`` is a variable token."""
    return isinstance(symbol, str) and len(symbol) > 0 and symbol[0] in GREEK_CHARS


def canonicalize(expr: ExprNode) -> ExprNode:
    """Rename variables to ``v0``, ``v1``, ... in first-occurrence order."""
    mapping: dict[str, str] = {}
    next_id = 0

    def visit(node: ExprNode) -> ExprNode:
        nonlocal next_id
        if isinstance(node, list):
            return [visit(x) for x in node]

        if is_variable(node):
            token = str(node)
            if token not in mapping:
                mapping[token] = f"v{next_id}"
                next_id += 1
            return mapping[token]

        return node

    return visit(expr)


def tree_size(expr: ExprNode) -> int:
    """Return the number of nodes in the nested expression tree."""
    if not isinstance(expr, list):
        return 1
    return 1 + sum(tree_size(x) for x in expr)


def tree_depth(expr: ExprNode) -> int:
    """Return the maximum list nesting depth in the expression tree."""
    if not isinstance(expr, list):
        return 0
    if not expr:
        return 1
    return 1 + max(tree_depth(x) for x in expr)


def variables(expr: ExprNode) -> set[str]:
    """Return all variable symbols occurring in ``expr``."""
    result: set[str] = set()

    def visit(node: ExprNode) -> None:
        if isinstance(node, list):
            for x in node:
                visit(x)
        elif is_variable(node):
            result.add(str(node))

    visit(expr)
    return result


def num_variables(expr: ExprNode) -> int:
    """Return the number of distinct variables in ``expr``."""
    return len(variables(expr))


__all__ = [
    "Expr",
    "ExprNode",
    "GREEK_CHARS",
    "is_variable",
    "canonicalize",
    "tree_size",
    "tree_depth",
    "variables",
    "num_variables",
]
