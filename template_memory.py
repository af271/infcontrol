"""Typed, input-role-preserving feature templates for MORK memory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Any

from infcontrol.operators import Formula
from william.composite import Composite
from william.library.base import Operator, Value
from william.library.types import Array, tuple_args
from william.mork_memory import MorkTemplateMemory
from william.structures import Graph, ValueNode
from william.structures.nodes import graph_element
from william.structures.sexpr_to_graph import build_graph, parse_sexpr


def _sexpr(expression: Any) -> str:
    if isinstance(expression, str):
        return expression
    if not isinstance(expression, list):
        raise ValueError("Invalid feature-template expression.")
    return "(" + " ".join(_sexpr(item) for item in expression) + ")"


def feature_record(root: ValueNode) -> str:
    roles = [leaf.output.name for leaf in root.leaves()]
    if not roles or any(role not in ("goal", "candidate", "one") for role in roles):
        raise ValueError(f"Unsupported feature-template input roles: {roles!r}")
    program = _encode_program(parse_sexpr(root.to_sexpr()))
    return f"(hyperon_feature_v2 (roles {' '.join(roles)}) {_sexpr(program)})"


def _encode_program(expression: Any) -> Any:
    if isinstance(expression, list):
        return [_encode_program(item) for item in expression]
    if expression == "$":
        return "william_bind"
    if isinstance(expression, str) and re.fullmatch(r"_\d+", expression):
        return f"william_ref{expression[1:]}"
    return expression


def _decode_program(expression: Any) -> Any:
    if isinstance(expression, list):
        return [_decode_program(item) for item in expression]
    if expression == "william_bind":
        return "$"
    if isinstance(expression, str) and re.fullmatch(r"william_ref\d+", expression):
        return f"_{expression[len('william_ref') :]}"
    return expression


@dataclass(frozen=True)
class FeatureTemplate:
    record: str
    count: int
    graph: Graph
    operator: Composite


def program_info(graph: Graph) -> dict[str, Any]:
    root = list(graph.roots())[0].clone_and_unpack()
    return {
        "record": feature_record(root),
        "operator_count": len(list(root.walk(val_nodes=False))),
    }


def primitive_templates(operators: list[Operator]) -> list[FeatureTemplate]:
    """Construct a fixed, data-independent prior over directly usable primitives."""
    roles_by_type = {Array[Formula]: ("goal", "candidate"), float: ("one",)}
    result = {}
    for operator in sorted(operators, key=lambda item: item.name):
        for input_spec, output_spec in operator.specs:
            inputs = tuple_args(input_spec)
            if output_spec not in (Array[int], Array[float], Array[bool]) or not inputs:
                continue
            if any(spec not in roles_by_type for spec in inputs):
                continue
            for roles in product(*(roles_by_type[spec] for spec in inputs)):
                root = graph_element(operator, inputs, output_spec)
                for leaf, role in zip(root.leaves(), roles, strict=True):
                    leaf.output = Value(None, spec=leaf.output.spec, name=role, permeable=role == "one")
                record = feature_record(root)
                result[record] = FeatureTemplate(record, 1, Graph([root]), Composite(root, name=f"prior_{len(result)}"))
    return sorted(result.values(), key=lambda item: item.record)


def load_templates(
    memory: MorkTemplateMemory, operators: list[Operator], min_operators: int = 1
) -> list[FeatureTemplate]:
    registry = {operator.name: operator for operator in operators}
    result = []
    for index, (record, count) in enumerate(memory.load()):
        parsed = parse_sexpr(record)
        if (
            not isinstance(parsed, list)
            or len(parsed) != 3
            or parsed[0] not in ("hyperon_feature_v1", "hyperon_feature_v2")
            or not isinstance(parsed[1], list)
            or not parsed[1]
            or parsed[1][0] != "roles"
        ):
            raise ValueError(f"Invalid Hyperon template record: {record!r}")
        roles = parsed[1][1:]
        program = _decode_program(parsed[2]) if parsed[0] == "hyperon_feature_v2" else parsed[2]
        root = build_graph(_sexpr(program), operators=registry, types={"F": Formula})
        leaves = list(root.leaves())
        if len(roles) != len(leaves) or any(role not in ("goal", "candidate", "one") for role in roles):
            raise ValueError(f"Invalid Hyperon template roles: {roles!r}")
        if not root.options:
            raise ValueError("A Hyperon feature template must contain an operator.")
        for leaf, role in zip(leaves, roles, strict=True):
            expected = float if role == "one" else Array[Formula]
            if leaf.output.spec != expected:
                raise ValueError(f"Template input role {role!r} has incompatible type {leaf.output.spec!r}.")
            leaf.output = Value(None, spec=leaf.output.spec, name=role, permeable=role == "one")
        if len(list(root.walk(val_nodes=False))) >= min_operators:
            result.append(FeatureTemplate(record, count, Graph([root]), Composite(root, name=f"mork_{index}")))
    return sorted(result, key=lambda item: (-item.count, item.record))
