import csv
import json
import os
from collections import Counter
from uuid import uuid4

import numpy as np
import pytest

from infcontrol.inference_control import InferenceControl
from infcontrol.model import PredictionModel, evaluate_graph
from infcontrol.mork_benchmark import compare, split_trace
from infcontrol.operators import Formula, SharedOperators, SkeletonizeFormula, TreeSize
from infcontrol.template_memory import feature_record, load_templates, primitive_templates, program_info
from william.library.base import Value
from william.library.basic_ops import Equal, LessThan
from william.library.types import Array
from william.structures.sexpr_to_graph import build_graph


class Memory:
    stores = {}

    def __init__(self, url, namespace, *, erase=False):
        self.namespace = namespace
        if erase or namespace not in self.stores:
            self.stores[namespace] = Counter()
        self.generation = "test"
        self.store = self.stores[namespace]

    def load(self):
        return sorted(self.store.items())

    def remember(self, records):
        self.store.update(set(records))


@pytest.fixture(params=["fixture", "mork"])
def memory_url(request, monkeypatch):
    if request.param == "mork":
        url = os.environ.get("WILLIAM_MORK_TEST_URL")
        if not url:
            pytest.skip("Set WILLIAM_MORK_TEST_URL for real MORK integration.")
        return url
    monkeypatch.setattr("infcontrol.inference_control.MorkTemplateMemory", Memory)
    return "fixture"


def write_trace(path, groups=12):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["goal", "candidate", "used", "theorem_index"])
        writer.writeheader()
        for index in range(groups):
            goal = ["->", f"g{index}", "b"]
            for candidate, used in ((goal, True), (["other", f"c{index}"], False)):
                writer.writerow(
                    {
                        "goal": json.dumps(goal),
                        "candidate": json.dumps(candidate),
                        "used": str(used),
                        "theorem_index": str(index),
                    }
                )


def test_selected_template_survives_restart_as_standalone_model(tmp_path, memory_url, monkeypatch):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    namespace = uuid4().hex
    options = dict(
        operators=[SharedOperators, LessThan],
        use_context=False,
        validation_fraction=0.2,
        test_fraction=0,
        mork_url=memory_url,
        memory_namespace=namespace,
        fitter_kwargs={"objective": "log_loss", "acceptance_metric": "log_loss"},
    )
    seed = InferenceControl(trace, max_popped=30, erase_memory=True, **options)
    list(seed.run())
    assert seed.accepted_graphs
    warm = InferenceControl(trace, max_popped=1, template_strategy="eager", **options)
    list(warm.run())
    assert warm.candidate_attempts == 1
    assert warm.accepted_template_sources[0]
    output = warm.save(tmp_path / "model")
    provenance = json.loads((output / "template_memory.json").read_text())
    assert provenance["features"][0]["source_templates"] == warm.accepted_template_sources[0]

    def disconnected(*args, **kwargs):
        pytest.fail("Saved predictor must work without MORK.")

    monkeypatch.setattr("william.mork_memory.urlopen", disconnected)
    restored = PredictionModel.load(output, operators=[SharedOperators, LessThan])
    np.testing.assert_allclose(
        restored.predict(warm.features[:, 0], warm.features[:, 1]),
        warm.predict(warm.features[:, 0], warm.features[:, 1]),
    )
    assert all(node.op.primitive for graph in restored.graphs for node in graph.walk(val_nodes=False))


def test_formula_dag_and_input_roles_roundtrip():
    registry = {op.name: op for op in (TreeSize(dl=8), Equal(dl=8))}
    root = build_graph(
        "((= $ Array[F]) (Array[bool] (equal (Array[float] (treesize _1)) (Array[float] (treesize _1)))))",
        operators=registry,
        types={"F": Formula},
    )
    for leaf in root.leaves():
        leaf.output = Value(None, spec=Array[Formula], name="goal", permeable=False)
    memory = Memory("", uuid4().hex)
    memory.remember([feature_record(root)])
    template = load_templates(memory, list(registry.values()))[0]
    assert feature_record(template.graph.nodes[0]) == feature_record(root)
    assert len(list(template.graph.leaves())) == 1


def test_templates_preserve_noncommutative_goal_candidate_binding():
    registry = {op.name: op for op in (TreeSize(dl=8), LessThan(dl=8))}
    root = build_graph(
        "(Array[bool] (lessthan (Array[float] (treesize Array[F])) (Array[float] (treesize Array[F]))))",
        operators=registry,
        types={"F": Formula},
    )
    for leaf, role in zip(root.leaves(), ("goal", "candidate"), strict=True):
        leaf.output = Value(None, spec=Array[Formula], name=role, permeable=False)
    memory = Memory("", uuid4().hex)
    memory.remember([feature_record(root)])
    template = load_templates(memory, list(registry.values()))[0]
    from infcontrol.model import formula_array
    from william.propagation_py import RandomPropagation

    result = evaluate_graph(
        template.graph,
        RandomPropagation(),
        formula_array(["a"]),
        formula_array([["b", "c"]]),
    )
    np.testing.assert_array_equal(result, [1])


def test_external_type_registry_is_explicit_and_rejects_overrides():
    registry = {"treesize": TreeSize(dl=8)}
    with pytest.raises(ValueError, match="Unknown template"):
        build_graph("(Array[float] (treesize Array[F]))", operators=registry)
    with pytest.raises(ValueError, match="override"):
        build_graph("(Array[float] (treesize Array[F]))", operators=registry, types={"int": Formula})


def test_unsupported_template_roles_fail_explicitly():
    memory = Memory("", uuid4().hex)
    memory.remember(["(hyperon_feature_v1 (roles candidate) (Array[float] (treesize int)))"])
    with pytest.raises(ValueError, match="incompatible"):
        load_templates(memory, [TreeSize(dl=8)])


def test_non_numeric_templates_are_available_for_search_not_direct_fit(tmp_path, monkeypatch):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    namespace = uuid4().hex
    memory = Memory("", namespace)
    memory.remember(["(hyperon_feature_v1 (roles goal) (Array[F] (skeletonizeformula Array[F])))"])
    monkeypatch.setattr("infcontrol.inference_control.MorkTemplateMemory", Memory)
    controller = InferenceControl(
        trace,
        operators=[SkeletonizeFormula],
        max_popped=0,
        mork_url="fixture",
        memory_namespace=namespace,
    )
    assert list(controller.run()) == []
    assert controller.template_attempts == 0


def test_chronological_comparison_never_trains_on_test_rows(tmp_path, monkeypatch):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    monkeypatch.setattr("infcontrol.inference_control.MorkTemplateMemory", Memory)
    from infcontrol import mork_benchmark

    original = mork_benchmark.InferenceControl
    seen = []

    def controller(path, **kwargs):
        seen.append([row["theorem_index"] for row in original._read_rows(path)])
        return original(
            path,
            operators=[SharedOperators, LessThan],
            fitter_kwargs={"objective": "log_loss", "acceptance_metric": "log_loss"},
            **kwargs,
        )

    monkeypatch.setattr(mork_benchmark, "InferenceControl", controller)
    controller._read_rows = original._read_rows
    controller._arrays_from_rows = original._arrays_from_rows
    report = compare(
        trace,
        tmp_path / "results",
        "fixture",
        uuid4().hex,
        seed_groups=4,
        train_groups=4,
        test_groups=4,
        max_popped=30,
        use_context=False,
    )
    assert [set(groups) for groups in seen] == [{"0", "1", "2", "3"}, {"4", "5", "6", "7"}, {"4", "5", "6", "7"}]
    assert report["memory_reused"]
    assert report["warm"]["candidate_attempts"] <= 30
    assert report["stages"]["test"]["first_theorem_index"] == 8
    assert (tmp_path / "results" / "comparison.json").is_file()


def test_split_rejects_unordered_or_short_traces(tmp_path):
    trace = tmp_path / "trace.csv"
    write_trace(trace, groups=4)
    with pytest.raises(ValueError, match="Need 6"):
        split_trace(trace, tmp_path / "out", 2, 2, 2)
    lines = trace.read_text().splitlines()
    trace.write_text("\n".join([lines[0], *reversed(lines[1:])]))
    with pytest.raises(ValueError, match="chronologically"):
        split_trace(trace, tmp_path / "out", 2, 2, 2)


def test_plain_save_clears_stale_memory_provenance(tmp_path):
    from infcontrol.model import save_model

    (tmp_path / "template_memory.json").write_text('{"features": []}')
    save_model(tmp_path, [], 0, [])
    assert not (tmp_path / "template_memory.json").exists()


def test_primitive_prior_is_fixed_and_contains_only_single_operator_programs():
    operators = [SharedOperators(dl=8), TreeSize(dl=8), LessThan(dl=8)]
    templates = primitive_templates(operators)
    assert templates
    assert [item.record for item in templates] == [
        item.record for item in primitive_templates(list(reversed(operators)))
    ]
    assert all(program_info(item.graph)["operator_count"] == 1 for item in templates)
    assert any("(roles candidate)" in item.record for item in templates)


def test_filter_excludes_primitive_memory_but_keeps_composed_programs():
    memory = Memory("", uuid4().hex)
    memory.remember(
        [
            "(hyperon_feature_v1 (roles goal) (Array[float] (treesize Array[F])))",
            "(hyperon_feature_v1 (roles goal one) (Array[bool] (lessthan (Array[float] (treesize Array[F])) float)))",
        ]
    )
    templates = load_templates(memory, [TreeSize(dl=8), LessThan(dl=8)], min_operators=2)
    assert len(templates) == 1
    assert program_info(templates[0].graph)["operator_count"] == 2


def test_training_deadline_includes_initialization_and_stops_before_search(tmp_path, monkeypatch):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    clock = [0.0]
    monkeypatch.setattr("infcontrol.inference_control.time", lambda: clock[0])
    controller = InferenceControl(trace, max_seconds=1, primitive_prior=True)
    clock[0] = 2
    assert list(controller.run()) == []
    assert controller.candidate_attempts == 0


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_training_deadlines_fail_before_loading_csv(tmp_path, seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        InferenceControl(tmp_path / "missing.csv", max_seconds=seconds)


def test_partial_features_can_ignore_unused_goal_and_constant(tmp_path):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    options = dict(
        operators=[SharedOperators],
        max_popped=30,
        use_context=False,
        fitter_kwargs={"objective": "log_loss", "acceptance_metric": "log_loss"},
        audit_candidates=True,
    )
    original = InferenceControl(trace, **options)
    assert list(original.run()) == []
    controller = InferenceControl(trace, allow_partial_features=True, **options)
    assert list(controller.run())
    assert controller.candidate_programs
    assert all(len(list(graph.roots())) == 1 for graph in controller.accepted_graphs)
    assert all(item["within_time_budget"] for item in controller.accepted_programs)


def test_primitive_control_is_chronological_and_has_no_memory(tmp_path, monkeypatch):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    monkeypatch.setattr("infcontrol.inference_control.MorkTemplateMemory", Memory)
    report = compare(
        trace,
        tmp_path / "comparison",
        "fixture",
        uuid4().hex,
        seed_groups=4,
        train_groups=4,
        test_groups=4,
        max_popped=30,
        primitive_control=True,
        allow_partial_features=True,
        use_context=False,
        min_template_operators=2,
    )
    assert report["primitive"]["loaded_templates"] == 0
    assert report["primitive"]["accepted_features_from_memory"] == 0
    assert report["stages"]["test"]["first_theorem_index"] == 8
    assert all(item["operator_count"] >= 2 for item in report["transferred_programs"])


def test_nested_shared_program_roundtrips_through_memory(memory_url):
    from infcontrol import inference_control
    from infcontrol.model import formula_array
    from infcontrol.operators import AbsDifference, Ratio
    from william.library.complex_ops import Sqrt
    from william.propagation_py import RandomPropagation
    from william.structures import Graph

    operators = [TreeSize(dl=8), Sqrt(dl=8), Ratio(dl=8), AbsDifference(dl=8)]
    root = build_graph(
        "((= $ (Array[float] (treesize Array[F]))) "
        "(= $ (Array[float] (sqrt _1))) "
        "(Array[float] (absdifference _2 (Array[float] (ratio _1 _2)))))",
        operators={op.name: op for op in operators},
        types={"F": Formula},
    )
    for leaf in root.leaves():
        leaf.output = Value(None, spec=Array[Formula], name="candidate", permeable=False)
    record = feature_record(root)
    assert "william_ref" in record
    assert "$" not in record
    namespace = uuid4().hex
    memory = inference_control.MorkTemplateMemory(memory_url, namespace, erase=True)
    memory.remember([record])
    templates = load_templates(memory, operators, min_operators=2)
    assert len(templates) == 1
    assert feature_record(templates[0].graph.nodes[0]) == record
    inputs = formula_array(["a", ["->", "a", "b"], ["->", ["->", "a", "b"], "c"]])
    np.testing.assert_allclose(
        evaluate_graph(templates[0].graph, RandomPropagation(), inputs, inputs),
        evaluate_graph(Graph([root]), RandomPropagation(), inputs, inputs),
    )
    assert program_info(templates[0].graph)["operator_count"] == 4


def test_deadline_is_forwarded_to_wunderbaum(tmp_path, monkeypatch):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    monkeypatch.setattr("infcontrol.inference_control.time", lambda: 100)
    controller = InferenceControl(trace, max_seconds=1, use_context=False)
    seen = {}

    def iterate(*args, **kwargs):
        seen.update(kwargs)
        return iter(())

    monkeypatch.setattr(controller.enum, "iterate", iterate)
    assert list(controller.run()) == []
    assert seen["deadline_ts"] == 101


def test_late_programs_are_not_marked_as_found_within_budget(tmp_path, monkeypatch):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    clock = [0]
    monkeypatch.setattr("infcontrol.inference_control.perf_counter", lambda: clock[0])
    controller = InferenceControl(trace, max_seconds=1, audit_candidates=True)
    template = primitive_templates([TreeSize(dl=8)])[0]
    clock[0] = 2
    controller._audit_graph(template.graph, accepted=True)
    assert not controller.accepted_programs[0]["within_time_budget"]


def test_later_chronological_window_skips_complete_groups(tmp_path):
    trace = tmp_path / "trace.csv"
    write_trace(trace, groups=16)
    paths = split_trace(trace, tmp_path / "stages", 4, 4, 4, skip_groups=4)
    assert {row["theorem_index"] for row in InferenceControl._read_rows(paths["seed"])} == {"4", "5", "6", "7"}
    assert {row["theorem_index"] for row in InferenceControl._read_rows(paths["test"])} == {"12", "13", "14", "15"}
    with pytest.raises(ValueError, match="skip_groups"):
        split_trace(trace, tmp_path / "invalid", 4, 4, 4, skip_groups=-1)


def test_general_operator_library_excludes_batch_reducers_and_exact_equality():
    from infcontrol.inference_control import GENERAL_OPERATORS

    names = {operator().name for operator in GENERAL_OPERATORS}
    assert not names & {"mean", "sum", "len", "equal", "sqrt", "maxsharedsubtreesize", "subtreejaccard"}


def test_dl_memory_uses_normal_heap_without_eager_fitting(tmp_path, memory_url, monkeypatch):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    from infcontrol import inference_control

    namespace = uuid4().hex
    memory = inference_control.MorkTemplateMemory(memory_url, namespace, erase=True)
    record = "(hyperon_feature_v2 (roles goal candidate) (Array[float] (sharedoperators Array[F] Array[F])))"
    memory.remember([record])
    controller = InferenceControl(
        trace,
        operators=[SharedOperators, LessThan],
        max_popped=200,
        use_context=False,
        allow_partial_features=True,
        audit_candidates=True,
        mork_url=memory_url,
        memory_namespace=namespace,
        remember_templates=False,
        fitter_kwargs={"objective": "log_loss", "acceptance_metric": "log_loss"},
    )
    before = memory.load()
    assert controller.template_strategy == "dl"
    assert controller.loaded_templates
    assert controller.initial_templates == []

    def no_eager_evaluation(*args, **kwargs):
        pytest.fail("DL mode must not run a separate template-fitting pass.")

    monkeypatch.setattr("infcontrol.inference_control.evaluate_graph", no_eager_evaluation)
    list(controller.run())
    assert controller.template_attempts == 0
    assert controller.candidate_attempts == controller.enum.num_popped
    assert controller.accepted_graphs
    assert any(controller.accepted_template_sources)
    assert memory.load() == before


@pytest.mark.parametrize("reverse", [False, True])
def test_eager_and_dl_comparison_uses_identical_frozen_memory(tmp_path, monkeypatch, reverse):
    trace = tmp_path / "trace.csv"
    write_trace(trace)
    monkeypatch.setattr("infcontrol.inference_control.MorkTemplateMemory", Memory)
    from infcontrol import mork_benchmark

    def controller(path, **kwargs):
        return InferenceControl(
            path,
            operators=[SharedOperators, LessThan],
            fitter_kwargs={"objective": "log_loss", "acceptance_metric": "log_loss"},
            **kwargs,
        )

    controller._read_rows = InferenceControl._read_rows
    controller._arrays_from_rows = InferenceControl._arrays_from_rows
    monkeypatch.setattr(mork_benchmark, "InferenceControl", controller)
    report = compare(
        trace,
        tmp_path / "results",
        "fixture",
        uuid4().hex,
        seed_groups=4,
        train_groups=4,
        test_groups=4,
        max_popped=100,
        use_context=False,
        allow_partial_features=True,
        eager_control=True,
        reverse_order=reverse,
    )
    assert report["phase_order"] == (
        ["seed", "warm", "eager", "cold"] if reverse else ["seed", "cold", "eager", "warm"]
    )
    assert report["warm"]["template_attempts"] == 0
    assert report["warm"]["template_strategy"] == "dl"
    assert report["eager"]["template_attempts"] > 0
    assert report["warm"]["memory_snapshot"] == report["eager"]["memory_snapshot"]
    assert report["warm"]["memory_snapshot_sha256"] == report["eager"]["memory_snapshot_sha256"]
    for phase in ("cold", "eager", "warm"):
        assert report[phase]["candidate_attempts"] <= 100
