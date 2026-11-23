"""Chronological seed/cold/warm feature-training comparison on Hyperon traces."""

from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from infcontrol.inference_control import GENERAL_OPERATORS, InferenceControl
from infcontrol.model import PredictionModel


def split_trace(
    csv_path: Path,
    output: Path,
    seed_groups: int,
    train_groups: int,
    test_groups: int,
    skip_groups: int = 0,
) -> dict[str, Path]:
    if min(seed_groups, train_groups, test_groups) < 2:
        raise ValueError("Each chronological stage needs at least two theorem groups.")
    if skip_groups < 0:
        raise ValueError("skip_groups must be non-negative.")
    total = skip_groups + seed_groups + train_groups + test_groups
    grouped: dict[int, list[dict[str, str]]] = {}
    previous = -1
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "theorem_index" not in reader.fieldnames:
            raise ValueError("Chronological MORK comparison requires theorem_index.")
        fields = reader.fieldnames
        for row in reader:
            index = float(row["theorem_index"])
            if not np.isfinite(index) or index < 0 or not index.is_integer():
                raise ValueError(f"Invalid theorem_index: {row['theorem_index']!r}")
            index = int(index)
            if index < previous:
                raise ValueError("Trace rows must be ordered chronologically by theorem_index.")
            if index != previous and len(grouped) == total:
                break
            grouped.setdefault(index, []).append(row)
            previous = index
    if len(grouped) < total:
        raise ValueError(f"Need {total} complete theorem groups, got {len(grouped)}.")
    output.mkdir(parents=True, exist_ok=True)
    items = list(grouped.values())
    paths = {}
    offset = skip_groups
    for stage, count in (("seed", seed_groups), ("train", train_groups), ("test", test_groups)):
        path = output / f"{stage}.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for group in items[offset : offset + count]:
                writer.writerows(group)
        offset += count
        paths[stage] = path
    return paths


def evaluate_test(model: PredictionModel, csv_path: Path) -> dict[str, float]:
    rows = InferenceControl._read_rows(csv_path)
    features, labels = InferenceControl._arrays_from_rows(rows)
    context = {name: np.asarray([float(row[name]) for row in rows]) for name in model.context_feature_names}
    probabilities = model.predict(features[:, 0], features[:, 1], context)
    clipped = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["theorem_index"], []).append(index)
    ranks = []
    recall5 = []
    for indices in groups.values():
        positives = sum(labels[index] for index in indices)
        if not positives:
            continue
        ranked = sorted(indices, key=lambda index: probabilities[index], reverse=True)
        ranks.append(max(rank for rank, index in enumerate(ranked, 1) if labels[index]))
        recall5.append(float(sum(labels[index] for index in ranked[:5]) / positives))
    if not ranks:
        raise ValueError("Held-out test groups contain no positive proof labels.")
    return {
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
        "k_at_full_recall": float(np.mean(ranks)),
        "recall_at_5": float(np.mean(recall5)),
        "all_at_5": float(np.mean(np.asarray(ranks) <= 5)),
    }


def compare(
    csv_path: Path,
    output: Path,
    mork_url: str,
    namespace: str,
    *,
    seed_groups: int = 30,
    train_groups: int = 30,
    test_groups: int = 30,
    max_popped: int = 3000,
    use_context: bool = True,
    max_seconds: float | None = None,
    seed_seconds: float | None = None,
    seed_max_popped: int | None = None,
    min_template_operators: int = 1,
    primitive_control: bool = False,
    operator_set: str = "full",
    allow_partial_features: bool = False,
    skip_groups: int = 0,
    eager_control: bool = False,
    reverse_order: bool = False,
) -> dict[str, Any]:
    if max_popped <= 0:
        raise ValueError("max_popped must be positive.")
    if seed_max_popped is not None and seed_max_popped <= 0:
        raise ValueError("seed_max_popped must be positive.")
    for seconds in (max_seconds, seed_seconds):
        if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
            raise ValueError("Time budgets must be finite and positive.")
    if operator_set not in ("full", "general"):
        raise ValueError("operator_set must be 'full' or 'general'.")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new, empty output directory to preserve previous benchmark results.")
    stages = split_trace(csv_path, output, seed_groups, train_groups, test_groups, skip_groups)
    report: dict[str, Any] = {
        "source_csv": str(csv_path.resolve()),
        "max_popped": max_popped,
        "use_context": use_context,
        "max_seconds": max_seconds,
        "seed_seconds": seed_seconds,
        "seed_max_popped": seed_max_popped,
        "min_template_operators": min_template_operators,
        "operator_set": operator_set,
        "allow_partial_features": allow_partial_features,
        "skip_groups": skip_groups,
        "template_strategy": "dl",
        "stages": {},
    }
    for name, path in stages.items():
        rows = InferenceControl._read_rows(path)
        report["stages"][name] = {
            "groups": len({row["theorem_index"] for row in rows}),
            "first_theorem_index": int(float(rows[0]["theorem_index"])),
            "last_theorem_index": int(float(rows[-1]["theorem_index"])),
        }
        report["stages"][name]["goals"] = [
            json.loads(next(row["goal"] for row in rows if row["theorem_index"] == group))
            for group in dict.fromkeys(row["theorem_index"] for row in rows)
        ]
    later_phases = [
        "cold",
        *(["primitive"] if primitive_control else []),
        *(["eager"] if eager_control else []),
        "warm",
    ]
    if reverse_order:
        later_phases.reverse()
    phases = ["seed", *later_phases]
    report["phase_order"] = phases
    phase_sources = {}
    for phase in phases:
        t0 = perf_counter()
        controller = InferenceControl(
            stages["seed" if phase == "seed" else "train"],
            max_popped=seed_max_popped if phase == "seed" and seed_max_popped is not None else max_popped,
            validation_fraction=0.2,
            test_fraction=0,
            use_context=use_context,
            mork_url=mork_url if phase in ("seed", "warm", "eager") else None,
            memory_namespace=namespace,
            erase_memory=phase == "seed",
            max_seconds=(seed_seconds if seed_seconds is not None else max_seconds) if phase == "seed" else max_seconds,
            min_template_operators=1 if phase == "seed" else min_template_operators,
            primitive_prior=phase == "primitive",
            audit_candidates=True,
            allow_partial_features=allow_partial_features,
            template_strategy="eager" if phase == "eager" else "dl",
            remember_templates=phase == "seed",
            **({"operators": GENERAL_OPERATORS} if operator_set == "general" else {}),
        )
        list(controller.run())
        budget_elapsed_seconds = perf_counter() - controller.started_at
        model_path = controller.save(output / phase)
        training_seconds = perf_counter() - t0
        sources = controller.accepted_template_sources
        phase_sources[phase] = sources
        snapshot = sorted((item.record, item.count) for item in controller.loaded_templates)
        programs = list(controller.candidate_programs.values())
        report[phase] = {
            "model_dir": str(model_path),
            "training_seconds": training_seconds,
            "budget_elapsed_seconds": budget_elapsed_seconds,
            "candidate_attempts": controller.candidate_attempts,
            "loaded_templates": len(controller.loaded_templates),
            "template_strategy": controller.template_strategy,
            "template_attempts": controller.template_attempts,
            "memory_snapshot_sha256": sha256(json.dumps(snapshot).encode()).hexdigest(),
            "memory_snapshot": snapshot,
            "accepted_features": len(controller.accepted_graphs),
            "accepted_features_from_memory": sum(bool(source) for source in sources),
            "accepted_programs": controller.accepted_programs,
            "evaluated_programs": programs,
            "accepted_context_features": controller.accepted_context_names,
            "operators": controller.operator_names,
            "test_metrics": evaluate_test(PredictionModel.load(model_path), stages["test"]),
        }
    warm_model = PredictionModel.load(output / "warm")
    if eager_control and report["warm"]["memory_snapshot_sha256"] != report["eager"]["memory_snapshot_sha256"]:
        raise ValueError("Warm and eager controls must load the identical frozen seed memory.")
    report["warm"]["feature_ablations"] = []
    for index in range(len(warm_model.graphs)):
        coefficients = warm_model.coefficients.copy()
        coefficients[len(warm_model.context_feature_names) + index] = 0
        ablated = PredictionModel(
            warm_model.graphs, list(coefficients), warm_model.initial_strength, warm_model.context_feature_names
        )
        report["warm"]["feature_ablations"].append(
            {
                "feature_index": index,
                "test_metrics_without_feature": evaluate_test(ablated, stages["test"]),
            }
        )
    cold_evaluated = {item["record"] for item in report["cold"]["evaluated_programs"]}
    cold_accepted = {item["record"] for item in report["cold"]["accepted_programs"]}
    report["transferred_programs"] = [
        {
            **program,
            "feature_index": index,
            "cold_evaluated": program["record"] in cold_evaluated,
            "cold_accepted": program["record"] in cold_accepted,
        }
        for index, (program, sources) in enumerate(
            zip(report["warm"]["accepted_programs"], phase_sources["warm"], strict=True)
        )
        if sources
    ]
    if primitive_control:
        prior_evaluated = {item["record"] for item in report["primitive"]["evaluated_programs"]}
        prior_accepted = {item["record"] for item in report["primitive"]["accepted_programs"]}
        for program in report["transferred_programs"]:
            program["primitive_control_evaluated"] = program["record"] in prior_evaluated
            program["primitive_control_accepted"] = program["record"] in prior_accepted
    report["memory_reused"] = report["warm"]["accepted_features_from_memory"] > 0
    report["limitations"] = [
        "Test rows never enter template memory or coefficient fitting.",
        "Proof-search timings require separate cascade runs with the exported cold/warm models.",
        "CSV theorem_index denotes trace order, not necessarily the MeTTa assertion index.",
        "A fixed candidate budget does not guarantee better ranking or reduced training time.",
        "Time budgets include CSV loading, context fitting, retrieval and search, but not model saving or held-out scoring.",
        "Deadlines are cooperative: an ongoing propagation, fit or HTTP request can overrun; late fits are rolled back.",
        "Cold program matching compares typed syntax and roles, not all semantic equivalences.",
        "Feature ablations zero one fitted coefficient without refitting; they diagnose contribution, not retrained alternatives.",
        "Only seed training writes memory; later variants use the same frozen seed generation.",
        "Warm uses the existing weighted DL heap only; eager additionally evaluates loaded numeric templates first.",
    ]
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mork-url", default="http://127.0.0.1:8000")
    parser.add_argument("--namespace", default="hyperon-cascade-v1")
    parser.add_argument("--seed-groups", type=int, default=30)
    parser.add_argument("--train-groups", type=int, default=30)
    parser.add_argument("--test-groups", type=int, default=30)
    parser.add_argument("--skip-groups", type=int, default=0)
    parser.add_argument("--max-popped", type=int, default=3000)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--max-seconds", type=float, help="Later-training wall-clock budget, including setup.")
    parser.add_argument("--seed-seconds", type=float, help="Separate historical learning budget.")
    parser.add_argument("--seed-max-popped", type=int)
    parser.add_argument("--min-template-operators", type=int, default=1)
    parser.add_argument("--primitive-control", action="store_true")
    parser.add_argument("--eager-control", action="store_true", help="Compare the previous eager template warmstart.")
    parser.add_argument(
        "--reverse-order", action="store_true", help="Reverse later-training order to check timing bias."
    )
    parser.add_argument("--operator-set", choices=("full", "general"), default="full")
    parser.add_argument("--partial-features", action="store_true", help="Evaluate programs with unused input roots.")
    args = parser.parse_args()
    report = compare(
        args.csv,
        args.output,
        args.mork_url,
        args.namespace,
        seed_groups=args.seed_groups,
        train_groups=args.train_groups,
        test_groups=args.test_groups,
        max_popped=args.max_popped,
        use_context=not args.no_context,
        max_seconds=args.max_seconds,
        seed_seconds=args.seed_seconds,
        seed_max_popped=args.seed_max_popped,
        min_template_operators=args.min_template_operators,
        primitive_control=args.primitive_control,
        operator_set=args.operator_set,
        allow_partial_features=args.partial_features,
        skip_groups=args.skip_groups,
        eager_control=args.eager_control,
        reverse_order=args.reverse_order,
    )
    summary = {
        **report,
        "stages": {
            name: {key: value for key, value in stage.items() if key != "goals"}
            for name, stage in report["stages"].items()
        },
        **{
            phase: {key: value for key, value in report[phase].items() if key != "evaluated_programs"}
            for phase in ("seed", "cold", "primitive", "eager", "warm")
            if phase in report
        },
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
