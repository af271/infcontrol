"""Isolated, budgeted training for the continuous chaining service."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from infcontrol.context_features import CONTEXT_COLUMNS
from infcontrol.inference_control import GENERAL_OPERATORS, InferenceControl
from infcontrol.model import PredictionModel
from infcontrol.mork_benchmark import evaluate_test
from infcontrol.template_memory import feature_record
from william.mork_memory import MorkTemplateMemory


def restore_memory(url: str, namespace: str, counts: dict[str, int]) -> MorkTemplateMemory:
    """Rebuild only this service's namespace from its durable local snapshot."""
    memory = MorkTemplateMemory(url, namespace, erase=True)
    remaining = Counter(counts)
    while remaining:
        memory.remember(list(remaining))
        remaining.subtract(list(remaining))
        remaining = +remaining
    return memory


def write_rows(path: Path, events: list[dict]) -> None:
    fields = ["goal", "candidate", "used", *CONTEXT_COLUMNS]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for event in events:
            request = event["request"]
            used = set(event["used_labels"])
            for index, (label, candidate) in enumerate(
                zip(request["candidate_labels"], request["candidates"], strict=True)
            ):
                writer.writerow(
                    {
                        "goal": json.dumps(request["goal"]),
                        "candidate": json.dumps(candidate),
                        "used": str(label in used),
                        **{name: event["context"][name][index] for name in CONTEXT_COLUMNS if name != "theorem_index"},
                        "theorem_index": event["seq"],
                    }
                )


def train_job(directory: Path) -> dict:
    spec = json.loads((directory / "input.json").read_text())
    events = spec["events"]
    # The outer gate is never passed to feature fitting or template extraction.
    split = len(events) - max(2, len(events) // 5)
    training, gate = events[:split], events[split:]
    write_rows(directory / "training.csv", training)
    write_rows(directory / "gate.csv", gate)
    memory = restore_memory(spec["mork_url"], spec["namespace"], spec["memory"])
    controller = InferenceControl(
        directory / "training.csv",
        operators=GENERAL_OPERATORS,
        max_popped=spec["max_popped"],
        max_seconds=spec["max_seconds"],
        mork_url=spec["mork_url"],
        memory_namespace=spec["namespace"],
        remember_templates=False,
        allow_partial_features=True,
        test_fraction=0,
    )
    list(controller.run())
    controller.save(directory / "model")
    candidate = evaluate_test(PredictionModel.load(directory / "model"), directory / "gate.csv")
    baseline = evaluate_test(PredictionModel.load(spec["base_model_dir"]), directory / "gate.csv")
    metrics = ("k_at_full_recall", "log_loss")
    passed = all(candidate[key] <= baseline[key] for key in metrics) and any(
        candidate[key] < baseline[key] for key in metrics
    )
    records = set()
    for graph in controller.accepted_graphs:
        root = list(graph.roots())[0].clone_and_unpack()
        records.update(feature_record(node) for node in root.walk(above=False, op_nodes=False) if node.options)
    result = {
        "passed": passed,
        "gate_rule": "No regression in mean full-recall rank or log-loss; strict improvement in at least one.",
        "candidate": candidate,
        "baseline": baseline,
        "training_event_seqs": [event["seq"] for event in training],
        "gate_event_seqs": [event["seq"] for event in gate],
        "candidate_attempts": controller.candidate_attempts,
        "loaded_templates": len(controller.loaded_templates),
        "accepted_features_from_memory": sum(bool(sources) for sources in controller.accepted_template_sources),
        "memory_generation": memory.generation,
        "template_records": sorted(records),
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_directory", type=Path)
    args = parser.parse_args()
    train_job(args.job_directory.resolve())


if __name__ == "__main__":
    main()
