"""Run inference control on the real feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from infcontrol.inference_control import InferenceControl

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "feature_table.csv"


def _accuracy(controller: InferenceControl, split: str) -> float:
    indices = {
        "train": controller.fitter.train_indices,
        "validation": controller.fitter.validation_indices,
        "test": controller.fitter.test_indices,
    }[split]
    predictions = controller.fitter.probabilities[indices] >= 0.5
    return float(np.mean(predictions == controller.labels[indices]))


def _print_metrics(controller: InferenceControl) -> None:
    feature_count = len(controller.fitter.features)
    print(f"features: {feature_count}")
    for split in ("train", "validation", "test"):
        loss = controller.log_loss(split)
        mean_observed_probability = np.exp(-loss)
        accuracy = _accuracy(controller, split)
        print(
            f"{split}: log_loss={loss:.6f}, "
            f"geometric_mean_observed_probability={mean_observed_probability:.6f}, "
            f"accuracy={accuracy:.6f}"
        )


def run(csv_path: Path, max_popped: int | None) -> None:
    controller = InferenceControl(csv_path, max_popped=max_popped)
    _print_metrics(controller)
    for graph, coefficient in controller.run():
        mem = controller.last_mem
        graph.render(mem)
        print(f"accepted coefficient: {coefficient:.6f}")
        _print_metrics(controller)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--max-popped", type=int, default=1000)
    args = parser.parse_args()
    run(args.csv, args.max_popped)


if __name__ == "__main__":
    main()
