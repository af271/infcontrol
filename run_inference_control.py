"""Run inference control on the real feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from infcontrol.inference_control import InferenceControl

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "feature_table.csv"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "default"


def run(
    csv_path: Path,
    max_popped: int | None,
    data_fraction: float = 1.0,
    model_dir: Path | None = None,
) -> None:
    controller = InferenceControl(
        csv_path,
        max_popped=max_popped,
        split_seed=13,
        data_fraction=data_fraction,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    controller.print_metrics()
    for graph, coefficient in controller.run():
        mem = controller.last_mem
        graph.render(mem, view=False)
        sexpr = list(graph.roots())[0].to_sexpr()
        print(f"\naccepted graph: {sexpr}")
        print(f"accepted coefficient: {coefficient:.6f}")
        controller.print_metrics()
    if model_dir is not None:
        saved_to = controller.save(model_dir)
        print(f"\nmodel saved to: {saved_to}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--max-popped", type=int, default=1000)
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of the CSV rows to use, keeping complete Goal groups.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        const=DEFAULT_MODEL_DIR,
        nargs="?",
        help="Save the trained model here for infcontrol-predict (default: %(const)s).",
    )
    args = parser.parse_args()
    run(args.csv, args.max_popped, args.data_fraction, args.model_dir)


if __name__ == "__main__":
    main()
