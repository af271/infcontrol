"""Run inference control on the real feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from infcontrol.inference_control import InferenceControl

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "feature_table.csv"


def run(csv_path: Path, max_popped: int | None) -> None:
    controller = InferenceControl(csv_path, max_popped=max_popped, split_seed=12)
    controller.print_metrics()
    for graph, coefficient in controller.run():
        mem = controller.last_mem
        graph.render(mem)
        print(f"accepted coefficient: {coefficient:.6f}")
        controller.print_metrics()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--max-popped", type=int, default=1000)
    args = parser.parse_args()
    run(args.csv, args.max_popped)


if __name__ == "__main__":
    main()
