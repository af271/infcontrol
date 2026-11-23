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
    mork_url: str | None = None,
    memory_namespace: str = "hyperon-features-v1",
    erase_memory: bool = False,
    max_seconds: float | None = None,
    min_template_operators: int = 1,
    allow_partial_features: bool = False,
    template_strategy: str = "dl",
) -> None:
    controller = InferenceControl(
        csv_path,
        max_popped=max_popped,
        split_seed=13,
        data_fraction=data_fraction,
        validation_fraction=0.2,
        test_fraction=0.2,
        mork_url=mork_url,
        memory_namespace=memory_namespace,
        erase_memory=erase_memory,
        max_seconds=max_seconds,
        min_template_operators=min_template_operators,
        allow_partial_features=allow_partial_features,
        template_strategy=template_strategy,
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
    parser.add_argument("--mork-url", default=None, help="Enable MORK template retrieval and learning.")
    parser.add_argument("--memory-namespace", default="hyperon-features-v1")
    parser.add_argument("--erase-memory", action="store_true", help="Start a fresh template generation.")
    parser.add_argument("--max-seconds", type=float, help="Cooperative training deadline, including setup.")
    parser.add_argument("--min-template-operators", type=int, default=1)
    parser.add_argument("--partial-features", action="store_true")
    parser.add_argument(
        "--template-strategy",
        choices=("dl", "eager"),
        default="dl",
        help="DL heap ordering (default), or the experimental eager template pass.",
    )
    args = parser.parse_args()
    run(
        args.csv,
        args.max_popped,
        args.data_fraction,
        args.model_dir,
        args.mork_url,
        args.memory_namespace,
        args.erase_memory,
        args.max_seconds,
        args.min_template_operators,
        args.partial_features,
        args.template_strategy,
    )


if __name__ == "__main__":
    main()
