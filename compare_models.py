"""Train and compare formula-only and context-enhanced models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infcontrol.context_features import write_context_feature_table
from infcontrol.inference_control import InferenceControl

DEFAULT_TRACE = (
    Path(__file__).resolve().parent
    / "data"
    / "pc-xp-llm-trace-full-theory-to-833.metta"
)
DEFAULT_CONTEXT_CSV = (
    Path(__file__).resolve().parent / "data" / "context_feature_table.csv"
)


def train_model(
    name: str,
    csv_path: Path,
    model_dir: Path,
    use_context: bool,
    data_fraction: float,
    max_popped: int,
) -> dict[str, float | int | str]:
    print(f"[{name}] initializing", flush=True)
    controller = InferenceControl(
        csv_path,
        use_context=use_context,
        data_fraction=data_fraction,
        validation_fraction=0.2,
        test_fraction=0.2,
        split_seed=42,
        max_popped=max_popped,
    )
    print(f"[{name}] rows={len(controller.rows)}; searching", flush=True)
    accepted = list(controller.run())
    print(
        f"[{name}] accepted graphs={len(accepted)}, context={len(controller.accepted_context_names)}",
        flush=True,
    )
    controller.save(model_dir)
    result = {
        "model": name,
        "rows": len(controller.rows),
        "graphs": len(controller.accepted_graphs),
        "context_features": len(controller.accepted_context_names),
    }
    for split in ("train", "validation", "test"):
        result[f"{split}_loss"] = controller.log_loss(split)
        result[f"{split}_k100"] = controller.k_at_recall(split, 1.0)
        result[f"{split}_all40"] = controller.all_at_k(split, 40)
        result[f"{split}_all100"] = controller.all_at_k(split, 100)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--context-csv", type=Path, default=DEFAULT_CONTEXT_CSV)
    parser.add_argument("--data-fraction", type=float, default=0.1)
    parser.add_argument("--max-popped", type=int, default=200)
    args = parser.parse_args()

    if not args.context_csv.exists():
        print(f"generating {args.context_csv}", flush=True)
        write_context_feature_table(args.trace, args.context_csv)

    results = [
        train_model(
            "model02",
            args.context_csv,
            Path("models/model02"),
            use_context=False,
            data_fraction=args.data_fraction,
            max_popped=args.max_popped,
        ),
        train_model(
            "model03",
            args.context_csv,
            Path("models/model03"),
            use_context=True,
            data_fraction=args.data_fraction,
            max_popped=args.max_popped,
        ),
    ]
    print("comparison", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
