"""Serve a saved inference-control model for prediction from any language.

Loads a model directory written by :func:`infcontrol.model.save_model` once,
then predicts use-probabilities for goal/candidate formula pairs given as
MeTTa-style JSON trees (the same format produced by ``metta_to_json.py``).

Two modes are supported:

- One-shot: ``--goal`` and ``--candidate`` are JSON strings, a single JSON
  result is printed and the process exits. Convenient for shell scripts.
- Long-running service: with no ``--goal``/``--candidate``, the process reads
  one JSON request per line from stdin and writes one JSON response per line
  to stdout, flushing after each line. This avoids reloading the model for
  every prediction and can be driven from any language that can spawn a
  subprocess and talk newline-delimited JSON over pipes, e.g. Rust's
  ``std::process::Command`` or MeTTa's Python interop.

Request formats (one per line):
    {"goal": [...], "candidate": [...]}
    {"goals": [[...], ...], "candidates": [[...], ...]}

Response formats:
    {"probability": 0.83}
    {"probabilities": [0.83, 0.12, ...]}
    {"error": "..."}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO

from infcontrol.context_features import OnlineContextTracker
from infcontrol.model import PredictionModel

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "default"


def _predict_one(model: PredictionModel, goal: object, candidate: object) -> float:
    return float(model.predict([goal], [candidate])[0])


def serve(model: PredictionModel, input_stream: IO[str], output_stream: IO[str]) -> None:
    """Answer one JSON request per input line until the stream closes."""
    context_tracker = OnlineContextTracker()
    for line in input_stream:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "observe":
                context_tracker.observe(
                    request["theory_labels"], request["used_labels"]
                )
                response: dict[str, object] = {"ok": True}
                output_stream.write(json.dumps(response) + "\n")
                output_stream.flush()
                continue
            if "goals" in request or "candidates" in request:
                context_features = request.get("context_features")
                if context_features is None and model.context_feature_names:
                    generated = context_tracker.features(request["candidate_labels"])
                    context_features = {
                        name: generated[name] for name in model.context_feature_names
                    }
                probabilities = model.predict(
                    request["goals"], request["candidates"], context_features
                )
                response = {"probabilities": probabilities.tolist()}
            else:
                probability = _predict_one(model, request["goal"], request["candidate"])
                response = {"probability": probability}
        except Exception as exc:  # noqa: BLE001 - report to caller, keep serving
            response = {"error": str(exc)}
        output_stream.write(json.dumps(response) + "\n")
        output_stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory written by infcontrol.model.save_model (default: %(default)s)",
    )
    parser.add_argument("--goal", type=str, help="Single goal formula as a JSON string")
    parser.add_argument("--candidate", type=str, help="Single candidate formula as a JSON string")
    args = parser.parse_args()

    model = PredictionModel.load(args.model_dir)

    if args.goal is not None or args.candidate is not None:
        if args.goal is None or args.candidate is None:
            parser.error("--goal and --candidate must be given together")
        probability = _predict_one(model, json.loads(args.goal), json.loads(args.candidate))
        print(json.dumps({"probability": probability}))
        return

    serve(model, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
