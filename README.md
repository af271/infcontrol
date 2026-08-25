# infcontrol

Symbolic inference-control utilities and feature operators.

## Local development

1. Create and activate a virtual environment.
2. Install William from the sibling repository in editable mode.
3. Install this repository in editable mode with dev dependencies.
4. Run tests.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../../william
pip install -e .[dev]
pytest
```

## Training and persisting a model

`InferenceControl` searches for accepted feature graphs and fits a logistic
model on top of them (see `run_inference_control.py`). To reuse a trained
model for prediction without repeating the (expensive) search, persist it to
a directory:

```python
controller = InferenceControl(csv_path, max_popped=1000)
list(controller.run())
controller.save("infcontrol/models/my_model")
```

This writes one `.dot` file per accepted feature graph plus a `model.json`
with the fitted logit coefficients into `infcontrol/models/my_model/`. The
same thing happens automatically when running the training script with
`--model-dir`:

```bash
python run_inference_control.py --model-dir infcontrol/models/default
```

A saved model directory is self-contained: it does not need the training CSV
or a live `InferenceControl` instance to make predictions. Load it with
`PredictionModel`:

```python
from infcontrol import PredictionModel

model = PredictionModel.load("infcontrol/models/default")
probabilities = model.predict(goal_formulas, candidate_formulas)
```

## Predicting from other languages

`infcontrol-predict` (installed as a console script, or runnable as
`python -m infcontrol.predict_cli`) loads a saved model once and serves
predictions over stdio, so any language that can spawn a subprocess and
exchange newline-delimited JSON can use it without a Python runtime
dependency of its own:

```bash
# one-shot: prints a single JSON result and exits
infcontrol-predict --model-dir infcontrol/models/default \
  --goal '["->", "g0", "b"]' --candidate '["->", "g0", "b"]'

# long-running service: one JSON request per input line, one JSON response
# per output line (flushed immediately), avoids reloading the model
infcontrol-predict --model-dir infcontrol/models/default
```

Request lines look like `{"goal": [...], "candidate": [...]}` (single pair)
or `{"goals": [...], "candidates": [...]}` (batched), matching the same
MeTTa-style JSON tree format used elsewhere in this repository (see
`metta_to_json.py`). Responses are `{"probability": ...}`,
`{"probabilities": [...]}`, or `{"error": "..."}`. Any language with a
subprocess API (e.g. Rust's `std::process::Command`) can drive this protocol
directly over the child process's stdin/stdout pipes.
