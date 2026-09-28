# infcontrol

Symbolic inference-control utilities for Hyperon/MeTTa theorem proving.

Given a proof **goal** and a pool of candidate **premises** (lemmas, facts,
previously derived formulas), `infcontrol` learns which candidates are likely
to actually be useful for proving the goal. It does this by:

1. Symbolically enumerating structural features over (goal, candidate) pairs
   — tree size/depth, shared operators/variables/symbols, subtree overlap,
   implication chain length, and more (see `operators.py`) — using William's
   type-directed program search (`Wunderbaum`).
2. Greedily fitting a logistic regression on top of the accepted features
   (`LogitFitter`), keeping only features that measurably improve held-out
   log-loss.
3. Persisting the resulting model so it can be used for prediction — from
   Python or from any other language — without repeating the search.

The output is a `predict(goal, candidate) -> probability` function: given a
goal and a candidate, how likely is it that the candidate will be used in the
proof? This is meant to guide inference control / premise selection in a
theorem prover.

## Installation

### 1. William

`infcontrol`'s feature search is built on top of
[William](https://pypi.org/project/william/), a symbolic program search and
compression library:

```bash
pip install william
```

(For local development against an unreleased William checkout, install it in
editable mode instead: `pip install -e ../../william`.)

### 2. This repository

```bash
python -m venv .venv
source .venv/bin/activate
pip install william
pip install -e .[dev]
```

`.[dev]` pulls in `pytest`/`ruff` for testing and linting. Run the test suite
with:

```bash
pytest
```

## Repository layout

- `dataset.py`, `feature_table.py` — turn a raw MeTTa proof trace
  (`data/*.metta`, converted to JSON with `metta_to_json.py`) into a
  `(goal, candidate, used)` CSV training table.
- `operators.py` — the symbolic feature operators searched over, and the
  `Formula` wrapper around parsed MeTTa/Hyperon expression trees.
- `inference_control.py` — `InferenceControl`: runs the feature search and
  logistic fit end to end, given a training CSV.
- `logit_fitter.py` — `LogitFitter`: stagewise logistic regression used to
  accept or reject each candidate feature.
- `model.py` — persists/loads a trained model (`PredictionModel`) so it can
  be used for prediction independently of the training data.
- `predict_cli.py` — command-line/stdio interface for predicting with a
  saved model from any language.
- `run_inference_control.py` — CLI entry point that trains a model on the
  feature table and optionally saves it.

## Training a model

```bash
python run_inference_control.py --max-popped 100000 --model-dir models/model01
```

- `--data-fraction` trains on a subset of theorem instances for exploratory runs.
  Use the default full dataset for reported results.
- `--max-popped` bounds how many candidate feature graphs are searched.
- `--model-dir` saves the trained model (accepted feature graphs + fitted
  logit coefficients) to that directory once training finishes.

Rows with a `theorem_index` column are split chronologically by theorem
instance. Reports include `all@k`, the fraction of theorems for which all known
proof premises occur in the first `k` candidates, and `k@100%`, the mean rank
of the last required premise.

A saved model directory is self-contained: it does not need the training CSV
or a live `InferenceControl` instance to make predictions afterwards.

### Contextual model03

`models/model03` is the contextual premise-ranking model used by the
ECAN/WILLIAM benchmark. It combines one symbolically discovered formula
feature with eight strictly chronological experience features: candidate age,
prior use and exposure counts, smoothed use probabilities, and recent-use
windows. The checked-in model is ready for inference.

Training data is intentionally not packaged: `data/` is ignored because the
generated context table is large. Exact retraining therefore requires the
original `pc-xp-llm-trace-full-theory-to-833.metta` trace. Given that trace,
run:

```bash
python compare_models.py \
  --trace data/pc-xp-llm-trace-full-theory-to-833.metta \
  --context-csv data/context_feature_table.csv \
  --data-fraction 0.1 \
  --max-popped 200
```

The script generates the context table if necessary, trains the formula-only
model02 baseline and contextual model03, and reports chronological
train/validation/test metrics. Increase `--data-fraction` and `--max-popped`
for a larger feature search. Context rows are computed before observing the
current theorem's proof, so current labels cannot leak into their own inputs.

## Predicting with a trained model

### From Python

```python
from infcontrol import PredictionModel

model = PredictionModel.load("models/model01")
probabilities = model.predict(goal_formulas, candidate_formulas)
```

### From the command line / other languages

`predict_cli.py` loads a saved model once and serves predictions, so any
language that can spawn a subprocess and exchange JSON can use it without a
Python runtime dependency of its own.

One-shot prediction:

```bash
python -m infcontrol.predict_cli --model-dir models/model01 \
  --goal '["->", ["→", "v0", "v1"], ["→", "v2", "v0"], "v2", "v1"]' \
  --candidate '["->", ["→", "v0", ["→", "v1", "v2"]], "v1", "v0", "v2"]'
```

```json
{"probability": 9.668303753894887e-47}
```

Goal and candidate are MeTTa-style JSON expression trees, the same format
produced by `metta_to_json.py`.

Long-running service (loads the model once, then answers one JSON request per
input line with one JSON response per output line, flushed immediately):

```bash
python -m infcontrol.predict_cli --model-dir models/model01
```

```bash
echo '{"goal": ["->", "g0", "b"], "candidate": ["->", "g0", "b"]}' \
  | python -m infcontrol.predict_cli --model-dir models/model01
# {"probability": 0.87}

echo '{"goals": [["->", "g0", "b"], ["->", "g1", "b"]], "candidates": [["->", "g0", "b"], ["->", "c1", "d"]]}' \
  | python -m infcontrol.predict_cli --model-dir models/model01
# {"probabilities": [0.87, 0.02]}
```

Errors are reported as `{"error": "..."}` without killing the process, so a
long-running caller can keep sending requests. This makes the CLI drivable
from e.g. Rust's `std::process::Command` or any other subprocess API.

If installed with `pip install -e .` (which registers the
`infcontrol-predict` console script), the `python -m infcontrol.predict_cli`
prefix above can be shortened to `infcontrol-predict`.

## Comparing proof-search logs

`search_log_benchmark.py` parses ECAN/WILLIAM search logs, separates appended
runs at `Iteration 2`, and marks runs without a daemon summary as incomplete.
It reports focused-tier hits, fallback count and time, search-time percentiles,
and nomination sizes. The full output also contains one record per theorem:

```bash
infcontrol-search-log pc-xp-union.log --summary-only
```

Omit `--summary-only` to retain theorem labels, tier outcomes, sizes, and
timings for paired comparisons between rankers.

For the complete ECAN/WILLIAM/Union installation, execution protocol,
reported results, and limitations, see the benchmark README under
`chaining-pc-xp-benchmark/experimental/metamath-aa/pc-xp-union/README.md` in
the sibling chaining repository.
