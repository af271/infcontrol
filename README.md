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

## MORK template memory and chronological cascade comparison

With WILLIAM's `feature/mork-template-memory` implementation installed, training
can use a live MORK server with weighted JSON exports:

```bash
python -m infcontrol.run_inference_control \
  --csv data/context_feature_table.csv --max-popped 3000 \
  --mork-url http://127.0.0.1:18761 --memory-namespace hyperon-features-v1 \
  --erase-memory --model-dir mork-results/seed
```

Omit `--erase-memory` in a subsequent training run to reuse templates. Without
`--mork-url`, the existing training behavior is unchanged and makes no memory
requests.

Accepted formula-feature programs and rooted subprograms are stored with
explicit `goal`, `candidate`, and `one` input roles and registered `Formula`
types. Input arrays and fitted coefficients are **not** stored as templates.
Retrieved templates enter the regular DAG search as count-weighted composite
operators, competing with primitives through the existing effective-DL heap.
This is the default `--template-strategy dl`: there is no separate template
evaluation pass. The previous experimental behavior remains available with
`--template-strategy eager`; it evaluates loaded numeric templates first, each
consuming one candidate attempt, before the remaining ordinary DAG search.
Non-numeric subprograms are search building blocks, not direct logistic features.
Coefficients are always fitted on the new training data and accepted using the
existing validation criterion.

For a reproducible chronological comparison:

```bash
python -m infcontrol.mork_benchmark \
  --csv data/context_feature_table.csv \
  --output mork-results/formula-transfer-v1 \
  --mork-url http://127.0.0.1:18761 \
  --namespace hyperon-formula-transfer-v1 \
  --seed-groups 30 --train-groups 30 --test-groups 30 \
  --max-popped 3000 --no-context
```

Use a **new output directory** for every comparison. Complete theorem groups
are read in chronological order: an early seed stage builds memory; cold and
warm models train on the same later rows at the same candidate budget; a final
test stage never enters fitting or memory. The training stages have their own
chronological validation partitions. The output includes `seed/`, `cold/`,
`warm/`, and `comparison.json` with held-out ranking metrics, training time,
candidate counts, and whether any retrieved features were actually selected.
Only the seed stage writes memory; later variants use the frozen historical
generation. This prevents one control from teaching another. Reports record the
loaded `(record, count)` snapshot and its SHA-256 fingerprint.
Goal statements in the report allow the cascade runner to map trace ordering
to MeTTa assertion indices rather than assuming these indices are identical.

`--no-context` isolates formula-feature transfer. By default the comparison
also includes the existing experience/context features. In the first small
contextual comparison, context explained the useful signal and no formula
template was selected: memory does **not** guarantee an improvement. The
formula-only run demonstrated selection of a retrieved feature on later data,
not universal superiority or a statistically established speedup.

### Timed transfer of multi-operator programs

For the stronger experiment, use a separate historical learning budget and the
same shorter wall-clock budget for cold, warm, and a static primitive-prior
control:

```bash
python -m infcontrol.mork_benchmark \
  --csv data/context_feature_table.csv \
  --output mork-results/new-timed-general-window-0 \
  --mork-url http://127.0.0.1:18761 --namespace new-timed-general-window-0 \
  --seed-seconds 30 --max-seconds 5 --max-popped 100000 \
  --min-template-operators 2 --primitive-control \
  --operator-set general --partial-features --no-context
```

`--operator-set general` uses formula size/depth/count measurements and general
arithmetic/comparison operations, not the specialized pairwise formula-similarity
operators. It excludes cross-row reducers (`Mean`, `Sum`, `Len`), exact
floating-point equality, and WILLIAM's precision-rounded `Sqrt`: those can
produce batch-dependent features or predicates driven by rounding artifacts.
For example, `n / sqrt(n) < sqrt(n)` is mathematically false for positive `n`,
but can become true with rounded square roots. `--partial-features` explicitly enables evaluation of a
computed feature root even when other inputs remain unused; the original
single-connected-root behavior remains the default.

The static control directly evaluates all type-compatible, single-operator
programs on the available input roles in fixed order before ordinary search.
It never sees the historical memory. Warm retrieval is filtered to programs
containing at least `--min-template-operators` primitive DAG operator nodes;
the seed still stores their complete rooted subprograms. Operator count is
structural complexity, not proof that a program has no simpler equivalent.

The time budget starts before CSV parsing and includes context fitting, MORK
retrieval, template evaluation and search. It excludes exporting the model
and scoring the held-out test set. Deadlines are **cooperative**, not hard
process timeouts: a propagation, fit or HTTP request already in progress may
finish after the deadline, but a fit that finishes late is rolled back and
does not enter the model. Reports include measured `budget_elapsed_seconds`
and mark accepted programs with `within_time_budget`.

The report records every distinct valid numeric program evaluated for fitting,
its first evaluation time, and accepted programs. `transferred_programs`
distinguishes programs Cold did not evaluate from those it evaluated but
rejected. The same comparison is recorded for the static control. Matching is
by typed syntax and input roles, not a general semantic-equivalence proof.
Warm feature ablations zero each coefficient separately, without refitting,
to measure its contribution to held-out metrics; they do not affect training
or template selection.

Repeat on later chronological windows using `--skip-groups 30` or
`--skip-groups 60`, each with a fresh output directory and namespace. Omit
`--no-context` to test against the existing contextual predictor. A ranking
gain in one window is not evidence of general speedup or better calibration.

Feature records now use `hyperon_feature_v2`: DAG binding/reference markers are
encoded as literal atoms so MORK does not reinterpret them as query variables.
The loader continues to accept existing v1 records.

#### Initial timed results

These historical reports used the **eager** warmstart, not today's DL-only default.
The local `mork-results/timed-stable-window-{0,30,60}/comparison.json` reports
use 30 historical seed groups, 30 later training groups, and 30 final test
groups, with a 30-second seed budget and a 5-second later-training budget.
The three overlapping chronological windows are exploratory, not independent
statistical replications. Mean position for full recall (lower is better):

| Skipped groups | Cold | Static primitive control | Warm |
| --- | ---: | ---: | ---: |
| 0 | 45.70 | 45.70 | 52.47 |
| 30 | 46.57 | 55.50 | 60.46 |
| 60 | 55.63 | 55.63 | 55.83 |

Warm accepted multi-operator programs not evaluated by either control within
their budgets in all three windows: respectively one, two, and one such
programs. This establishes faster access to particular program structures,
**not better held-out model quality**. Warm did not beat Cold in these windows.
Some programs help within the warm model's coefficient-zeroing ablation,
while others hurt; validation acceptance alone does not establish useful
transfer to future theorems.

In `timed-stable-context-window-0`, Warm also selected two five-operator
programs at approximately 3.70 and 4.52 seconds which Cold did not evaluate.
However, full-recall position changed from 18.97 to 19.53 and log-loss from
0.12878 to 0.12994, both slightly worse. A PeTTa cascade smoke run verified
actual scoring by this contextual model and found the first held-out proof.
It is not a proof-search performance comparison.

Earlier `general-program-transfer-v3` and `timed-general-window-*` probes
included rounded square roots and/or exact equality. Their apparent gains
are not the basis for a positive transfer claim; the stable results above
exclude those operations. The next research question is selecting templates
that transfer reliably, rather than merely adding more remembered programs.

#### Controlled eager versus DL-only ablation

```bash
python -m infcontrol.mork_benchmark \
  --csv data/context_feature_table.csv \
  --output mork-results/new-dl-ablation --namespace new-dl-ablation \
  --mork-url http://127.0.0.1:18761 \
  --seed-seconds 30 --max-seconds 5 --max-popped 100000 \
  --min-template-operators 2 --operator-set general --partial-features \
  --no-context --eager-control
```

The additional `eager/` variant uses exactly the same loaded templates and
counts as `warm/`; a fingerprint mismatch raises an error. `warm/` uses only
the normal weighted-DL search (`template_attempts == 0`). `eager/` adds the old
explicit template-first pass. Use `--reverse-order` on a repeat to reverse
the cold/eager/warm execution order. No weighting constants, fitter settings,
operator library or acceptance criteria differ between these variants.

Eight local reports at
`mork-results/dl-ablation-{formula-0,formula-30,formula-60,context-0}-r{1,2}`
measure these four scenarios twice with the same 30/30/30 groups and 30/5-second
budgets. The repeat reverses later-training order. Both repeats produced
identical held-out metrics and identical historical template/count snapshots.
Mean position for full recall (lower is better):

| Scenario | Cold | Eager warmstart | DL-only memory |
| --- | ---: | ---: | ---: |
| Formula, skip 0 groups | 49.83 | 48.13 | 48.13 |
| Formula, skip 30 groups | 46.57 | 66.79 | 64.86 |
| Formula, skip 60 groups | 55.63 | 55.83 | 53.37 |
| Context, skip 0 groups | 18.97 | 19.53 | 19.53 |

DL-only improves on eager in two formula windows, matches it in the others,
but does not consistently beat Cold. Log-loss remains worse than Cold in all
four scenarios, including those with better full-recall ranking. These are
exploratory overlapping windows, not independent evidence of general superiority.
Measured setup-inclusive later-training times were 5.00-5.15 seconds; all
accepted programs finished fitting within the five-second deadline.

Removing the eager pass does not imply a weak memory prior. Inspection of the
unchanged native DL weights found template costs around 4.4-5.6 bits versus
8.2-8.6 bits for primitives, depending on the scenario. Thus templates can
still appear very early through the regular heap. The ablation does not
establish that overfitting is the cause: it shows the explicit eager pass
alone does not explain the remaining generalization gap.

The companion cascade benchmark now measures actual ECAN/cold/MORK proof
search with scoring-inclusive deadlines and full theorem wall times, rather
than inferring speed from offline ranks. Local reports are retained in
`mork-results/chainer-context-dl-confirm-v1` (six repetitions, assertions 67-96)
and `mork-results/chainer-context-dl-later-v1` (four repetitions, assertions
97-126), using the unchanged contextual `dl-ablation-context-0-r1` models.
All variants solve all 30 tasks in each window. Warm versus Cold median
evaluation times are 4.005 versus 4.048 seconds in the first window and
2.238 versus 2.199 seconds in the second: no robust aggregate MORK advantage
is established. A repeatable local improvement on `syl10` is documented,
alongside all other per-theorem results, in the cascade benchmark README.
Its `compare_cascade.py` command reproduces this three-way measurement.

Saved feature graphs are expanded back to the domain's primitive operators,
so the predictor requires neither MORK nor dynamic template classes.
`template_memory.json` accompanies memory-enabled models and records their
generation, loaded templates, and source templates for each accepted feature.
The cascade's `--require-memory` flag can reject models without actual selected
memory features.

Run integration tests against a real server:

```bash
WILLIAM_MORK_TEST_URL=http://127.0.0.1:18761 \
  python -m pytest tests/test_template_memory.py
```

See the chaining benchmark's `experimental/metamath-aa/pc-xp-union/README.md`
for cold/warm proof-search commands and total-process timing. This is currently
batch retraining from existing traces, **not** continuous live ingestion or
durable memory across MORK-server restarts.

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
