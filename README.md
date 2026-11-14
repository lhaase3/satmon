# SatMon: spacecraft telemetry anomaly detection

CSCI 5622 experiment asking whether supervised models improve anomaly-event
recall over a rolling statistical baseline under a fixed analyst review budget.
The experiment compares rolling 3σ, L2 Logistic Regression, and Random Forest
on real SMAP/MSL telemetry. The original SatMon frontend is preserved as a design
reference and is **not required to reproduce these results**.

## Reproduce the headline result

Use **Python 3.12** (original experiments: 3.12.10). Run from the repository root.
Until the course branch is merged into `main`, clone it explicitly:

```bash
git clone --branch course-backend-rebuild https://github.com/lhaase3/satmon.git
cd satmon
python -m venv .venv
```

Activate the environment in your terminal:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Then run these commands in the activated environment. Each command is one line
and works in PowerShell or Bash:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/download_data.py
python scripts/run_experiment.py --annotations data/raw/labeled_anomalies.csv --telemetry-dir data/raw/archive/data/data/test --splits splits/channels.json --protocol configs/protocol.json --training-order splits/training_channel_order.json --output-dir results/reproduction
```

The download script needs internet access and downloads the public Kaggle archive
(about 86 MB), without a Kaggle package or API key. It installs only the annotation
CSV and 80 required telemetry arrays. SHA-256 hashes are checked against
`data/manifest.json`; changed source files cause an error rather than silently
changing the experiment. Existing matching files are reused. Raw inputs are
ignored by Git. See [data provenance](data/README.md).

**Fixed seed: 42** for channel splitting, model random states, and bootstrap
resampling. The runner reuses the saved split and training-row order; it does not
repeat validation tuning or choose new settings. Dependencies are pinned in
`requirements-lock.txt`; declarations are in `pyproject.toml`.

Verified on Windows with a fresh Python 3.12.10 environment and a fresh data
download on November 14, 2026. macOS/Linux have not been independently tested.

### Expected held-out results

The test set contains 16 channels and 19 annotated events.

| Model | Events detected | Event Recall@5 | False alerts |
| --- | ---: | ---: | ---: |
| Rolling 3σ | 17/19 | 89.5% | 146 |
| Random Forest | 15/19 | 78.9% | 297 |
| Logistic Regression | 13/19 | 68.4% | 316 |

The runner prints these counts and writes `summary.csv`, `channels.csv`,
`events.csv`, `alerts.csv`, and `failure_scores.csv` under `results/reproduction/`.
The original reference artifacts remain under `results/final_test/`.

To regenerate uncertainty tables, spacecraft/event-type slices, and the failure
figure after the headline command:

```bash
python scripts/bootstrap_results.py --channels results/reproduction/channels.csv --events results/reproduction/events.csv --output-dir results/reproduction --reference-dir results/final_test --seed 42 --draws 5000
python scripts/plot_failures.py --results-dir results/reproduction --output-dir results/reproduction/figures
```

The bootstrap command checks its four tables against the saved reference tables.
The figure and its diagnostic CSV go in `results/reproduction/figures/`.
Validation/PCA selection and model interpretation remain documented in the
notebook; the commands above reproduce the frozen test evaluation, bootstrap
analysis, and failure figure. Run correctness checks with `python -m pytest`.

## Data and experimental setup

Data come from [Telemanom](https://github.com/khundman/telemanom), distributed
through [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl).
Annotations describe intervals in the original test sequences. We use those
sequences for a separate spacecraft-stratified **48/16/16 channel**
train/validation/test split, saved in `splits/channels.json`. P-2 is excluded
because its duplicate annotation rows conflict; previously explored P-1 is
reserved for training. Only the first telemetry column is used.

Supervised inputs are current value, first difference, and current-inclusive
100-sample rolling mean and population standard deviation. A training target is
positive when any annotated sample overlaps that trailing window. All methods
start scoring at original index 100. Scaling and models are fit on training
channels only; validation selected settings before final test evaluation.

- **Rolling 3σ:** absolute deviation from the previous 100 samples, excluding the
  current sample; candidates have scores strictly greater than 3.
- **Logistic Regression:** training-fitted standardization, L2 regularization,
  `C=1`, probability threshold `>0.13`.
- **Random Forest:** 100 trees, maximum depth 8, minimum leaf size 100,
  probability threshold `>0.25`.

Frozen settings and detailed conventions are in `configs/protocol.json`.

## Evaluation and limitations

**Event Recall@5 = distinct annotated events detected / total annotated events.**
Each channel is divided into consecutive blocks of 1,000 original samples, with
at most five alerts per block, including the final partial block. Candidates are
ranked by descending score, with earlier indices breaking ties. Alerts within
10 samples of an already selected alert are suppressed within the same block.
An event is detected if a selected alert falls inside its inclusive endpoints;
each event counts once across blocks. A false alert falls outside every event.
Warm-up events remain in the denominator.

Telemanom's upstream normalization uses original test-sequence extrema, limiting
online-causality claims. Alert ranking is retrospective within each block.
The held-out set has only 19 events; paired whole-channel bootstrap recall
intervals for differences between the baseline and supervised models include
zero, so these results do not establish clear recall superiority. Validation
selection was exploratory. See [measured findings](report/findings.md) for details.

## Repository organization and original application

- `src/satmon/`, `tests/`: experiment implementation and correctness checks.
- `scripts/`, `configs/`, `splits/`: acquisition, reproduction, frozen settings,
  channel assignments, and training order.
- `notebooks/`, `results/`, `report/`: development experiments, measured artifacts,
  findings, and the [AI assistance log](report/ai_use_log.md).
- `web/`, `static/`, `index.html`: preserved frontend prototype, disconnected
  from the experiment. Its demo metrics are not research results.

The original application and older backend experiments are preserved in Git at
[`pre-course-rebuild`](https://github.com/lhaase3/satmon/tree/pre-course-rebuild)
(commit `66f2807`). No database, web server, or old application setup is needed.
