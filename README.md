# SatMon: spacecraft telemetry anomaly detection

CSCI 5622 backend project comparing rolling 3-sigma, regularized Logistic
Regression, and Random Forest under a fixed analyst review budget.

**Status: scaffold only.** Data loading, training, and evaluation are not yet
implemented. No measured performance is claimed.

## Setup

Target Python 3.11 or 3.12. From this repository in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Dependencies live in pyproject.toml. A validated environment lock and exact
results-reproduction command will be added once the experiment works.

## Locked experiment

- Real SMAP/MSL Telemanom data; annotations refer to original test sequences.
- Approximately 60/20/20 train/validation/test channels, seed 42; save assignments.
- Trailing features only; fit learned preprocessing on training data only.
- Primary metric: distinct events detected with at most five alerts per channel
  per consecutive block of 1,000 original samples.
- Shared alert suppression, ranking, and matching for all three methods.
- Validate PCA plus Logistic Regression before final test evaluation.
- Analyze spacecraft and reliable anomaly-type slices; bootstrap whole channels.
- Freeze all model, feature, and threshold choices before inspecting test results.

## Layout

- src/satmon/: backend module contracts; algorithms not yet implemented.
- configs/: experiment protocol.
- scripts/ and tests/: future commands and synthetic correctness tests.
- data/: provenance instructions; raw/ and processed/ are ignored.
- splits/, results/, report/: assignments, measured artifacts, and AI-use log.
- docs/rebuild.md: removed components, replacements, and next steps.

## Frontend reference

The original web/, static/, and root index.html are retained unchanged as design
references for frontend work after the course deadline. They are not connected to
the new backend. Their demo data and displayed metrics are not experiment results.

## Original application

Preserved in local Git tag `pre-course-rebuild` at commit `66f2807`.
The rebuild branch is `course-backend-rebuild`. Inspect old code with:

```powershell
git show pre-course-rebuild:scripts/detect_zscore.py
```

The old application included synthetic telemetry, simulated metrics, and flawed
evaluation. Its example scores are not research evidence.
