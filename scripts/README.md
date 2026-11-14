# Experiment commands

`python scripts/download_data.py` downloads the public dataset and verifies the
81 required input files against `data/manifest.json`. It reuses matching inputs;
`--output-dir` selects an alternate destination. See the root README for setup.

run_experiment.py fits frozen models and exports primary counts; see the root
README for data, split, protocol, and training-order arguments.

Reproduce overall paired channel-bootstrap tables from the regenerated counts:

```powershell
.\.venv\Scripts\python.exe scripts/bootstrap_results.py `
  --channels results/reproduction/channels.csv `
  --events results/reproduction/events.csv `
  --output-dir results/reproduction `
  --reference-dir results/final_test
```

Defaults: 5,000 draws, seed 42, 95% percentile intervals. Identical channel draws
are used across methods. This conditions on fitted models and does not include
retraining uncertainty. With --events, it also reproduces spacecraft and event-type
slice summaries and intervals. Event-type draws condition on channels containing
that type. Degenerate perfect-recall intervals do not prove population perfection.
The runner also exports events.csv and alerts.csv so event identities and selected
positions can be compared, plus frozen scores for the two failure-plot examples.

After rerunning run_experiment.py, reproduce the failure figure:

```powershell
.\.venv\Scripts\python.exe scripts/plot_failures.py `
  --results-dir results/reproduction `
  --output-dir results/reproduction/figures
```

This redraws the post-test diagnostic examples; it does not fit or tune a model.
The script checks that neither annotated event exceeds its frozen threshold and
writes a diagnostic CSV. Rendering can differ across Matplotlib/font versions;
the original figure in results/figures/ remains untouched.
