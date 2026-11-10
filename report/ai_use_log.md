# AI assistance log

## 2026-11-06: audit and backend cleanup

AI reviewed source and identified synthetic data, heuristic labels, leakage,
row-level splitting, and incorrect event recall. These are source observations,
not runtime findings. At the student's explicit request, AI removed the legacy
frontend, APIs, database/deployment infrastructure, out-of-scope models, demo data,
and old tests. AI created the package scaffold, config, packaging, documentation,
and replacement plan; the student did not author those changes.

Original code is preserved in local Git tag pre-course-rebuild. No algorithms
were implemented or experimental results generated. A recursive PowerShell
deletion was blocked by command policy; tracked files were removed with git rm.
No unresolved approval block remains.

Record actual contributions, verification, and corrections as work continues.

### Frontend preservation correction

The student clarified that the existing frontend should remain as a future design
reference. AI restored web/, static/, and root index.html unchanged from the
original Git snapshot. Removing these assets exceeded the student's intended
scope; backend work remains the current priority.

### Git isolation

At the student's request, AI committed the cleanup on course-backend-rebuild and
prepared to push that branch and the pre-course-rebuild backup tag to GitHub.
The original main branch was not modified.



## 2026-11-07–08: data preparation and baseline

- I inspected real Telemanom arrays and annotations in a notebook.
- Both annotation copies contained conflicting P-2 rows. Following an
  AI recommendation, I excluded P-2 and documented the decision.
- P-1 was reserved for training because we explored it before splitting.
- We saved a spacecraft-stratified 48/16/16 channel split with seed 42.
- AI supplied code for annotation parsing, label construction, causal
  features, alert selection, event matching, baseline scoring, and tests.
  I entered/adapted the code and ran the checks; it was not independently
  authored entirely by me.
- The test suite reached 50 passing tests. These verify selected behaviors,
  not the statistical validity of the entire experiment.
- The rolling 3σ training diagnostic detected 58/65 events, with 1,019
  false alerts among 1,221 selected alerts.
- C-2 inspection distinguished a budget/ranking miss from a threshold miss.
- AI guidance initially used an incorrect archive path and later suggested
  an array-length reference that belonged to another notebook loop. We
  identified these through errors and corrected them.
- Validation and final test performance have not been evaluated.

## 2026-11-08–09: supervised models and validation selection

- The previous entry describes the state before validation experiments. Validation
  has now been used for selection; final test performance remains unevaluated.
- AI supplied model builders, multi-channel assembly and evaluation helpers,
  notebook experiments, tuning loops, and interpretations. The student entered
  and ran this code and shared the outputs; this is substantial AI-generated code.
- We compared Logistic Regression C values 0.01, 1, and 100, four forest
  depth/leaf configurations, probability thresholds, and windows 50 and 100.
  Some threshold values were chosen after inspecting validation probabilities.
  This was exploratory, staged validation selection, not a predefined joint grid.
- AI identified an initial comparison mismatch: baseline predictions began at 50
  while supervised rows began at 49. Window comparisons were rerun with a common
  prediction start of 100. Earlier diagnostics are not the final comparison.
- Selected window: 100. Original-feature Logistic Regression: C=1, threshold 0.13.
  Forest: 100 trees, depth 8, minimum leaf size 100, threshold 0.25.
  The baseline retains its strict 3-sigma threshold and prior-only reference.
- User-reported common-timeline validation results: baseline 17/19 events and
  181 false alerts; forest 16/19 and 177; Logistic Regression 15/19 and 242.
- PCA was repeated at window 100. Two components retained about 83.7% variance,
  detecting 15/19 events with 227 false alerts; three components matched original
  Logistic Regression aggregate counts. We retained direct feature interpretation
  for the primary Logistic Regression comparison and recorded the PCA tradeoff.
- These are validation observations, not evidence of held-out generalization.
  No bootstrap uncertainty or final deployment recommendation exists yet.
- AI recorded the selected protocol and closeout notes. Before final testing,
  review the settings and reproduce the notebook in a clean kernel. Do not use
  cached earlier window-50 model objects for the selected window-100 experiment.
