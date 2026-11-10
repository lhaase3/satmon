# Validation selection checkpoint — 2026-11-09

Selected procedure is recorded in configs/protocol.json. This file documents the
decision before final test evaluation, not a completed test experiment.

| Method | Window | Threshold | Validation detected/events | False alerts |
| --- | --- | --- | --- | --- |
| Rolling 3-sigma | 100 | >3 | 17/19 | 181 |
| Original-feature Logistic Regression, C=1 | 100 | >0.13 | 15/19 | 242 |
| Forest, depth=8, leaf=100, trees=100 | 100 | >0.25 | 16/19 | 177 |
| PCA(2) plus Logistic Regression, C=1 | 100 | >0.13 | 15/19 | 227 |

Primary selection uses event recall, then false alerts for ties. Baseline is the
validation leader. PCA reduced logistic false alerts without increasing recall;
retain original features for the interpretable primary comparison. This is not
an uncertainty-qualified claim that PCA never helps.

Source tables: results/model_selection/. Window-50 and initial training tables
are exploratory history and must not be presented as the final common-timeline
comparison. Hyperparameters were selected in stages, not in a full joint search.

## Before opening final test results

1. Save the notebook and restart/run all in the project environment. Confirm saved
   tables reproduce; do not continue to final testing on a failed rerun.
2. Review configs/protocol.json. Keep the exact channel assignments unchanged.
3. Use window-100 models trained on training channels only. In the current
   notebook these are window_models[(100, "logistic")] and
   window_models[(100, "forest")], not older selected_logistic/selected_forest
   objects from window 50. No training-plus-validation refit is planned.
4. Check final arrays/annotations for integrity without tuning against their
   patterns. If corrupt inputs or implementation errors arise, document them.
5. Produce per-channel/per-event records for all three primary methods, then
   whole-channel bootstrap intervals and predefined spacecraft/type slices.
   Do not change settings after seeing test performance.

## Limitations to carry into the report

- Upstream test-extrema scaling is an online-causality limitation that our causal
  feature checks cannot remove.
- Inclusive endpoints are an explicit project convention, supported by upstream
  interval representation rather than independently verified annotation history.
- Full-block ranking represents retrospective block review, not instantaneous
  online alert delivery. Suppression does not cross block boundaries.
- Overlap-positive targets extend after event ends; detection credits only alert
  timestamps inside original events. All annotated events remain in denominator.
- Excluding P-2 and reserving explored P-1 for training must be disclosed.
- No standalone reproduction command, validated dependency lock, bootstrap
  analysis, or final test results exist yet. These remain required work.
