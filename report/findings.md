# Measured findings — 2026-11-10

Working notes for the six-page report, not a finished report or new experiment.
Sources: results/final_test/, results/model_selection/, configs/protocol.json.

## Decision and comparison (R1–R3)

Analyst budget: at most five alerts per channel per 1,000 original samples.
One missed event costs ten false-alert investigations in the secondary summary.
Primary selection remains event recall. Ranking is retrospective within blocks.

| Method | Detected/events | Recall | 95% channel-bootstrap interval | False alerts | Selected alerts | Secondary cost |
| --- | --- | --- | --- | --- | --- | --- |
| Rolling 3-sigma | 17/19 | 89.5% | 76.2–100% | 146 | 184 | 166 |
| Random Forest | 15/19 | 78.9% | 63.6–94.1% | 297 | 370 | 337 |
| Logistic Regression | 13/19 | 68.4% | 45.0–88.9% | 316 | 403 | 376 |

There are 16 test channels and 96,988 eligible feature rows. Overlapping rows
are not independent evidence. Bootstrap resamples whole channels, 5,000 times,
seed 42, with identical draws across methods for paired comparisons.
Baseline-minus-forest recall difference: +10.5 percentage points, interval
[-10.5, 31.6]. Baseline-minus-logistic: +21.1 points, interval [-5.9, 52.4].
Both intervals include zero; no clear recall superiority is established.
Intervals are conditional on fitted models, not retraining or split selection.

## PCA decision (R4)

Window-100 validation: two PCA components retained 83.7% training variance and
detected 15/19 events with 227 false alerts, compared with 242 for original-feature
Logistic Regression at equal recall. Three components retained 99.0% and matched
original aggregate outcomes. This did not change the preferred approach. Retain
original-feature Logistic Regression for direct feature interpretation; report
the observed PCA reduction without claiming it is reliable or universal.

## Predefined failure slices (R5)

See spacecraft_slices.csv and event_type_slices.csv for all methods and intervals.
Baseline MSL: 7/9 events, six channels, 16,786 samples, recall 77.8%, interval
[50%, 100%]. Baseline SMAP: 10/10 events, ten channels, 81,802 samples.
Baseline's weakest predefined slice is contextual: 5/7 events across five
channels, recall 71.4%, interval [40%, 100%]. Point: 12/12 across eleven channels.
Perfect observed recall yields a degenerate empirical interval; it does not
establish perfect population detection. Lowest method/type recall is logistic
contextual: 4/7, interval [0%, 90.9%]. Type counts concern labeled events, not a
partition of all normal telemetry samples or false alerts.

Observed misses, investigated after testing without changing settings:

- F-7 [1250,1450]: two baseline candidates; strongest score 3.256. Five stronger
  separated out-of-event candidates exhausted the block budget (lowest selected
  score 3.979). Event candidates were not near selected alerts.
- T-12 [630,750]: baseline maximum score 2.470, below threshold 3; no candidates.
- A-9 [4569,8433]: logistic maximum event probability 0.129217, below threshold
  0.13; no candidates. This is an overlap-window probability, not an instantaneous
  anomaly probability. No threshold adjustment was made after this observation.

These findings identify algorithmic failure stages, not spacecraft fault causes.
Figure: results/figures/held_out_failure_examples.png.

## Recommendation and reversal condition (R6)

Recommend the statistical baseline as the candidate for a controlled analyst
review pilot under these assumptions, given its observed recall, lower alert
burden, and simplicity. This is not validation for flight-critical deployment.
Reverse the recommendation if a frozen flexible model, evaluated on additional
independent labeled channels, shows a positive paired recall improvement interval
while meeting the same budget and offering an acceptable false-alert burden.

## Required qualifications and remaining work

- Upstream test-extrema scaling limits online-causality claims; raw physical timing
  and units are not established. P-2 excluded; explored P-1 reserved for training.
- Exploratory, staged validation search, not full joint optimization. Labeled
  event types retain source definitions; do not equate point with one sample.
- Endpoints inclusive by project convention. Warm-up events retained. Suppression
  within blocks only. Frozen settings are in configs/protocol.json.
- Validation 32% baseline false-alert reduction is not a held-out comparison.
- November 14 update: standalone test evaluation, bootstrap, failure-plot, and
  hash-verified data-download commands are available in the root README. A fresh
  Windows Python 3.12.10 environment reproduced the headline counts and bootstrap
  tables. Dependency and local dataset hash records are included; the original
  dataset release version remains unknown. Final report writing remains separate.
