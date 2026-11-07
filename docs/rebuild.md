# Backend replacement plan

Original snapshot: local Git tag pre-course-rebuild, commit 66f2807.
Working branch: course-backend-rebuild. Cleanup is committed separately from main.

| Removed | Reason | Replacement |
| --- | --- | --- |
| demo servers | Simulated responses outside research scope | CLI commands and measured tables/figures |
| api/ | Heuristic targets, random results, leakage | models.py and shared evaluation.py |
| services/ | Database, streaming, advanced models outside scope | File-based inputs and pure backend functions |
| load_telemanom.py, synthetic caches | Generated demo telemetry | Real array and annotation loading in data.py |
| load_nasa_data.py | CSV discovery incompatible with per-channel NPY | Verified acquisition and inventory |
| detect_zscore.py | Database coupling, no budget evaluation | Adapt rolling-statistics idea in baseline.py |
| Isolation Forest and LSTM scripts | Outside required comparison | Logistic Regression and Random Forest |
| advanced_pipeline.py | Future-dependent interpolation, deletes extremes | Causal features preserving anomaly samples |
| evaluate_detection.py | Repeated predictions inflate event recall | Unique-event matching under shared budget |
| DB utilities and monitoring | Operational work outside research scope | Reproducible experiment artifacts |
| old tests | Nonexistent detector class imports | Synthetic tests of actual new functions |
| Docker/deployment/CI | Depend on removed app and inconsistent checks | Focused CI once real tests exist |
| old requirements | Heavy unused frameworks/infrastructure | Minimal pyproject.toml; validated lock later |

Useful z-score and feature ideas remain available in Git. The unrelated resume
PDF was left untouched. No old pipeline is retained as trusted research code.

## Preserved frontend reference

Following the student's clarification, web/, static/, and root index.html were
restored unchanged from pre-course-rebuild. Keep their visual design for frontend
implementation after the course deadline. They are not connected to the new
backend, and their displayed metrics must not be used as experimental evidence.

## Next steps

1. Student implements annotation inventory in src/satmon/data.py. Input: original
   CSV. Output: row/channel/event counts and duplicate/bounds warnings. Parse
   intervals safely; do not silently deduplicate P-2.
2. Load original arrays, preserve sample indices, save channel splits, validate.
3. Implement small causal feature set and overlap labels with boundary tests.
4. Implement shared alerts/event matching, then rolling statistical baseline.
5. Train Logistic Regression and Random Forest; tune on validation channels.
6. Compare PCA and freeze all choices before final test evaluation.
7. Evaluate held-out data, slices, channel bootstrap, failures, recommendation.

Define endpoints, warm-up coverage, thresholds, suppression, ties, partial blocks,
cross-block event identity, and alert timestamp versus observation-window overlap.
Full-block ranking represents retrospective review, not instant online selection.

This cleanup establishes structure only, not a working experiment pipeline.
