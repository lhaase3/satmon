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
