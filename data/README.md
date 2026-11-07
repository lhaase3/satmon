# Research data

Real telemetry has not been downloaded or validated in this rebuild.
Use raw/ for original files and processed/ for generated tables; both are ignored.
Record source version, download date, hashes, and distribution/license terms.

Source to verify: https://github.com/khundman/telemanom
Annotations: labeled_anomalies.csv; refer to original test sequences.

The audit found duplicate P-2 rows, event-type metadata, and upstream documentation
stating scaling uses original test-set extrema. Recheck against acquired files;
document the future-information limitation. Do not infer physical timing or units
from anonymized channel names. Original train/test directories are distinct from
our supervised channel split. Removed synthetic NPZ files were not NASA data.
