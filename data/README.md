# Research data

Real telemetry was downloaded by the student during the November 7 session.
The archive version and exact download timestamp were not captured.
raw/ and processed/ are ignored by Git. manifest.json records local NPY and
annotation paths, byte sizes, and SHA-256 hashes as of November 10; original
train/test arrays also have shapes and dtypes recorded. Hashes identify local
bytes, not an independently authenticated release.

## Acquisition and preparation

Original research/code: https://github.com/khundman/telemanom
Dataset distribution used:
https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

From the repository root, run `python scripts/download_data.py`. This downloads
the public Kaggle ZIP without an API key and installs the 80 experiment channels
from the original test directory plus the annotations under data/raw/. It checks
every installed file against manifest.json and refuses changed or missing data.
Matching existing inputs are reused. The original archive also contains training
arrays and other artifacts, which this script does not install because they are
not used by this experiment. Historical local paths:

- raw/labeled_anomalies.csv: annotations used by the experiment.
- raw/archive/labeled_anomalies.csv: packaged copy; hashes match.
- raw/archive/data/data/train/: 82 original training arrays.
- raw/archive/data/data/test/: 82 original test arrays.

All supervised partitions use annotated original test sequences; their assignments
are saved in splits/channels.json. P-2 is excluded for conflicting duplicate rows.
P-1 is reserved for training following exploration. 80 channels are eligible.
Do not treat the number of downloaded arrays as the number of eligible annotated
channels. See the root README for the runner and annotation-hash verification.

## Distribution terms: verification incomplete

The code repository includes Caltech copyright and redistribution conditions:
https://github.com/khundman/telemanom/blob/master/LICENSE.txt
Do not infer dataset terms from this software license or SatMon's former MIT
badge. The Kaggle page displays “Data files © Original Authors”
(student-verified November 10, 2026). This does not identify a standard
reuse license. Dataset redistribution permissions remain unverified;
raw data is excluded from this repository. Users obtain it from the
linked distribution page. The upstream software's Caltech terms are
recorded separately and are not assumed to license the data.

Source to verify: https://github.com/khundman/telemanom
Annotations: labeled_anomalies.csv; refer to original test sequences.

The audit found duplicate P-2 rows, event-type metadata, and upstream documentation
stating scaling uses original test-set extrema. Recheck against acquired files;
document the future-information limitation. Do not infer physical timing or units
from anonymized channel names. Original train/test directories are distinct from
our supervised channel split. Removed synthetic NPZ files were not NASA data.
