"""Fetch and verify the exact Telemanom files required by the experiment."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
)


def matches(path, record):
    return (
        path.is_file()
        and path.stat().st_size == record["bytes"]
        and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "splits/channels.json").read_text(encoding="utf-8"))
    channels = splits["train"] + splits["validation"] + splits["test"]
    paths = ["labeled_anomalies.csv"] + [
        f"archive/data/data/test/{channel}.npy" for channel in channels
    ]
    records = {entry["path"]: entry for entry in manifest["files"]}
    pending = [path for path in paths if not matches(args.output_dir / path, records[path])]
    if not pending:
        print(f"Verified all {len(paths)} required files in {args.output_dir}")
        return

    # Extract only explicitly named inputs; never extract arbitrary archive paths.
    with tempfile.TemporaryDirectory(prefix="satmon-download-") as temporary:
        staging = Path(temporary)
        archive = staging / "dataset.zip"
        print("Downloading Telemanom archive (~86 MB) from Kaggle...", flush=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "SatMon-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as dataset:
            for path in pending:
                member_path = path.removeprefix("archive/")
                candidates = [
                    name for name in dataset.namelist()
                    if name == member_path or name.endswith("/" + member_path)
                ]
                # The annotation may occur in multiple packaged locations.
                valid = None
                for name in candidates:
                    target = staging / PurePosixPath(path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with dataset.open(name) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    if matches(target, records[path]):
                        valid = target
                        break
                if valid is None:
                    raise ValueError(f"Missing or changed dataset file: {path}; refusing unverified data")
        # Verify the complete download before replacing any destination files.
        for path in pending:
            destination = args.output_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / path, destination)
    print(f"Verified all {len(paths)} required files in {args.output_dir}")


if __name__ == "__main__":
    main()
