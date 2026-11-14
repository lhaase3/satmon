import ast
import numpy as np
import json
from pathlib import Path

def parse_intervals(text: str, n_samples: int) -> list[tuple[int, int]]:
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Invalud interval text") from exc

    if not isinstance(parsed, list):
        raise ValueError("Expected a list of intervals")

    intervals = []

    for interval in parsed:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"Expetced two endpoints, got {interval!r}")

        start, end = interval

        if type(start) is not int or type(end) is not int:
            raise ValueError(f"Endpoints must be integers: {interval!r}")

        if not 0 <= start <= end < n_samples:
            raise ValueError(
                f"Interval {interval!r} is outside valid bounds "
                f"0 through {n_samples - 1}, or is reversed"
            )

        intervals.append((start, end))


    return intervals

def make_sample_labels(n_samples: int, intervals: list[tuple[int, int]],) -> np.ndarray:
    labels = np.zeros(n_samples, dtype=np.int8)

    for start, end in intervals:
        if not 0 <= start <= end < n_samples:
            raise ValueError(f"Invalid interval: ({start}, {end})")
        labels[start: end + 1] = 1

    return labels

def make_window_labels(
    sample_labels: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return prediction indices and overlap labels for complete windows."""
    sample_labels = np.asarray(sample_labels)

    if sample_labels.ndim != 1:
        raise ValueError("sample_labels must be one-dimensional")

    if not np.isin(sample_labels, [0, 1]).all():
        raise ValueError("sample_labels must contain only 0 and 1")

    if type(window_size) is not int:
        raise ValueError("window_size must be an integer")

    if not 1 <= window_size <= len(sample_labels):
        raise ValueError(
            "window_size must be between 1 and the number of samples"
        )
    prediction_indices = np.arange(
        window_size - 1,
        len(sample_labels),
    )
    window_labels = []

    for t in prediction_indices:
        window = sample_labels[t - window_size + 1 : t + 1]

        window_labels.append(int(np.any(window)))

    return prediction_indices, np.asarray(window_labels, dtype=np.int8)

def load_channel_splits(path: str | Path) -> dict[str, list[str]]:
    """Load saved channel assignments and reject overlapping splits."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))

    splits = {
        name: record[name]
        for name in ["train", "validation", "test"]
    }

    for name, channels in splits.items():
        if not isinstance(channels, list) or not channels:
            raise ValueError(f"{name} must be a nonempty list")

        if any(not isinstance(channel, str) or not channel for channel in channels):
            raise ValueError(f"{name} contains an invalid channel ID")

        if len(channels) != len(set(channels)):
            raise ValueError(f"{name} contains duplicate channels")

    for first, second in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
    ]:
        if not set(splits[first]).isdisjoint(splits[second]):
            raise ValueError(f"{first} and {second} share channels")

    return splits


def load_channel_signal(
    test_dir: str | Path,
    channel_id: str,
    expected_length: int,
) -> np.ndarray:
    """Load and validate the telemetry column of one original test array."""
    path = Path(test_dir) / f"{channel_id}.npy"
    array = np.load(path, allow_pickle=False)

    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{channel_id}: expected a two-dimensional array")

    if array.shape[0] != expected_length:
        raise ValueError(
            f"{channel_id}: expected {expected_length} samples, "
            f"found {array.shape[0]}"
        )

    signal = np.asarray(array[:, 0], dtype=float)

    if not np.isfinite(signal).all():
        raise ValueError(f"{channel_id}: telemetry contains nonfinite values")

    return signal


