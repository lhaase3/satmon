import numpy as np
import pandas as pd
from satmon.data import make_sample_labels, make_window_labels, load_channel_signal, parse_intervals
from pathlib import Path

def make_basic_features(signal: np.ndarray) -> pd.DataFrame:
    """Return current value and change, indexed by original sample position."""
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    if len(signal) < 2:
        raise ValueError("signal must contain at least two samples")

    if not np.isfinite(signal).all():
        raise ValueError("signal must contain only finite values")

    return pd.DataFrame(
        {
            "value": signal[1:],
            "delta": np.diff(signal),
        },
        index=pd.Index(
            np.arange(1, len(signal)),
            name="sample_index",
        ),
    )


def make_rolling_features(
    signal: np.ndarray,
    window_size: int,
) -> pd.DataFrame:
    """Return causal, current-inclusive rolling mean and population std."""
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    if not np.isfinite(signal).all():
        raise ValueError("signal must contain only finite values")

    if type(window_size) is not int:
        raise ValueError("window_size must be an integer")

    if not 1 <= window_size <= len(signal):
        raise ValueError(
            "window_size must be between 1 and the number of samples"
        )

    series = pd.Series(signal)
    rolling = series.rolling(
        window=window_size,
        min_periods=window_size,
    )

    features = pd.DataFrame({
        "rolling_mean": rolling.mean(),
        "rolling_std": rolling.std(ddof=0),
    })

    features = features.iloc[window_size - 1:].copy()
    features.index.name = "sample_index"

    return features



def make_channel_examples(
    signal: np.ndarray,
    intervals: list[tuple[int, int]],
    channel_id: str,
    window_size: int,
) -> pd.DataFrame:
    """Assemble aligned features and overlap labels for one channel."""
    basic = make_basic_features(signal)
    rolling = make_rolling_features(signal, window_size)

    # Inner join retains only indices available in both feature tables.
    examples = basic.join(rolling, how="inner")

    sample_labels = make_sample_labels(len(signal), intervals)
    prediction_indices, window_labels = make_window_labels(
        sample_labels,
        window_size,
    )

    labels_by_index = pd.Series(
        window_labels,
        index=prediction_indices,
        name="label",
    )

    examples = examples.join(labels_by_index, how="inner")
    examples["channel_id"] = channel_id

    return examples


def assemble_split_examples(
    annotations: pd.DataFrame,
    test_dir: str | Path,
    window_size: int,
    first_prediction_index: int,
) -> pd.DataFrame:
    """Assemble channel examples using an explicit common prediction start."""
    if annotations.empty or not annotations["chan_id"].is_unique:
        raise ValueError("Expected nonempty, unique channel annotations")

    if (
        type(first_prediction_index) is not int
        or first_prediction_index < max(1, window_size - 1)
    ):
        raise ValueError("Prediction start must cover feature warm-up")

    tables = []

    for _, row in annotations.iterrows():
        signal = load_channel_signal(
            test_dir,
            channel_id=row["chan_id"],
            expected_length=int(row["num_values"]),
        )

        intervals = parse_intervals(
            row["anomaly_sequences"],
            len(signal),
        )

        examples = make_channel_examples(
            signal,
            intervals,
            channel_id=row["chan_id"],
            window_size=window_size,
        ).reset_index()

        examples = examples.loc[
            examples["sample_index"] >= first_prediction_index
        ].copy()

        if examples.empty:
            raise ValueError(
                f"{row['chan_id']}: no eligible prediction rows"
            )

        tables.append(examples)

    return pd.concat(tables, ignore_index=True)