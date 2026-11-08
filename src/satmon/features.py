import numpy as np
import pandas as pd


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
