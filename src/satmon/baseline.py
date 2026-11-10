import numpy as np
import pandas as pd


def rolling_3sigma(
    signal: np.ndarray,
    window_size: int,
) -> pd.DataFrame:
    """Score each sample against the previous window_size observations."""
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    if not np.isfinite(signal).all():
        raise ValueError("signal must contain only finite values")

    if type(window_size) is not int:
        raise ValueError("window_size must be an integer")

    if not 2 <= window_size < len(signal):
        raise ValueError(
            "window_size must be at least 2 and smaller than the signal"
        )

    series = pd.Series(signal)

    # At index t, shifted contains x[t - 1], excluding the current sample.
    past = series.shift(1).rolling(
        window=window_size,
        min_periods=window_size,
    )

    past_mean = past.mean().iloc[window_size:]
    past_std = past.std(ddof=0).iloc[window_size:]
    current = series.iloc[window_size:]

    deviation = np.abs(
        current.to_numpy() - past_mean.to_numpy()
    )
    std = past_std.to_numpy()

    scores = np.zeros(len(current), dtype=float)

    np.divide(
        deviation,
        std,
        out=scores,
        where=std > 0,
    )

    # A change from an exactly constant reference receives maximum severity.
    changed_from_constant = (std == 0) & (deviation > 0)
    scores[changed_from_constant] = np.finfo(float).max

    return pd.DataFrame(
        {
            "score": scores,
            "is_candidate": scores > 3.0,
        },
        index=pd.Index(
            np.arange(window_size, len(signal)),
            name="sample_index",
        ),
    )
