import numpy as np

from satmon.baseline import rolling_3sigma
from satmon.evaluation import evaluate_channel


def test_score_uses_previous_samples():
    result = rolling_3sigma(
        np.array([0.0, 2.0, 4.0]),
        window_size=2,
    )

    # Past mean = 1, population std = 1; score = |4 - 1| / 1.
    np.testing.assert_array_equal(result.index, [2])
    assert result.loc[2, "score"] == 3.0

    # The rule is strictly greater than 3.
    assert not result.loc[2, "is_candidate"]


def test_constant_signal_has_zero_score():
    result = rolling_3sigma(
        np.array([2.0, 2.0, 2.0, 2.0]),
        window_size=3,
    )

    assert result.loc[3, "score"] == 0.0
    assert not result.loc[3, "is_candidate"]


def test_change_from_constant_reference_is_candidate():
    result = rolling_3sigma(
        np.array([2.0, 2.0, 2.0, 8.0]),
        window_size=3,
    )

    assert np.isfinite(result.loc[3, "score"])
    assert result.loc[3, "is_candidate"]


def test_future_change_does_not_change_earlier_scores():
    original = np.array([0.0, 2.0, 4.0, 1.0, 3.0])
    modified = original.copy()
    modified[4] = 100.0

    before = rolling_3sigma(original, window_size=2)
    after = rolling_3sigma(modified, window_size=2)

    np.testing.assert_array_equal(
        before.loc[[2, 3]].to_numpy(),
        after.loc[[2, 3]].to_numpy(),
    )

def test_baseline_detects_synthetic_event():
    signal = np.array([2.0, 2.0, 2.0, 8.0, 2.0, 2.0])

    scored = rolling_3sigma(signal, window_size=3)
    candidates = scored.loc[scored["is_candidate"]]

    result = evaluate_channel(
        sample_indices=candidates.index.to_numpy(),
        scores=candidates["score"].to_numpy(),
        intervals=[(3, 3)],
    )

    assert result["selected_alerts"] == [3]
    assert result["n_detected_events"] == 1
    assert result["n_missed_events"] == 0
    assert result["n_false_alerts"] == 0
    assert result["event_recall_at_5"] == 1.0