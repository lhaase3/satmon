import numpy as np

from satmon.features import make_basic_features


def test_basic_features_preserve_values_and_indices():
    features = make_basic_features(
        np.array([2.0, 3.0, 3.0, 1.0])
    )

    np.testing.assert_array_equal(features.index, [1, 2, 3])
    np.testing.assert_array_equal(features["value"], [3.0, 3.0, 1.0])
    np.testing.assert_array_equal(features["delta"], [1.0, 0.0, -2.0])


def test_future_values_do_not_change_past_features():
    original = np.array([2.0, 3.0, 3.0, 1.0])
    modified = original.copy()
    modified[3] = 100.0

    before = make_basic_features(original)
    after = make_basic_features(modified)

    np.testing.assert_array_equal(
        before.loc[[1, 2]].to_numpy(),
        after.loc[[1, 2]].to_numpy(),
    )