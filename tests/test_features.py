import numpy as np

from satmon.features import make_basic_features, make_rolling_features, make_channel_examples


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

def test_rolling_features_do_not_use_future_values():
    original = np.array([2.0, 3.0, 3.0, 1.0, 2.0])
    modified = original.copy()
    modified[4] = 100.0

    before = make_rolling_features(original, window_size=3)
    after = make_rolling_features(modified, window_size=3)

    # Changing sample 4 must not affect features at samples 2 or 3.
    np.testing.assert_array_equal(
        before.loc[[2, 3]].to_numpy(),
        after.loc[[2, 3]].to_numpy(),
    )


def test_channel_examples_align_features_and_labels():
    examples = make_channel_examples(
        signal=np.array([2.0, 3.0, 3.0, 1.0, 2.0, 2.0, 2.0]),
        intervals=[(2, 3)],
        channel_id="DEMO",
        window_size=3,
    )

    np.testing.assert_array_equal(
        examples.index, [2, 3, 4, 5, 6]
    )
    np.testing.assert_array_equal(
        examples["value"], [3.0, 1.0, 2.0, 2.0, 2.0]
    )
    np.testing.assert_array_equal(
        examples["delta"], [0.0, -2.0, 1.0, 0.0, 0.0]
    )
    np.testing.assert_array_equal(
        examples["label"], [1, 1, 1, 1, 0]
    )

    np.testing.assert_allclose(
        examples["rolling_mean"],
        [8 / 3, 7 / 3, 2.0, 5 / 3, 2.0],
    )
    np.testing.assert_allclose(
        examples["rolling_std"],
        [np.sqrt(2 / 9), np.sqrt(8 / 9), np.sqrt(2 / 3),
         np.sqrt(2 / 9), 0.0],
    )

    assert examples["channel_id"].eq("DEMO").all()
    assert not examples.isna().any().any()
    assert examples.index.name == "sample_index"