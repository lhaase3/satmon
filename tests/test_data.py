import pytest
import numpy as np

from satmon.data import parse_intervals, make_sample_labels, make_window_labels


def test_preserves_event_order():
    assert parse_intervals("[[7, 9], [2, 4]]", 10) == [
        (7, 9),
        (2, 4),
    ]


def test_accepts_empty_annotations():
    assert parse_intervals("[]", 10) == []


def test_accepts_boundary_indices():
    assert parse_intervals("[[0, 9]]", 10) == [(0, 9)]


@pytest.mark.parametrize(
    "text",
    [
        "not a list",     # Invalid literal
        "42",             # Wrong outer structure
        "[[1]]",          # Missing endpoint
        "[[True, 3]]",    # Boolean endpoint
        "[[1.5, 3]]",     # Noninteger endpoint
        "[[-1, 3]]",      # Negative index
        "[[4, 2]]",       # Reversed interval
        "[[2, 10]]",      # Endpoint outside sequence
    ],
)
def test_rejects_invalid_intervals(text):
    with pytest.raises(ValueError):
        parse_intervals(text, 10)

def test_labels_multiple_events():
    actual = make_sample_labels(10, [(2, 3), (7, 8)])
    expected = [0, 0, 1, 1, 0, 0, 0, 1, 1, 0]
    np.testing.assert_array_equal(actual, expected)


def test_labels_no_events():
    np.testing.assert_array_equal(
        make_sample_labels(4, []),
        [0, 0, 0, 0],
    )


def test_labels_include_first_and_last_samples():
    np.testing.assert_array_equal(
        make_sample_labels(5, [(0, 0), (4, 4)]),
        [1, 0, 0, 0, 1],
    )


@pytest.mark.parametrize("interval", [(-1, 2), (3, 2), (2, 5)])
def test_labels_reject_invalid_bounds(interval):
    with pytest.raises(ValueError):
        make_sample_labels(5, [interval])

def test_window_labels_preserve_prediction_indices():
    sample_labels = np.array([0, 0, 0, 1, 1, 0, 0, 0])

    indices, labels = make_window_labels(sample_labels, 3)

    np.testing.assert_array_equal(indices, [2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(labels, [0, 1, 1, 1, 1, 0])

@pytest.mark.parametrize("window_size", [0, -1, 4, 1.5, True])
def test_window_labels_reject_invalid_sizes(window_size):
    with pytest.raises(ValueError):
        make_window_labels(np.array([0, 1, 0]), window_size)

def test_window_size_one_matches_sample_labels():
    samples = np.array([0, 1, 0])

    indices, labels = make_window_labels(samples, 1)

    np.testing.assert_array_equal(indices, [0, 1, 2])
    np.testing.assert_array_equal(labels, samples)


def test_window_covering_entire_sequence():
    indices, labels = make_window_labels(
        np.array([0, 1, 0]),
        3,
    )

    np.testing.assert_array_equal(indices, [2])
    np.testing.assert_array_equal(labels, [1])