from satmon.evaluation import match_events, select_alerts, evaluate_channel


def test_duplicate_alerts_detect_event_once():
    assert match_events([3, 4, 12], [(2, 5), (8, 10)]) == [
        True,
        False,
    ]


def test_includes_both_event_endpoints():
    assert match_events([2, 10], [(2, 5), (8, 10)]) == [
        True,
        True,
    ]


def test_alert_after_event_gets_no_credit():
    assert match_events([6], [(2, 5)]) == [False]


def test_no_alerts_leaves_events_undetected():
    assert match_events([], [(2, 5), (8, 10)]) == [
        False,
        False,
    ]


def test_no_events_returns_empty_list():
    assert match_events([3], []) == []


def test_event_crossing_block_boundary_is_one_event():
    assert match_events([999, 1001], [(995, 1005)]) == [True]

def test_selects_five_highest_scores():
    assert select_alerts(
        [10, 30, 50, 70, 90, 110],
        [0.1, 0.6, 0.3, 0.5, 0.2, 0.4],
        suppression_distance=0,
    ) == [30, 50, 70, 90, 110]


def test_suppresses_nearby_weaker_candidates():
    assert select_alerts(
        [100, 103, 200],
        [0.9, 0.8, 0.7],
    ) == [100, 200]


def test_score_ties_favor_earlier_sample():
    assert select_alerts(
        [103, 100],
        [0.9, 0.9],
    ) == [100]


def test_budget_resets_at_block_boundary():
    assert select_alerts(
        [998, 999, 1000, 1001],
        [0.8, 0.9, 0.7, 0.6],
        max_alerts=1,
        suppression_distance=0,
    ) == [999, 1000]


def test_suppression_does_not_cross_blocks():
    assert select_alerts(
        [999, 1000],
        [0.9, 0.8],
    ) == [999, 1000]


def test_fewer_than_five_candidates():
    assert select_alerts(
        [20, 80],
        [0.5, 0.7],
    ) == [20, 80]


def test_no_candidates_returns_no_alerts():
    assert select_alerts([], []) == []


def test_budgeted_alerts_detect_unique_events():
    selected = select_alerts(
        sample_indices=[100, 103, 300, 999, 1001, 1500],
        scores=[0.9, 0.8, 0.7, 0.95, 0.85, 0.6],
    )

    events = [
        (95, 110),      # Nearby duplicate candidates
        (990, 1010),    # One event crossing a block boundary
        (1700, 1750),   # Missed event
    ]

    detected = match_events(selected, events)

    assert selected == [100, 300, 999, 1001, 1500]
    assert detected == [True, True, False]
    assert sum(detected) / len(events) == 2 / 3

def test_evaluate_channel_counts():
    result = evaluate_channel(
        sample_indices=[100, 103, 300, 999, 1001, 1500],
        scores=[0.9, 0.8, 0.7, 0.95, 0.85, 0.6],
        intervals=[(95, 110), (990, 1010), (1700, 1750)],
    )

    assert result["selected_alerts"] == [100, 300, 999, 1001, 1500]
    assert result["event_detected"] == [True, True, False]
    assert result["n_events"] == 3
    assert result["n_detected_events"] == 2
    assert result["n_missed_events"] == 1
    assert result["n_false_alerts"] == 2
    assert result["event_recall_at_5"] == 2 / 3


def test_evaluate_channel_without_events():
    result = evaluate_channel(
        sample_indices=[20, 80],
        scores=[0.5, 0.7],
        intervals=[],
    )

    assert result["event_detected"] == []
    assert result["n_events"] == 0
    assert result["n_detected_events"] == 0
    assert result["n_missed_events"] == 0
    assert result["n_false_alerts"] == 2
    assert result["event_recall_at_5"] is None


def test_evaluate_channel_without_candidates():
    result = evaluate_channel(
        sample_indices=[],
        scores=[],
        intervals=[(10, 20)],
    )

    assert result["selected_alerts"] == []
    assert result["event_detected"] == [False]
    assert result["n_missed_events"] == 1
    assert result["n_false_alerts"] == 0
    assert result["event_recall_at_5"] == 0.0