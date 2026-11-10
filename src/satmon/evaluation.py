import numpy as np

def match_events(
        alert_indices: list[int],
        intervals: list[tuple[int, int]],
) -> list[bool]:
    detected = []

    for start, end in intervals:
        detected.append(
            any(start <= t <= end for t in alert_indices)
        )

    return detected

def select_alerts(
    sample_indices,
    scores,
    block_size=1000,
    max_alerts=5,
    suppression_distance=10,
) -> list[int]:
    """Select ranked, separated alerts within each monitoring block.

    Call separately for each channel. Inputs must contain candidate alerts only.
    Suppression applies within each block and includes the distance boundary.
    """
    indices = np.asarray(sample_indices)
    scores = np.asarray(scores, dtype=float)

    if indices.ndim != 1 or scores.ndim != 1:
        raise ValueError("Indices and scores must be one-dimensional")

    if len(indices) != len(scores):
        raise ValueError("Indices and scores must have equal lengths")

    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    if type(max_alerts) is not int or not 0 <= max_alerts <= 5:
        raise ValueError("max_alerts must be an integer from 0 to 5")

    if (
        type(suppression_distance) is not int
        or suppression_distance < 0
    ):
        raise ValueError(
            "suppression_distance must be a nonnegative integer"
        )

    if len(indices) == 0:
        return []

    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Sample indices must be integers")

    if np.any(indices < 0):
        raise ValueError("Sample indices must be nonnegative")

    if len(np.unique(indices)) != len(indices):
        raise ValueError("Sample indices must be unique")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")

    if max_alerts == 0:
        return []

    block_ids = indices // block_size
    selected = []

    for block_id in np.unique(block_ids):
        positions = np.flatnonzero(block_ids == block_id)

        # Higher score first; earlier sample wins a score tie.
        ranked_positions = sorted(
            positions,
            key=lambda position: (
                -scores[position],
                int(indices[position]),
            ),
        )

        block_alerts = []

        for position in ranked_positions:
            candidate = int(indices[position])

            too_close = any(
                abs(candidate - chosen) <= suppression_distance
                for chosen in block_alerts
            )

            if too_close:
                continue

            block_alerts.append(candidate)

            if len(block_alerts) == max_alerts:
                break

        selected.extend(block_alerts)

    return sorted(selected)


def evaluate_channel(
    sample_indices,
    scores,
    intervals: list[tuple[int, int]],
    suppression_distance: int = 10,
) -> dict:
    """Evaluate candidate alerts for one channel under the fixed budget."""
    selected = select_alerts(
        sample_indices,
        scores,
        block_size=1000,
        max_alerts=5,
        suppression_distance=suppression_distance,
    )

    detected = match_events(selected, intervals)

    false_alerts = sum(
        not any(start <= t <= end for start, end in intervals)
        for t in selected
    )

    n_events = len(intervals)
    n_detected = sum(detected)

    return {
        "selected_alerts": selected,
        "event_detected": detected,
        "n_events": n_events,
        "n_detected_events": n_detected,
        "n_missed_events": n_events - n_detected,
        "n_false_alerts": false_alerts,
        "event_recall_at_5": (
            n_detected / n_events if n_events else None
        ),
    }

def evaluate_scored_channels(
    scored_examples,
    annotations,
    threshold: float,
    suppression_distance: int = 10,
) -> dict:
    """Pool event counts across channels using one candidate threshold."""
    from satmon.data import parse_intervals

    total_events = 0
    total_detected = 0
    total_alerts = 0
    total_false = 0

    if not annotations["chan_id"].is_unique:
        raise ValueError("Annotations must contain one row per channel")

    if set(scored_examples["channel_id"]) != set(annotations["chan_id"]):
        raise ValueError("Scored and annotated channel sets must match")

    for _, annotation in annotations.iterrows():
        channel_id = annotation["chan_id"]
        rows = scored_examples.loc[
            scored_examples["channel_id"] == channel_id
        ]

        intervals = parse_intervals(
            annotation["anomaly_sequences"],
            int(annotation["num_values"]),
        )

        candidates = rows.loc[rows["score"] > threshold]

        result = evaluate_channel(
            sample_indices=candidates["sample_index"].to_numpy(),
            scores=candidates["score"].to_numpy(),
            intervals=intervals,
            suppression_distance=suppression_distance,
        )

        total_events += result["n_events"]
        total_detected += result["n_detected_events"]
        total_alerts += len(result["selected_alerts"])
        total_false += result["n_false_alerts"]

    return {
        "threshold": threshold,
        "events": total_events,
        "detected": total_detected,
        "event_recall_at_5": (
            total_detected / total_events if total_events else None
        ),
        "selected_alerts": total_alerts,
        "false_alerts": total_false,
        "secondary_cost": (
            10 * (total_events - total_detected) + total_false
        ),
    }
