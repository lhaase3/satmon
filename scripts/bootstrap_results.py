"""Reproduce paired whole-channel recall intervals without model fitting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--events", type=Path,
                        help="Include spacecraft and event-type slice tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()

    if args.draws < 1:
        raise ValueError("draws must be positive")

    channels = pd.read_csv(args.channels)
    methods = ["baseline", "forest", "logistic"]
    if channels.empty or channels.duplicated(["channel", "method"]).any():
        raise ValueError("Expected nonempty, unique channel/method records")
    if set(channels["method"]) != set(methods):
        raise ValueError("Expected exactly the three primary methods")

    events = channels.pivot(
        index="channel", columns="method", values="n_events"
    ).reindex(columns=methods).sort_index()
    detected = channels.pivot(
        index="channel", columns="method", values="n_detected_events"
    ).reindex(index=events.index, columns=methods)

    if events.isna().any().any() or detected.isna().any().any():
        raise ValueError("Every channel must have results for all methods")
    for table in [events, detected]:
        values = table.to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values != np.floor(values)).any()):
            raise ValueError("Event counts must be finite nonnegative integers")
    if (detected > events).any().any():
        raise ValueError("Detected events cannot exceed event counts")
    for method in methods:
        if not events[method].equals(events["baseline"]):
            raise ValueError("Methods must use identical event denominators")

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(events), size=(args.draws, len(events)))
    event_totals = events.to_numpy()[draws].sum(axis=1)
    detected_totals = detected.to_numpy()[draws].sum(axis=1)
    if (event_totals == 0).any():
        raise ValueError("Zero-event bootstrap draw; define handling explicitly")
    recalls = detected_totals / event_totals
    observed = detected.sum() / events.sum()

    interval_rows = []
    for column, method in enumerate(methods):
        lower, upper = np.quantile(recalls[:, column], [0.025, 0.975])
        interval_rows.append({
            "method": method,
            "event_recall_at_5": observed[method],
            "ci_lower": lower,
            "ci_upper": upper,
        })

    difference_rows = []
    for other in ["forest", "logistic"]:
        differences = recalls[:, 0] - recalls[:, methods.index(other)]
        lower, upper = np.quantile(differences, [0.025, 0.975])
        difference_rows.append({
            "comparison": f"baseline minus {other}",
            "observed_recall_difference": observed["baseline"] - observed[other],
            "ci_lower": lower,
            "ci_upper": upper,
        })

    outputs = {
        "recall_intervals.csv": pd.DataFrame(interval_rows),
        "paired_recall_differences.csv": pd.DataFrame(difference_rows),
    }
    if args.events:
        event_rows = pd.read_csv(args.events)
        if event_rows.duplicated(["method", "channel", "event_id"]).any():
            raise ValueError("Duplicate event records")
        if not event_rows["detected"].isin([True, False]).all():
            raise ValueError("Event outcomes must be Boolean")
        event_counts = event_rows.groupby(["method", "channel"]).agg(
            n_events=("detected", "size"), n_detected_events=("detected", "sum")
        ).sort_index()
        expected_counts = channels.set_index(["method", "channel"])[
            ["n_events", "n_detected_events"]
        ].sort_index()
        pd.testing.assert_frame_equal(event_counts, expected_counts)

        spacecraft = channels.groupby(["method", "spacecraft"]).agg(
            n_channels=("channel", "nunique"),
            n_samples=("n_samples", "sum"),
            n_scored_samples=("n_scored_samples", "sum"),
            n_events=("n_events", "sum"),
            n_detected_events=("n_detected_events", "sum"),
            n_false_alerts=("n_false_alerts", "sum"),
        ).reset_index()
        types = event_rows.groupby(["method", "event_type"]).agg(
            n_channels=("channel", "nunique"),
            n_events=("detected", "size"),
            n_detected_events=("detected", "sum"),
        ).reset_index()

        # Each slice family restarts seed 42, matching the original notebook.
        # Event-type bootstraps condition on channels containing that type.
        for family, labels, summary, filename in [
            ("spacecraft", ["MSL", "SMAP"], spacecraft, "spacecraft_slices.csv"),
            ("event_type", ["contextual", "point"], types, "event_type_slices.csv"),
        ]:
            slice_rng = np.random.default_rng(args.seed)
            intervals = []
            for label in labels:
                if family == "spacecraft":
                    counts = channels.loc[channels[family] == label]
                else:
                    counts = event_rows.loc[event_rows[family] == label].groupby(
                        ["channel", "method"]
                    ).agg(n_events=("detected", "size"),
                          n_detected_events=("detected", "sum")).reset_index()
                denominators = counts.pivot(
                    index="channel", columns="method", values="n_events"
                ).reindex(columns=methods).sort_index()
                numerators = counts.pivot(
                    index="channel", columns="method", values="n_detected_events"
                ).reindex(index=denominators.index, columns=methods)
                if (denominators.empty or denominators.isna().any().any()
                        or numerators.isna().any().any()):
                    raise ValueError(f"Incomplete {family} slice: {label}")
                sample = slice_rng.integers(
                    0, len(denominators), size=(args.draws, len(denominators))
                )
                denominator = denominators.to_numpy()[sample].sum(axis=1)
                if (denominator == 0).any():
                    raise ValueError(f"Zero-event draw in {label}")
                values = numerators.to_numpy()[sample].sum(axis=1) / denominator
                for column, method in enumerate(methods):
                    lower, upper = np.quantile(values[:, column], [0.025, 0.975])
                    intervals.append({"method": method, family: label,
                                      "ci_lower": lower, "ci_upper": upper})
            summary["event_recall_at_5"] = (
                summary["n_detected_events"] / summary["n_events"]
            )
            outputs[filename] = summary.merge(
                pd.DataFrame(intervals), on=["method", family], validate="one_to_one"
            )
    # Compare before writing, preserving the original reference artifacts.
    if args.reference_dir:
        for filename, table in outputs.items():
            reference = pd.read_csv(args.reference_dir / filename)
            pd.testing.assert_frame_equal(
                table, reference, check_exact=False, rtol=1e-12, atol=1e-12
            )
        print(f"PASS: {len(outputs)} bootstrap tables match the original results.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in outputs.items():
        table.to_csv(args.output_dir / filename, index=False)
        print(table.to_string(index=False))
    print(f"Channel bootstrap: {args.draws} draws, seed {args.seed}.")


if __name__ == "__main__":
    main()
