import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

from satmon.data import load_channel_splits
from satmon.features import assemble_split_examples
from satmon.models import make_logistic_pipeline, make_random_forest
from satmon.baseline import rolling_3sigma
from satmon.data import load_channel_signal, parse_intervals
from satmon.evaluation import evaluate_channel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-order", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    split_record = json.loads(args.splits.read_text(encoding="utf-8"))

    annotation_hash = hashlib.sha256(
        args.annotations.read_bytes()
    ).hexdigest()

    if annotation_hash != split_record["annotation_sha256"]:
        raise ValueError("Annotation file differs from the saved split source")

    splits = load_channel_splits(args.splits)
    annotations = pd.read_csv(args.annotations)

    split_annotations = {}

    for name, channel_ids in splits.items():
        selected = annotations.loc[
            annotations["chan_id"].isin(channel_ids)
        ].copy()

        if (
            not selected["chan_id"].is_unique
            or set(selected["chan_id"]) != set(channel_ids)
        ):
            raise ValueError(f"{name}: missing or duplicate channel annotations")

        split_annotations[name] = selected

    training_order = json.loads(
        args.training_order.read_text(encoding="utf-8")
    )

    if (
        len(training_order) != len(set(training_order))
        or set(training_order) != set(splits["train"])
    ):
        raise ValueError(
            "Training order must contain every training channel exactly once"
        )

    split_annotations["train"] = (
        split_annotations["train"]
        .set_index("chan_id")
        .loc[training_order]
        .reset_index()
    )

    training_examples = assemble_split_examples(
        split_annotations["train"],
        test_dir=args.telemetry_dir,
        window_size=protocol["window_size"],
        first_prediction_index=protocol["first_prediction_index"],
    )

    print("Training channels:", training_examples["channel_id"].nunique())
    print("Training rows:", len(training_examples))

    feature_columns = protocol["feature_columns"]
    X_train = training_examples[feature_columns]
    y_train = training_examples["label"]

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training labels must contain both classes")

    logistic_settings = protocol["logistic_regression"]
    forest_settings = protocol["random_forest"]

    logistic = make_logistic_pipeline(C=logistic_settings["C"])

    forest = make_random_forest(
        max_depth=forest_settings["max_depth"],
        min_samples_leaf=forest_settings["min_samples_leaf"],
        n_estimators=forest_settings["n_estimators"],
    )

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    print(
        "Logistic iterations:",
        logistic.named_steps["model"].n_iter_,
    )
    print("Forest trees:", len(forest.estimators_))
    print("Frozen models fitted on training channels only.")

    test_annotations = split_annotations["test"]

    test_examples = assemble_split_examples(
        test_annotations,
        test_dir=args.telemetry_dir,
        window_size=protocol["window_size"],
        first_prediction_index=protocol["first_prediction_index"],
    )

    models = {
        "logistic": (logistic, logistic_settings["threshold"]),
        "forest": (forest, forest_settings["threshold"]),
    }

    channel_results = []
    event_results = []
    alert_results = []
    failure_scores = []

    for _, annotation in test_annotations.iterrows():
        channel_id = annotation["chan_id"]

        signal = load_channel_signal(
            args.telemetry_dir,
            channel_id,
            int(annotation["num_values"]),
        )
        events = parse_intervals(
            annotation["anomaly_sequences"],
            len(signal),
        )
        event_types = [
            value.strip()
            for value in annotation["class"].strip("[]").split(",")
            if value.strip()
        ]
        if len(event_types) != len(events) or any(
            value not in {"point", "contextual"} for value in event_types
        ):
            raise ValueError(f"{channel_id}: invalid event-type annotations")

        rows = test_examples.loc[
            test_examples["channel_id"] == channel_id
        ]

        baseline_scores = rolling_3sigma(
            signal,
            window_size=protocol["window_size"],
        )
        if channel_id == "T-12":
            eligible = baseline_scores.loc[
                baseline_scores.index >= protocol["first_prediction_index"]
            ]
            failure_scores.append(pd.DataFrame({
                "method": "baseline", "channel": channel_id,
                "sample_index": eligible.index.to_numpy(),
                "score": eligible["score"].to_numpy(),
                "threshold": protocol["baseline"]["threshold"],
            }))
        baseline_candidates = baseline_scores.loc[
            (baseline_scores.index >= protocol["first_prediction_index"])
            & (baseline_scores["score"] > protocol["baseline"]["threshold"])
        ]

        candidates_by_method = {
            "baseline": (
                baseline_candidates.index.to_numpy(),
                baseline_candidates["score"].to_numpy(),
            )
        }

        for name, (model, threshold) in models.items():
            positive_column = np.flatnonzero(model.classes_ == 1).item()
            probabilities = model.predict_proba(
                rows[feature_columns]
            )[:, positive_column]

            mask = probabilities > threshold
            if channel_id == "A-9" and name == "logistic":
                failure_scores.append(pd.DataFrame({
                    "method": name, "channel": channel_id,
                    "sample_index": rows["sample_index"].to_numpy(),
                    "score": probabilities, "threshold": threshold,
                }))

            candidates_by_method[name] = (
                rows["sample_index"].to_numpy()[mask],
                probabilities[mask],
            )

        for name, (indices, scores) in candidates_by_method.items():
            result = evaluate_channel(
                indices,
                scores,
                events,
                suppression_distance=protocol["suppression_distance"],
            )

            channel_results.append({
                "method": name,
                "channel": channel_id,
                "spacecraft": annotation["spacecraft"],
                "n_samples": len(signal),
                "n_scored_samples": len(rows),
                "n_events": result["n_events"],
                "n_detected_events": result["n_detected_events"],
                "n_missed_events": result["n_missed_events"],
                "n_selected_alerts": len(result["selected_alerts"]),
                "n_false_alerts": result["n_false_alerts"],
            })
            for event_id, ((start, end), event_type, detected) in enumerate(
                zip(events, event_types, result["event_detected"])
            ):
                event_results.append({
                    "method": name, "channel": channel_id,
                    "spacecraft": annotation["spacecraft"],
                    "event_id": event_id, "start": start, "end": end,
                    "event_type": event_type, "detected": detected,
                })
            for sample_index in result["selected_alerts"]:
                alert_results.append({
                    "method": name, "channel": channel_id,
                    "sample_index": sample_index,
                })

    channel_results = pd.DataFrame(channel_results)

    summary = channel_results.groupby("method")[
        [
            "n_events",
            "n_detected_events",
            "n_missed_events",
            "n_selected_alerts",
            "n_false_alerts",
        ]
    ].sum()

    summary["event_recall_at_5"] = (
        summary["n_detected_events"] / summary["n_events"]
    )
    summary["secondary_cost"] = (
        protocol["missed_anomaly_relative_cost"]
        * summary["n_missed_events"]
        + summary["n_false_alerts"]
    )

    print(summary.to_string())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    channel_results.to_csv(
        args.output_dir / "channels.csv",
        index=False,
    )
    summary.to_csv(args.output_dir / "summary.csv")
    pd.DataFrame(event_results).to_csv(args.output_dir / "events.csv", index=False)
    pd.DataFrame(
        alert_results, columns=["method", "channel", "sample_index"]
    ).to_csv(args.output_dir / "alerts.csv", index=False)
    pd.concat(failure_scores, ignore_index=True).to_csv(
        args.output_dir / "failure_scores.csv", index=False
    )


if __name__ == "__main__":
    main()
