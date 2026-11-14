"""Plot predefined post-test diagnostic examples from regenerated frozen scores."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scores = pd.read_csv(args.results_dir / "failure_scores.csv")
    events = pd.read_csv(args.results_dir / "events.csv")
    diagnostics = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    for ax, channel, method, title, ylabel in [
        (axes[0], "T-12", "baseline",
         "T-12: baseline scores remain below threshold during the event",
         "Absolute rolling z-score"),
        (axes[1], "A-9", "logistic",
         "A-9: Logistic Regression probabilities remain below threshold in event",
         "Predicted overlap probability"),
    ]:
        rows = scores.loc[
            (scores["channel"] == channel) & (scores["method"] == method)
        ].sort_values("sample_index")
        event = events.loc[
            (events["channel"] == channel) & (events["method"] == method)
        ]
        if len(event) != 1 or rows.empty:
            raise ValueError(f"Expected scores and one event for {channel}")
        if rows["sample_index"].duplicated().any():
            raise ValueError(f"Duplicate score indices for {channel}")
        if not np.isfinite(rows["score"]).all() or rows["threshold"].nunique() != 1:
            raise ValueError(f"Invalid scores or threshold for {channel}")
        start, end = int(event.iloc[0]["start"]), int(event.iloc[0]["end"])
        threshold = float(rows.iloc[0]["threshold"])
        inside = rows.loc[rows["sample_index"].between(start, end)]
        if inside.empty or inside["score"].gt(threshold).any():
            raise ValueError(f"Expected candidate-generation failure for {channel}")
        diagnostics.append({
            "channel": channel, "method": method, "start": start, "end": end,
            "max_event_score": inside["score"].max(), "threshold": threshold,
            "event_candidates": int(inside["score"].gt(threshold).sum()),
        })
        view = rows.loc[rows["sample_index"].between(500, 900)] if channel == "T-12" else rows
        ax.plot(view["sample_index"], view["score"], linewidth=1)
        ax.axhline(threshold, color="black", linestyle="--", label="Threshold")
        ax.axvspan(start, end, color="red", alpha=0.2, label="Annotated event")
        ax.set(title=title, xlabel="Original sample index", ylabel=ylabel)
        ax.legend()

    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "held_out_failure_examples.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / "failure_plot_diagnostics.csv", index=False
    )
    print(f"Saved {output}")
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()
