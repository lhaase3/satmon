# scripts/detect_isoforest.py
"""
Isolation Forest detector for a single channel/time window.

What it does:
1) Query Telemetry for [channel_id, start, end]
2) Build features per timestamp:
   - value
   - delta (first difference)
   - rolling mean (window W)
   - rolling std  (window W)
3) Fit IsolationForest (unsupervised) on those features
4) Convert point-wise outliers to anomaly WINDOWS (merge small gaps)
5) Write windows into anomaly_detection with method="isoforest"
"""

import argparse
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from services.api.db import SessionLocal, init_db
from services.api.models import Telemetry, Anomaly

def load_timeseries(db, channel_id: int, start: str, end: str) -> pd.DataFrame:
    q = (db.query(Telemetry)
           .filter(Telemetry.channel_id == channel_id,
                   Telemetry.ts >= start,
                   Telemetry.ts <= end)
           .order_by(Telemetry.ts))
    rows = q.all()
    if not rows:
        return pd.DataFrame(columns=["ts","value"])
    return pd.DataFrame({"ts": [r.ts for r in rows],
                         "value": [float(r.value) for r in rows]})

def make_features(df: pd.DataFrame, roll_win: int) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a small, informative feature set.
    - value
    - delta      (first difference)
    - roll_mean  (captures trend)
    - roll_std   (captures local variability)
    Fill early NaNs with backfill/zero to keep dimensions aligned.
    """
    df = df.sort_values("ts").copy()
    df["delta"] = df["value"].diff()
    df["roll_mean"] = df["value"].rolling(roll_win, min_periods=max(3, roll_win//3)).mean()
    df["roll_std"]  = df["value"].rolling(roll_win, min_periods=max(3, roll_win//3)).std()

    # Minimal imputation so we can fit the model
    feat_df = df[["value","delta","roll_mean","roll_std"]].copy()
    feat_df = feat_df.bfill().fillna(0.0)

    X = feat_df.to_numpy(dtype=float)
    return df, X

def group_anomaly_windows(
    ts: pd.Series,
    is_outlier: np.ndarray,
    severity: np.ndarray,
    gap_tol: pd.Timedelta,
) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """
    Convert point anomalies to windows.
    - Merge contiguous outliers.
    - Allow tiny gaps <= gap_tol (e.g., 1 minute) to keep a single window.
    Returns a list of (window_start, window_end, max_severity).
    """
    windows = []
    in_run = False
    start = end = None
    maxsev = 0.0

    for i in range(len(ts)):
        t = ts.iloc[i]
        out = bool(is_outlier[i])
        sev = float(severity[i])

        if out:
            if not in_run:
                # start a new window
                in_run = True
                start = end = t
                maxsev = sev
            else:
                # continue current window
                end = t
                if sev > maxsev:
                    maxsev = sev
        else:
            if in_run:
                # are we still within the gap tolerance?
                # if time since last inlier is small, keep the window open
                # (we'll only close when a gap gets too large)
                # To implement this, look ahead: if next timestamp exists and the time
                # since 'end' is > gap_tol, then close; else keep going.
                # Simpler: close now *unless* (i < len(ts)-1 and ts[i+1]-end <= gap_tol)
                if i < len(ts) - 1 and (ts.iloc[i+1] - end) <= gap_tol:
                    # small gap, do nothing (keep window open)
                    pass
                else:
                    windows.append((start, end, maxsev))
                    in_run = False
                    start = end = None
                    maxsev = 0.0

    if in_run:
        windows.append((start, end, maxsev))

    return windows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel-id", type=int, required=True)
    ap.add_argument("--start", required=True, help="ISO timestamp (UTC), e.g. 2025-01-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="ISO timestamp (UTC)")
    ap.add_argument("--roll-win", type=int, default=60, help="rolling window size (points)")
    ap.add_argument("--contamination", type=float, default=0.01, help="expected outlier fraction")
    ap.add_argument("--n-est", type=int, default=200, help="IsolationForest n_estimators")
    ap.add_argument("--max-samples", type=str, default="auto", help="'auto' or int")
    ap.add_argument("--gap-min", type=int, default=1, help="merge gaps up to N minutes")
    args = ap.parse_args()

    init_db()
    with SessionLocal() as db:
        df = load_timeseries(db, args.channel_id, args.start, args.end)
        if df.empty:
            print("No data in range.")
            return

        df_feat, X = make_features(df, args.roll_win)

        # Fit IF on the window’s data (unsupervised)
        clf = IsolationForest(
            n_estimators=args.n_est,
            contamination=args.contamination,
            max_samples=(None if args.max_samples == "None" else args.max_samples),
            random_state=42,
        )
        clf.fit(X)

        # For IsolationForest:
        # - predict() returns -1 for outliers, +1 for inliers
        # - decision_function(): higher => more normal, lower => more anomalous
        pred = clf.predict(X)  # -1 / +1
        decf = clf.decision_function(X)
        severity = -decf  # larger => more anomalous (nice for a 'score' field)

        # Group point outliers into windows (merge ≤ gap-min)
        gap_tol = pd.to_timedelta(args.gap_min, unit="min")
        windows = group_anomaly_windows(df_feat["ts"], pred == -1, severity, gap_tol)

        # Write to DB
        objs = []
        for (ws, we, maxsev) in windows:
            objs.append(Anomaly(
                channel_id=args.channel_id,
                window_start=ws.to_pydatetime(),
                window_end=we.to_pydatetime(),
                score=float(maxsev),
                label=True,
                method="isoforest",
                params={
                    "roll_win": args.roll_win,
                    "contamination": args.contamination,
                    "n_estimators": args.n_est,
                    "max_samples": args.max_samples,
                    "features": ["value","delta","roll_mean","roll_std"],
                },
            ))
        if objs:
            db.bulk_save_objects(objs)
            db.commit()
        print(f"Inserted {len(objs)} anomalies.")

if __name__ == "__main__":
    main()
