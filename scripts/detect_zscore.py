import argparse, pandas as pd, numpy as np
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from pathlib import Path
import sys

# Add the project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.api.db import SessionLocal, init_db
from services.api.models import Telemetry, Anomaly

def detect_anomalies(df: pd.DataFrame, win=100, z=3.0, min_pts=20):
    df = df.sort_values("ts").copy()
    df["mu"] = df["value"].rolling(win, min_periods=min_pts).mean()
    df["sigma"] = df["value"].rolling(win, min_periods=min_pts).std()
    df["z"] = (df["value"] - df["mu"]) / df["sigma"]
    mask = df["sigma"].notna() & (df["z"].abs() >= z)

    # group contiguous anomaly points
    anomalies = []
    if mask.any():
        in_run = False
        start = None
        for ts, is_out, score in zip(df["ts"], mask, df["z"].abs().fillna(0)):
            if is_out and not in_run:
                in_run = True; start = ts; maxz = score
            elif is_out and in_run:
                maxz = max(maxz, score)
            elif not is_out and in_run:
                anomalies.append((start, prev_ts, float(maxz)))
                in_run = False
            prev_ts = ts
        if in_run:
            anomalies.append((start, df["ts"].iloc[-1], float(maxz)))
    return anomalies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-id", type=int, required=True)
    parser.add_argument("--start", required=True)  # ISO
    parser.add_argument("--end", required=True)
    parser.add_argument("--win", type=int, default=100)
    parser.add_argument("--z", type=float, default=3.0)
    args = parser.parse_args()

    init_db()
    with SessionLocal() as db:
        q = (db.query(Telemetry)
               .filter(Telemetry.channel_id==args.channel_id,
                       Telemetry.ts>=args.start,
                       Telemetry.ts<=args.end)
               .order_by(Telemetry.ts))
        rows = q.all()
        if not rows:
            print("No data in range."); return

        df = pd.DataFrame({"ts":[r.ts for r in rows], "value":[r.value for r in rows]})
        anomalies = detect_anomalies(df, win=args.win, z=args.z)
        objs = []
        for s,e,score in anomalies:
            objs.append(Anomaly(channel_id=args.channel_id, window_start=s, window_end=e,
                                score=score, label=True, method="zscore", params={"win":args.win,"z":args.z}))
        if objs:
            db.bulk_save_objects(objs); db.commit()
        print(f"Inserted {len(objs)} anomalies.")

if __name__ == "__main__":
    main()
