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

"""
helper function to pull a slice of telemetry from Postgres and turn it into a tidy pandas
 DataFrame you can feed to detectors
"""
def load_timeseries(db, channel_id: int, start: str, end: str) -> pd.DataFrame:
    q = (db.query(Telemetry)
           .filter(Telemetry.channel_id == channel_id,
                   Telemetry.ts >= start,
                   Telemetry.ts <= end)
            .order_by(Telemetry.ts)) # sort ascening by timestamp so the eries in in time order
    rows = q.all()
    if not rows: # handle no data case
        return pd.DataFrame(columns=["ts", "value"])
    # convert rows to DataFrame
    return pd.DataFrame({"ts": [r.ts for r in rows],
                         "value": [float(r.value) for r in rows]})