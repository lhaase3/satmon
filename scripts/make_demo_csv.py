# scripts/make_demo_csv.py
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
N = 600  # 10 hours @ 1-min cadence
ts = [start + timedelta(minutes=i) for i in range(N)]

# Smooth signal + noise + one spike to trigger anomalies
base = 20 + 2*np.sin(np.linspace(0, 12*np.pi, N))
noise = np.random.normal(0, 0.2, size=N)
values = base + noise
values[350:360] += 6.0  # anomaly burst

df = pd.DataFrame({"ts": ts, "value": values})
Path("data").mkdir(exist_ok=True, parents=True)
out = Path("data/demo_temp.csv")
df.to_csv(out, index=False)
print("Wrote", out)
print("start:", df['ts'].iloc[0].isoformat(), "end:", df['ts'].iloc[-1].isoformat())
