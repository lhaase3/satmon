import argparse, pandas as pd
from datetime import timezone
from sqlalchemy.orm import Session
from services.api.db import SessionLocal, init_db
from services.api.models import Channel, Telemetry
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def upsert_channel(db: Session, source: str, channel_key: str, units: str = None):
    ch = db.query(Channel).filter_by(source=source, channel_key=channel_key).first()
    if not ch:
        ch = Channel(source=source, channel_key=channel_key, units=units)
        db.add(ch); db.commit(); db.refresh(ch)
    return ch.id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--source", default="DEMO")
    parser.add_argument("--key", required=True, help="e.g., PWR.BATT_TEMP")
    parser.add_argument("--units", default=None)
    args = parser.parse_args()

    init_db()
    df = pd.read_csv(args.csv)
    if "ts" not in df or "value" not in df:
        raise SystemExit("CSV must have columns: ts,value")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    with SessionLocal() as db:
        cid = upsert_channel(db, args.source, args.key, args.units)
        # bulk insert
        rows = []
        for r in df.itertuples(index=False):
            rows.append(Telemetry(channel_id=cid, ts=r.ts.to_pydatetime(), value=float(r.value)))
        db.bulk_save_objects(rows)
        db.commit()
        print(f"Loaded {len(rows)} rows into channel_id={cid}")

if __name__ == "__main__":
    main()
