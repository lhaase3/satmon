from fastapi import FastAPI, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
from .db import init_db, get_db
from .models import Channel, Telemetry, Anomaly
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="SatMon API")

# allow read-only GETs from any origin (fine for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# serve /viewer/* from the local web/ folder
STATIC_DIR = Path(__file__).resolve().parents[2] / "web"
app.mount("/viewer", StaticFiles(directory=STATIC_DIR, html=True), name="viewer")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/channels")
def list_channels(source: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(Channel)
    if source:
        q = q.filter(Channel.source == source)
    return [ {"id":c.id,"source":c.source,"channel_key":c.channel_key,"units":c.units} for c in q.limit(500) ]

@app.get("/timeseries")
def get_timeseries(
    channel_id: int,
    start: datetime,
    end: datetime,
    db: Session = Depends(get_db),
):
    rows = (db.query(Telemetry)
              .filter(Telemetry.channel_id==channel_id,
                      Telemetry.ts>=start,
                      Telemetry.ts<=end)
              .order_by(Telemetry.ts)
              .all())
    return {"channel_id": channel_id,
            "points": [{"ts": r.ts.isoformat(), "value": r.value} for r in rows]}

from typing import Optional
# ...

@app.get("/anomalies")
def get_anomalies(
    channel_id: int,
    start: datetime,
    end: datetime,
    method: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = (db.query(Anomaly)
           .filter(Anomaly.channel_id == channel_id,
                   Anomaly.window_start <= end,
                   Anomaly.window_end >= start))
    if method:
        q = q.filter(Anomaly.method == method)
    rows = q.order_by(Anomaly.window_start).all()
    return [{"id": r.id,
             "window_start": r.window_start,
             "window_end": r.window_end,
             "score": r.score,
             "label": r.label,
             "method": r.method} for r in rows]

@app.get("/healthz")
def healthz(db: Session = Depends(get_db)):
    # simple ping to ensure the DB session can be created
    return {"status": "ok"}
