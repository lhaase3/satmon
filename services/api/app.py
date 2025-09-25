from fastapi import FastAPI, Depends, Query, HTTPException, WebSocket
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
from .db import init_db, get_db
from .models import Channel, Telemetry, Anomaly
from .websockets import websocket_endpoint
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

# serve static files from the local web/ folder
STATIC_DIR = Path(__file__).resolve().parents[2] / "web"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# serve the main dashboard at root
@app.get("/")
async def read_root():
    from fastapi.responses import FileResponse
    return FileResponse(STATIC_DIR / "index.html")

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

@app.websocket("/ws/telemetry/{channel_id}")
async def websocket_telemetry(websocket: WebSocket, channel_id: int):
    """WebSocket endpoint for real-time telemetry streaming."""
    await websocket_endpoint(websocket, channel_id)

@app.post("/run-detection")
async def run_detection(
    channel_id: int,
    method: str = "zscore",
    start: datetime = None,
    end: datetime = None,
    db: Session = Depends(get_db)
):
    """Run anomaly detection on demand for demo purposes."""
    import subprocess
    import sys
    from pathlib import Path
    
    # Set default time range if not provided
    if start is None:
        start = datetime(2018, 1, 1)
    if end is None:
        end = datetime(2018, 1, 5)
    
    # Get the script path
    script_name = f"detect_{method}.py"
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    
    if not script_path.exists():
        raise HTTPException(status_code=400, detail=f"Unknown detection method: {method}")
    
    try:
        # Run the detection script
        result = subprocess.run([
            sys.executable, str(script_path),
            "--channel-id", str(channel_id),
            "--start", start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "--end", end.strftime("%Y-%m-%dT%H:%M:%SZ")
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return {"status": "success", "message": result.stdout.strip()}
        else:
            return {"status": "error", "message": result.stderr.strip()}
            
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Detection timed out"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
