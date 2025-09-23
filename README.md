**SatMon — Satellite Telemetry & Anomaly Detection**
Backend-first project that ingests satellite-style telemetry into Postgres, exposes it via a FastAPI API, and flags anomalies with statistical + ML detectors (rolling z-score, Isolation Forest).

**Status**
actively building

What it does
- Ingest → Store → Query: normalize time-series into Postgres and serve via REST.
- Detect anomalies: rolling z-score and Isolation Forest write windows to the DB.
- Developer-friendly: small, composable services + scripts; Dockerized Postgres.

**Stack**
FastAPI • SQLAlchemy • PostgreSQL • scikit-learn • Pandas • Uvicorn • Docker

**Repo layout (current)**
satmon/
  .env                     # DATABASE_URL (see Quickstart)
  requirements.txt
  data/                    # demo CSV goes here
  services/
    api/
      app.py               # /channels, /timeseries, /anomalies
      db.py                # engine/session, init_db(), .env loading
      models.py            # Channel, Telemetry, Anomaly tables
  scripts/
    make_demo_csv.py       # synth telemetry with an injected spike
    load_csv_channel.py    # CSV -> Channel + Telemetry
    detect_zscore.py       # rolling z-score detector
    detect_isoforest.py    # Isolation Forest detector

**Quickstart (Windows / PowerShell)**
Prereqs: Python 3.11+, Docker Desktop

# 0) Clone and enter
git clone <your repo url> satmon
cd satmon

# 1) Virtual env + deps
python -m venv .venv
. .\.venv\Scripts\Activate
pip install -r requirements.txt

# 2) Start Postgres in Docker (maps to localhost:5432)
docker run --name satmon-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=satmon `
  -p 5432:5432 -d postgres:16

# 3) .env at repo root
ni .env -Value 'DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/satmon'

# 4) Run the API (keeps running)
uvicorn services.api.app:app --reload

Open Swagger: http://127.0.0.1:8000/docs
In a second terminal (activate venv again):

# 5) Create a demo CSV (10h @ 1-min cadence with an injected spike)
python scripts\make_demo_csv.py

# 6) Load demo channel + telemetry into Postgres
python -m scripts.load_csv_channel --csv data\demo_temp.csv --source DEMO --key PWR.BATT_TEMP --units C

# 7) Run detectors
python -m scripts.detect_zscore   --channel-id 1 --start 2025-01-01T00:00:00Z --end 2025-01-01T10:00:00Z --win 60 --z 3.0
python -m scripts.detect_isoforest --channel-id 1 --start 2025-01-01T00:00:00Z --end 2025-01-01T10:00:00Z --roll-win 60 --contamination 0.01 --n-est 200

**Try the endpoints**
- Channels
  http://127.0.0.1:8000/channels
- Time series
  http://127.0.0.1:8000/timeseries?channel_id=1&start=2025-01-01T00:00:00Z&end=2025-01-01T10:00:00Z
- Anomalies (all)
  http://127.0.0.1:8000/anomalies?channel_id=1&start=2025-01-01T00:00:00Z&end=2025-01-01T10:00:00Z
- Anomalies by method
  ...&method=zscore or ...&method=isoforest

**API (current)**
- GET /channels?source=SMAP|DEMO... → list channel metadata
- GET /timeseries?channel_id&start&end → ordered {ts,value} points
- GET /anomalies?channel_id&start&end[&method] → anomaly windows with scores

**How it Works**
Scripts (or future workers) ingest data and write to channel + telemetry. 
Detectors compute outliers over a time window and persist windows into anomaly_detection 
with a consistent score and method. The FastAPI service is a clean read API for dashboards,
notebooks, or alerts.
