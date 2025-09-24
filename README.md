# SatMon - Satellite Telemetry Anomaly Detection

A production-ready backend system for ingesting satellite telemetry data and detecting anomalies using machine learning. Built for aerospace applications with real NASA datasets.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Features

### Core Capabilities
- **Real-time Telemetry Ingestion** - Structured time-series data storage
- **Multi-Algorithm Anomaly Detection** - Z-score, Isolation Forest, and extensible ML pipeline
- **REST API** - FastAPI with automatic OpenAPI documentation
- **Time-Series Database** - PostgreSQL with optimized schema for telemetry data
- **Interactive Visualization** - Web-based dashboard for exploring anomalies
- **Ground Truth Evaluation** - Performance metrics against labeled datasets

### Anomaly Detection Methods
- **Rolling Z-Score** - Statistical baseline with hysteresis
- **Isolation Forest** - Multivariate feature detection with:
  - Value, delta, rolling statistics features
  - Smart window grouping and gap tolerance
  - Configurable contamination thresholds

### Data Sources
- **NASA JPL Telemanom** - SMAP/MSL labeled anomaly dataset
- **CSV Import** - Generic telemetry data loading
- **Extensible Architecture** - Ready for SatNOGS, CelesTrak, and live streams

## 📋 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/satmon.git
cd satmon

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup
```bash
# Start PostgreSQL with Docker
docker-compose up postgres -d

# Set database URL
echo "DATABASE_URL=postgresql://satmon:satmon123@localhost:5432/satmon" > .env
```

### 3. Load Demo Data
```bash
# Load NASA Telemanom dataset (example channel)
python scripts/load_telemanom.py --channel-id P-1

# Or load your own CSV data
python scripts/load_csv_channel.py \
    --csv data/demo_temp.csv \
    --key "TEMP.BATTERY" \
    --source "DEMO" \
    --units "°C"
```

### 4. Run API Server
```bash
uvicorn services.api.app:app --reload
```

### 5. View Dashboard
Open http://localhost:8000/viewer/viewer.html

## 🔍 Usage Examples

### Run Anomaly Detection
```bash
# Rolling Z-Score detection
python scripts/detect_zscore.py \
    --channel-id 1 \
    --start "2018-01-01T00:00:00Z" \
    --end "2018-01-02T00:00:00Z" \
    --win 100 --z 3.0

# Isolation Forest detection
python scripts/detect_isoforest.py \
    --channel-id 1 \
    --start "2018-01-01T00:00:00Z" \
    --end "2018-01-02T00:00:00Z" \
    --contamination 0.01
```

### Evaluate Performance
```bash
# Compare methods against ground truth
python scripts/evaluate_detection.py \
    --channel-id 1 \
    --methods zscore isoforest \
    --plot
```

### API Endpoints
```bash
# List available channels
curl "http://localhost:8000/channels"

# Get time series data
curl "http://localhost:8000/timeseries?channel_id=1&start=2018-01-01T00:00:00Z&end=2018-01-02T00:00:00Z"

# Get detected anomalies
curl "http://localhost:8000/anomalies?channel_id=1&start=2018-01-01T00:00:00Z&end=2018-01-02T00:00:00Z&method=isoforest"
```

## 🏗️ Architecture

```
satmon/
├── services/
│   └── api/                 # FastAPI application
│       ├── app.py          # REST endpoints
│       ├── models.py       # SQLAlchemy models
│       └── db.py           # Database configuration
├── scripts/                 # Data processing & ML
│   ├── load_telemanom.py   # NASA dataset loader
│   ├── detect_zscore.py    # Statistical anomaly detection
│   ├── detect_isoforest.py # ML-based detection
│   └── evaluate_detection.py # Performance evaluation
├── web/                    # Frontend dashboard
│   └── viewer.html         # Interactive time series viewer
└── data/                   # Local data storage
```

### Database Schema
- **`channel`** - Telemetry channel metadata (source, units, description)
- **`telemetry`** - Time-series data points (timestamp, value, quality)
- **`anomaly_detection`** - Detected anomalies (windows, scores, methods, parameters)

## 🚀 Production Deployment

### Docker Compose (Full Stack)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api
```

### Environment Variables
```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname
API_HOST=0.0.0.0
API_PORT=8000
```

## 📊 Performance & Evaluation

The system includes comprehensive evaluation against the NASA JPL Telemanom dataset:

- **Precision/Recall/F1-Score** metrics
- **Window-based evaluation** with configurable overlap tolerance
- **Method comparison** visualizations
- **Ground truth labeling** from academic research

### Benchmark Results (Example)
| Method | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Z-Score | 0.847 | 0.732 | 0.785 |
| Isolation Forest | 0.891 | 0.798 | 0.842 |

## 🔬 Research Foundation

Built on proven aerospace anomaly detection research:
- **JPL Telemanom Paper** - Hundman et al. (2018) "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
- **NASA SMAP/MSL Mission Data** - Real spacecraft telemetry with expert-labeled anomalies
- **Industry-Standard Methods** - Statistical process control and unsupervised ML

## 🚧 Roadmap

### Phase 2 - Live Data Integration
- [ ] SatNOGS community satellite frames
- [ ] CelesTrak TLE orbit state tracking
- [ ] Real-time streaming with Apache Kafka

### Phase 3 - Advanced ML
- [ ] LSTM autoencoder for multivariate detection
- [ ] Transformer models for sequence anomalies
- [ ] Multi-mission model transfer learning

### Phase 4 - Operations
- [ ] Alerting system (email, webhook, Slack)
- [ ] Model drift detection & retraining
- [ ] Grafana/Prometheus monitoring integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA JPL** for the Telemanom dataset
- **SatNOGS Community** for open satellite data
- **FastAPI & SQLAlchemy** teams for excellent tools

## 📞 Contact

**Project Link:** https://github.com/YOUR_USERNAME/satmon

---

*Built for the space industry. Ready for production. Open for collaboration.*