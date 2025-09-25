#!/usr/bin/env python3
"""
Simple demo server for SatMon project
Serves the enhanced dashboard with mock data for demonstration
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

app = FastAPI(title="SatMon Demo", description="Satellite Telemetry Anomaly Detection Demo")

# Mount static files
web_dir = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    html_file = web_dir / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding='utf-8'), status_code=200)

@app.get("/channels")
async def get_channels():
    """Mock channels endpoint"""
    return [
        {
            "id": 1,
            "name": "demo_temp_sensor",
            "channel_key": "TEMP_01",
            "description": "Primary Temperature Sensor",
            "source": "Spacecraft Bus"
        },
        {
            "id": 2,
            "name": "demo_power_voltage", 
            "channel_key": "PWR_VOLT_01",
            "description": "Solar Panel Voltage",
            "source": "Power Subsystem"
        },
        {
            "id": 3,
            "name": "demo_attitude_x",
            "channel_key": "ATT_X_01", 
            "description": "Attitude X-Axis",
            "source": "ADCS"
        }
    ]

@app.get("/timeseries")
async def get_timeseries(channel: str = "demo_temp_sensor", limit: int = 500):
    """Mock timeseries data endpoint"""
    
    # Generate realistic satellite telemetry data
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(limit, 0, -1)]
    
    if "temp" in channel.lower():
        # Temperature data: -20°C to +60°C with slow variations
        base_temp = 20 + 15 * np.sin(np.linspace(0, 4*np.pi, limit))
        noise = np.random.normal(0, 2, limit)
        values = base_temp + noise
        
        # Add some anomalies
        anomaly_indices = [100, 101, 102, 350, 351, 450]
        for idx in anomaly_indices:
            if idx < len(values):
                values[idx] += np.random.uniform(15, 25)  # Temperature spikes
                
    elif "power" in channel.lower() or "volt" in channel.lower():
        # Voltage data: 28V ± 2V with orbital variations
        base_voltage = 28 + 1.5 * np.sin(np.linspace(0, 6*np.pi, limit))
        noise = np.random.normal(0, 0.1, limit)
        values = base_voltage + noise
        
        # Power system anomalies
        anomaly_indices = [200, 201, 380, 381, 382]
        for idx in anomaly_indices:
            if idx < len(values):
                values[idx] -= np.random.uniform(3, 8)  # Voltage drops
                
    elif "attitude" in channel.lower():
        # Attitude data: ±180 degrees with slow drift
        base_attitude = 45 * np.sin(np.linspace(0, 2*np.pi, limit))
        noise = np.random.normal(0, 1, limit)
        values = base_attitude + noise
        
        # Attitude anomalies
        anomaly_indices = [150, 151, 152, 153, 400, 401]
        for idx in anomaly_indices:
            if idx < len(values):
                values[idx] += np.random.uniform(30, 60)  # Large attitude deviations
    else:
        # Generic telemetry
        values = 50 + 10 * np.sin(np.linspace(0, 8*np.pi, limit)) + np.random.normal(0, 2, limit)
    
    # Convert to the expected format
    data = []
    for i, (ts, val) in enumerate(zip(timestamps, values)):
        data.append({
            "timestamp": ts.isoformat(),
            "value": float(val),
            "channel": channel
        })
    
    return data

@app.get("/anomalies")
async def get_anomalies(channel: str = "demo_temp_sensor"):
    """Mock anomalies endpoint"""
    
    now = datetime.now()
    
    if "temp" in channel.lower():
        anomalies = [
            {
                "id": 1,
                "timestamp": (now - timedelta(minutes=400)).isoformat(),
                "value": 45.2,
                "channel": channel,
                "method": "zscore",
                "score": 3.8,
                "severity": "high"
            },
            {
                "id": 2,
                "timestamp": (now - timedelta(minutes=399)).isoformat(),
                "value": 47.1,
                "channel": channel,
                "method": "zscore", 
                "score": 4.2,
                "severity": "high"
            },
            {
                "id": 3,
                "timestamp": (now - timedelta(minutes=150)).isoformat(),
                "value": 42.8,
                "channel": channel,
                "method": "isolation_forest",
                "score": -0.85,
                "severity": "medium"
            }
        ]
    elif "power" in channel.lower() or "volt" in channel.lower():
        anomalies = [
            {
                "id": 4,
                "timestamp": (now - timedelta(minutes=300)).isoformat(),
                "value": 22.1,
                "channel": channel,
                "method": "zscore",
                "score": -3.2,
                "severity": "high"
            },
            {
                "id": 5,
                "timestamp": (now - timedelta(minutes=120)).isoformat(),
                "value": 21.8,
                "channel": channel,
                "method": "isolation_forest",
                "score": -0.78,
                "severity": "high"
            }
        ]
    elif "attitude" in channel.lower():
        anomalies = [
            {
                "id": 6,
                "timestamp": (now - timedelta(minutes=350)).isoformat(),
                "value": 95.6,
                "channel": channel,
                "method": "zscore",
                "score": 4.1,
                "severity": "critical"
            },
            {
                "id": 7,
                "timestamp": (now - timedelta(minutes=100)).isoformat(),
                "value": -87.3,
                "channel": channel,
                "method": "isolation_forest",
                "score": -0.91,
                "severity": "high"
            }
        ]
    else:
        anomalies = []
    
    return anomalies

@app.post("/run-detection")
async def run_detection(request: dict):
    """Mock anomaly detection endpoint"""
    channel = request.get("channel", "demo_temp_sensor")
    algorithm = request.get("algorithm", "zscore")
    
    # Simulate processing time (LSTM takes longer)
    import time
    if algorithm == "lstm_autoencoder":
        time.sleep(2)  # Deep learning takes more time
    else:
        time.sleep(1)  # Traditional algorithms
    
    # Return mock results based on channel and algorithm
    if "temp" in channel.lower():
        if algorithm == "zscore":
            anomalies_detected = 3
        elif algorithm == "isolation_forest":
            anomalies_detected = 2
        else:  # lstm_autoencoder
            anomalies_detected = 4  # Deep learning often finds more subtle patterns
    elif "power" in channel.lower():
        if algorithm == "zscore":
            anomalies_detected = 2
        elif algorithm == "isolation_forest":
            anomalies_detected = 3
        else:  # lstm_autoencoder
            anomalies_detected = 5
    elif "attitude" in channel.lower():
        if algorithm == "zscore":
            anomalies_detected = 4
        elif algorithm == "isolation_forest":
            anomalies_detected = 2
        else:  # lstm_autoencoder
            anomalies_detected = 3
    else:
        anomalies_detected = 1
    
    return {
        "status": "success",
        "channel": channel,
        "algorithm": algorithm,
        "anomalies_detected": anomalies_detected,
        "processing_time": 1.0,
        "message": f"Successfully processed {channel} using {algorithm.upper()} algorithm"
    }

@app.get("/algorithm-comparison")
async def algorithm_comparison(channel: str = "demo_temp_sensor"):
    """Compare performance of all three algorithms"""
    
    # Simulate running all algorithms
    import time
    time.sleep(2)  # Simulate processing time
    
    if "temp" in channel.lower():
        results = {
            "zscore": {
                "precision": 0.364,
                "recall": 1.0,
                "f1_score": 0.615,
                "anomalies_detected": 3,
                "processing_time": 0.12,
                "confidence": "high"
            },
            "isolation_forest": {
                "precision": 0.095,
                "recall": 0.50,
                "f1_score": 0.160,
                "anomalies_detected": 2,
                "processing_time": 0.45,
                "confidence": "medium"
            },
            "lstm_autoencoder": {
                "precision": 0.287,
                "recall": 0.833,
                "f1_score": 0.427,
                "anomalies_detected": 4,
                "processing_time": 2.3,
                "confidence": "high"
            }
        }
    elif "power" in channel.lower():
        results = {
            "zscore": {
                "precision": 0.425,
                "recall": 0.85,
                "f1_score": 0.565,
                "anomalies_detected": 2,
                "processing_time": 0.11,
                "confidence": "high"
            },
            "isolation_forest": {
                "precision": 0.123,
                "recall": 0.67,
                "f1_score": 0.208,
                "anomalies_detected": 3,
                "processing_time": 0.38,
                "confidence": "medium"
            },
            "lstm_autoencoder": {
                "precision": 0.312,
                "recall": 0.91,
                "f1_score": 0.464,
                "anomalies_detected": 5,
                "processing_time": 2.1,
                "confidence": "high"
            }
        }
    else:  # attitude
        results = {
            "zscore": {
                "precision": 0.298,
                "recall": 0.95,
                "f1_score": 0.453,
                "anomalies_detected": 4,
                "processing_time": 0.13,
                "confidence": "high"
            },
            "isolation_forest": {
                "precision": 0.087,
                "recall": 0.40,
                "f1_score": 0.143,
                "anomalies_detected": 2,
                "processing_time": 0.42,
                "confidence": "low"
            },
            "lstm_autoencoder": {
                "precision": 0.256,
                "recall": 0.75,
                "f1_score": 0.383,
                "anomalies_detected": 3,
                "processing_time": 2.4,
                "confidence": "medium"
            }
        }
    
    return {
        "channel": channel,
        "comparison_results": results,
        "recommendation": "zscore",  # Best overall performer
        "summary": {
            "best_precision": "zscore",
            "best_recall": "lstm_autoencoder", 
            "best_f1": "zscore",
            "fastest": "zscore"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SatMon Demo"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting SatMon Demo Server...")
    print("📊 Dashboard: http://localhost:8000")
    print("🔧 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)