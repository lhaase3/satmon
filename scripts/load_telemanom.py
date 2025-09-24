#!/usr/bin/env python3
"""
Load NASA JPL Telemanom dataset for SMAP/MSL telemetry.
https://github.com/khundman/telemanom

Note: The actual telemanom data is now hosted on Kaggle. This script creates
synthetic telemetry data that mimics the structure and patterns of the original
NASA dataset for demonstration purposes.

For the real data, you would need to:
1. pip install kaggle
2. Setup Kaggle API credentials
3. kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta

# Add the project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.api.db import SessionLocal, init_db
from services.api.models import Channel, Telemetry, Anomaly

def generate_synthetic_telemetry(channel_id: str, n_points: int = 8760) -> tuple[np.ndarray, list]:
    """
    Generate synthetic telemetry data that mimics real satellite patterns.
    Creates realistic time series with injected anomalies.
    """
    np.random.seed(42)  # Reproducible results
    
    # Base patterns by channel type
    base_patterns = {
        'P': {'mean': 0.0, 'std': 0.1, 'trend': 0.0001, 'seasonal_amp': 0.05},  # Power
        'S': {'mean': 0.2, 'std': 0.08, 'trend': -0.0002, 'seasonal_amp': 0.15}, # Solar
        'E': {'mean': -0.1, 'std': 0.12, 'trend': 0.0, 'seasonal_amp': 0.08},   # Electrical  
        'T': {'mean': 0.1, 'std': 0.06, 'trend': 0.0001, 'seasonal_amp': 0.12}, # Thermal
        'C': {'mean': 0.0, 'std': 0.09, 'trend': 0.0, 'seasonal_amp': 0.06},    # Command
        'A': {'mean': 0.15, 'std': 0.07, 'trend': -0.0001, 'seasonal_amp': 0.09} # Attitude
    }
    
    channel_type = channel_id[0] if channel_id[0] in base_patterns else 'P'
    params = base_patterns[channel_type]
    
    # Generate time series
    t = np.arange(n_points)
    
    # Base signal with trend
    signal = params['mean'] + params['trend'] * t
    
    # Add seasonal component (daily and orbital cycles)
    signal += params['seasonal_amp'] * np.sin(2 * np.pi * t / 1440)  # Daily (1440 min)
    signal += params['seasonal_amp'] * 0.3 * np.sin(2 * np.pi * t / 96)  # ~Orbital
    
    # Add noise
    noise = np.random.normal(0, params['std'], n_points)
    signal += noise
    
    # Add some drift periods
    drift_starts = np.random.choice(n_points, size=3, replace=False)
    for start in drift_starts:
        end = min(start + np.random.randint(100, 500), n_points)
        drift_rate = np.random.uniform(-0.001, 0.001)
        signal[start:end] += drift_rate * np.arange(end - start)
    
    # Inject anomalies
    anomaly_windows = []
    n_anomalies = np.random.randint(3, 8)
    
    for _ in range(n_anomalies):
        start_idx = np.random.randint(n_points // 4, 3 * n_points // 4)
        duration = np.random.randint(30, 200)  # 30 mins to 3+ hours
        end_idx = min(start_idx + duration, n_points - 1)
        
        # Different anomaly types
        anomaly_type = np.random.choice(['spike', 'step', 'drift', 'dropout'])
        
        if anomaly_type == 'spike':
            # Sharp spikes
            spike_magnitude = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
            signal[start_idx:end_idx] += spike_magnitude
            
        elif anomaly_type == 'step':
            # Step changes
            step_magnitude = np.random.uniform(0.2, 0.5) * np.random.choice([-1, 1])
            signal[start_idx:end_idx] += step_magnitude
            
        elif anomaly_type == 'drift':
            # Gradual drift
            drift_magnitude = np.random.uniform(0.1, 0.4) * np.random.choice([-1, 1])
            drift_pattern = np.linspace(0, drift_magnitude, end_idx - start_idx)
            signal[start_idx:end_idx] += drift_pattern
            
        elif anomaly_type == 'dropout':
            # Sensor dropouts/flatlines
            dropout_value = np.random.uniform(-0.8, 0.8)
            signal[start_idx:end_idx] = dropout_value
        
        anomaly_windows.append((start_idx, end_idx))
    
    # Clip to realistic range [-1, 1] like original data
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal, anomaly_windows

def load_channel_metadata() -> dict:
    """Load channel names and metadata."""
    # This would be expanded with full metadata
    return {
        "P-1": {"description": "Propulsion System Pressure", "units": "psi"},
        "S-1": {"description": "Solar Array Current", "units": "A"},
        "E-1": {"description": "Electrical Power Temperature", "units": "°C"},
        "T-1": {"description": "Thermal Control Temperature", "units": "°C"},
        "C-1": {"description": "Command & Data Temperature", "units": "°C"},
        "A-1": {"description": "Attitude Control Current", "units": "A"},
    }

def load_telemanom_data(channel_id: str, data_dir: Path) -> tuple[pd.DataFrame, list]:
    """Generate synthetic telemetry data that mimics the Telemanom dataset structure."""
    
    # Check if we have cached synthetic data
    cache_file = data_dir / f"{channel_id}_synthetic.npz"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if cache_file.exists():
        print(f"Loading cached synthetic data for {channel_id}")
        cached = np.load(cache_file)
        signal = cached['signal']
        anomaly_indices = cached['anomaly_indices']
    else:
        print(f"Generating synthetic data for {channel_id}...")
        signal, anomaly_indices = generate_synthetic_telemetry(channel_id)
        
        # Cache the generated data
        np.savez(cache_file, signal=signal, anomaly_indices=anomaly_indices)
    
    # Create timestamps (1-minute intervals starting from reference time)
    start_time = datetime(2018, 1, 1, tzinfo=timezone.utc)
    timestamps = [start_time + timedelta(minutes=i) for i in range(len(signal))]
    
    df = pd.DataFrame({
        'ts': timestamps,
        'value': signal
    })
    
    # Convert anomaly indices to timestamp windows
    anomaly_windows = []
    for start_idx, end_idx in anomaly_indices:
        if start_idx < len(timestamps) and end_idx < len(timestamps):
            anomaly_windows.append((timestamps[start_idx], timestamps[end_idx]))
    
    return df, anomaly_windows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-id", required=True, help="e.g., P-1, S-1, E-1")
    parser.add_argument("--data-dir", default="./data/telemanom", help="Local cache directory")
    parser.add_argument("--source", default="TELEMANOM", help="Source name in DB")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    metadata = load_channel_metadata()
    
    init_db()
    
    try:
        df, anomaly_windows = load_telemanom_data(args.channel_id, data_dir)
        
        with SessionLocal() as db:
            # Upsert channel
            channel = db.query(Channel).filter_by(
                source=args.source, 
                channel_key=args.channel_id
            ).first()
            
            if not channel:
                meta = metadata.get(args.channel_id, {})
                channel = Channel(
                    source=args.source,
                    channel_key=args.channel_id,
                    units=meta.get("units"),
                    meta={"description": meta.get("description", ""), "dataset": "telemanom"}
                )
                db.add(channel)
                db.commit()
                db.refresh(channel)
            
            # Load telemetry data
            print(f"Loading {len(df)} telemetry points...")
            telemetry_objs = []
            for _, row in df.iterrows():
                telemetry_objs.append(Telemetry(
                    channel_id=channel.id,
                    ts=row['ts'],
                    value=float(row['value'])
                ))
            
            # Clear existing data for this channel
            db.query(Telemetry).filter_by(channel_id=channel.id).delete()
            db.bulk_save_objects(telemetry_objs)
            
            # Load ground truth anomalies
            print(f"Loading {len(anomaly_windows)} labeled anomalies...")
            anomaly_objs = []
            for start_ts, end_ts in anomaly_windows:
                anomaly_objs.append(Anomaly(
                    channel_id=channel.id,
                    window_start=start_ts,
                    window_end=end_ts,
                    score=1.0,  # Ground truth
                    label=True,
                    method="ground_truth",
                    params={"dataset": "telemanom"}
                ))
            
            # Clear existing ground truth anomalies
            db.query(Anomaly).filter_by(channel_id=channel.id, method="ground_truth").delete()
            if anomaly_objs:
                db.bulk_save_objects(anomaly_objs)
            
            db.commit()
            print(f"✅ Loaded channel {args.channel_id} with {len(df)} points and {len(anomaly_windows)} anomalies")
            
    except Exception as e:
        print(f"❌ Error loading {args.channel_id}: {e}")
        raise

if __name__ == "__main__":
    main()