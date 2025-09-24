#!/usr/bin/env python3
"""Quick script to check what's in our database."""

import sys
from pathlib import Path

# Add the project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.api.db import SessionLocal, init_db
from services.api.models import Channel, Telemetry, Anomaly
from datetime import datetime

def main():
    init_db()
    
    with SessionLocal() as db:
        # Check channels
        channels = db.query(Channel).all()
        print(f"📡 Found {len(channels)} channels:")
        for ch in channels:
            print(f"  ID: {ch.id}, Source: {ch.source}, Key: {ch.channel_key}, Units: {ch.units}")
        
        if channels:
            # Check telemetry for first channel
            ch = channels[0]
            telemetry_count = db.query(Telemetry).filter_by(channel_id=ch.id).count()
            print(f"\n📊 Channel {ch.id} has {telemetry_count} telemetry points")
            
            # Show date range
            first_point = db.query(Telemetry).filter_by(channel_id=ch.id).order_by(Telemetry.ts).first()
            last_point = db.query(Telemetry).filter_by(channel_id=ch.id).order_by(Telemetry.ts.desc()).first()
            
            if first_point and last_point:
                print(f"  📅 Date range: {first_point.ts} to {last_point.ts}")
            
            # Check ground truth anomalies
            gt_anomalies = db.query(Anomaly).filter_by(channel_id=ch.id, method="ground_truth").count()
            print(f"  🎯 Ground truth anomalies: {gt_anomalies}")
            
            # Check detected anomalies
            detected = db.query(Anomaly).filter_by(channel_id=ch.id).filter(Anomaly.method != "ground_truth").count()
            print(f"  🔍 Detected anomalies: {detected}")

if __name__ == "__main__":
    main()