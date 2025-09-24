#!/usr/bin/env python3
"""Script to inspect anomaly windows and understand the evaluation results."""

import sys
from pathlib import Path

# Add the project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.api.db import SessionLocal, init_db
from services.api.models import Anomaly
import pandas as pd

def main():
    init_db()
    
    with SessionLocal() as db:
        channel_id = 2  # P-1 channel
        
        print(f"🔍 Inspecting anomaly windows for channel {channel_id}")
        print("=" * 60)
        
        # Get all anomalies for this channel
        anomalies = db.query(Anomaly).filter_by(channel_id=channel_id).order_by(Anomaly.window_start).all()
        
        methods = {}
        for anom in anomalies:
            if anom.method not in methods:
                methods[anom.method] = []
            methods[anom.method].append(anom)
        
        for method, anoms in methods.items():
            print(f"\n📊 {method.upper()} ({len(anoms)} windows):")
            for i, anom in enumerate(anoms, 1):
                duration = anom.window_end - anom.window_start
                print(f"  {i:2d}. {anom.window_start} to {anom.window_end}")
                print(f"      Duration: {duration}, Score: {anom.score:.3f}")
        
        # Look for overlaps between ground truth and detected
        if 'ground_truth' in methods:
            gt_windows = methods['ground_truth']
            print(f"\n🎯 Checking overlaps with {len(gt_windows)} ground truth windows...")
            
            for det_method, det_anoms in methods.items():
                if det_method == 'ground_truth':
                    continue
                    
                print(f"\n{det_method}:")
                overlaps = 0
                for gt in gt_windows:
                    for det in det_anoms:
                        # Check for any overlap
                        if (det.window_start <= gt.window_end and 
                            det.window_end >= gt.window_start):
                            print(f"  ✅ OVERLAP: GT {gt.window_start}-{gt.window_end} ↔ DET {det.window_start}-{det.window_end}")
                            overlaps += 1
                            break
                if overlaps == 0:
                    print(f"  ❌ No overlaps found")
                else:
                    print(f"  Found {overlaps} overlapping windows")

if __name__ == "__main__":
    main()