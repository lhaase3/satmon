#!/usr/bin/env python3
"""
Evaluate anomaly detection performance against ground truth labels.
Computes precision, recall, F1-score for different methods.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session
from pathlib import Path
import sys

# Add the project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.api.db import SessionLocal, init_db
from services.api.models import Anomaly
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def load_anomalies(db: Session, channel_id: int, method: str = None) -> pd.DataFrame:
    """Load anomalies from database."""
    q = db.query(Anomaly).filter_by(channel_id=channel_id)
    if method:
        q = q.filter_by(method=method)
    
    rows = q.all()
    if not rows:
        return pd.DataFrame(columns=['window_start', 'window_end', 'method', 'score'])
    
    return pd.DataFrame([{
        'window_start': r.window_start,
        'window_end': r.window_end,
        'method': r.method,
        'score': r.score,
        'label': r.label
    } for r in rows])

def compute_overlap_metrics(pred_windows: list, true_windows: list, tolerance_seconds: int = 60) -> dict:
    """
    Compute precision/recall based on window overlap.
    
    A predicted window is a True Positive if it overlaps with any ground truth window
    (within tolerance). A ground truth window is detected if any prediction overlaps it.
    """
    if not pred_windows:
        return {
            'precision': 0.0,
            'recall': 0.0 if true_windows else 1.0,
            'f1': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': len(true_windows)
        }
    
    if not true_windows:
        return {
            'precision': 0.0,
            'recall': 1.0,
            'f1': 0.0,
            'tp': 0,
            'fp': len(pred_windows),
            'fn': 0
        }
    
    # Convert to timestamps for easier comparison
    pred_intervals = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in pred_windows]
    true_intervals = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in true_windows]
    
    # Find overlaps
    tp = 0  # True positives
    detected_true = set()  # Track which ground truth windows were detected
    
    for i, (pred_start, pred_end) in enumerate(pred_intervals):
        found_overlap = False
        for j, (true_start, true_end) in enumerate(true_intervals):
            # Check if windows overlap (with tolerance)
            tolerance = pd.Timedelta(seconds=tolerance_seconds)
            overlap_start = max(pred_start - tolerance, true_start)
            overlap_end = min(pred_end + tolerance, true_end)
            
            if overlap_start <= overlap_end:
                found_overlap = True
                detected_true.add(j)
                break
        
        if found_overlap:
            tp += 1
    
    fp = len(pred_windows) - tp  # False positives
    fn = len(true_windows) - len(detected_true)  # False negatives
    
    precision = tp / len(pred_windows) if pred_windows else 0.0
    recall = tp / len(true_windows) if true_windows else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def evaluate_channel(db: Session, channel_id: int, methods: list = None) -> dict:
    """Evaluate all methods for a channel against ground truth."""
    
    # Load ground truth
    gt_df = load_anomalies(db, channel_id, method="ground_truth")
    if gt_df.empty:
        print(f"⚠️  No ground truth available for channel {channel_id}")
        return {}
    
    true_windows = [(row.window_start, row.window_end) for _, row in gt_df.iterrows()]
    print(f"Ground truth: {len(true_windows)} anomaly windows")
    
    # Load predictions for each method
    if not methods:
        methods = ["zscore", "isoforest"]
    
    results = {}
    for method in methods:
        pred_df = load_anomalies(db, channel_id, method=method)
        if pred_df.empty:
            print(f"⚠️  No predictions found for method '{method}'")
            continue
        
        pred_windows = [(row.window_start, row.window_end) for _, row in pred_df.iterrows()]
        print(f"{method}: {len(pred_windows)} predicted windows")
        
        metrics = compute_overlap_metrics(pred_windows, true_windows)
        results[method] = metrics
        
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1']:.3f}")
        print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        print()
    
    return results

def plot_comparison(results: dict, channel_id: int, save_path: str = None):
    """Create a comparison plot of different methods."""
    if not results:
        return
    
    methods = list(results.keys())
    metrics = ['precision', 'recall', 'f1']
    
    data = []
    for method in methods:
        for metric in metrics:
            data.append({
                'Method': method,
                'Metric': metric.upper(),
                'Score': results[method][metric]
            })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Method', y='Score', hue='Metric')
    plt.title(f'Anomaly Detection Performance - Channel {channel_id}')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-id", type=int, required=True)
    parser.add_argument("--methods", nargs="+", default=["zscore", "isoforest"])
    parser.add_argument("--plot", action="store_true", help="Generate comparison plot")
    parser.add_argument("--tolerance", type=int, default=60, help="Overlap tolerance in seconds")
    args = parser.parse_args()
    
    init_db()
    
    with SessionLocal() as db:
        print(f"Evaluating channel {args.channel_id}...")
        print("=" * 50)
        
        results = evaluate_channel(db, args.channel_id, args.methods)
        
        if results and args.plot:
            plot_comparison(results, args.channel_id)
        
        # Summary table
        if results:
            print("\nSummary:")
            print("-" * 50)
            print(f"{'Method':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 50)
            for method, metrics in results.items():
                print(f"{method:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1']:<10.3f}")

if __name__ == "__main__":
    main()