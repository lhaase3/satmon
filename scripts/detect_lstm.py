#!/usr/bin/env python3
"""
LSTM Autoencoder for multivariate anomaly detection.
Based on NASA JPL research methodologies.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import argparse
from pathlib import Path
import sys
import joblib
from datetime import datetime

# Add the project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.api.db import SessionLocal, init_db
from services.api.models import Telemetry, Anomaly

class LSTMAutoencoder:
    def __init__(self, sequence_length=60, n_features=1, encoding_dim=32):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        """Build LSTM autoencoder architecture."""
        # Encoder
        input_layer = keras.layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers with dropout for regularization
        x = keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(input_layer)
        x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
        encoded = keras.layers.LSTM(self.encoding_dim, return_sequences=False, dropout=0.2)(x)
        
        # Decoder
        x = keras.layers.RepeatVector(self.sequence_length)(encoded)
        x = keras.layers.LSTM(self.encoding_dim, return_sequences=True, dropout=0.2)(x)
        x = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
        decoded = keras.layers.TimeDistributed(keras.layers.Dense(self.n_features))(x)
        
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return self.model
    
    def create_sequences(self, data):
        """Create sequences for LSTM training."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:(i + self.sequence_length)])
        return np.array(sequences)
    
    def fit(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train the autoencoder."""
        # Normalize data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X = self.create_sequences(data_scaled)
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Add early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def detect_anomalies(self, data, threshold_percentile=95):
        """Detect anomalies using reconstruction error."""
        # Normalize data
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X = self.create_sequences(data_scaled)
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Calculate reconstruction errors (MSE)
        mse = np.mean(np.power(X - predictions, 2), axis=(1, 2))
        
        # Dynamic threshold based on training data distribution
        threshold = np.percentile(mse, threshold_percentile)
        
        # Find anomalies
        anomalies = mse > threshold
        
        # Convert back to original indices (accounting for sequence offset)
        anomaly_indices = []
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                # Use middle of sequence as anomaly point
                original_idx = i + self.sequence_length // 2
                anomaly_indices.append(original_idx)
        
        return anomaly_indices, mse, threshold
    
    def save_model(self, filepath):
        """Save model and scaler."""
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(model_path) + '_model.h5')
        
        # Save scaler
        joblib.dump(self.scaler, str(model_path) + '_scaler.pkl')
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'encoding_dim': self.encoding_dim
        }
        import json
        with open(str(model_path) + '_metadata.json', 'w') as f:
            json.dump(metadata, f)
    
    def load_model(self, filepath):
        """Load model and scaler."""
        model_path = Path(filepath)
        
        # Load model
        self.model = keras.models.load_model(str(model_path) + '_model.h5')
        
        # Load scaler
        self.scaler = joblib.load(str(model_path) + '_scaler.pkl')
        
        # Load metadata
        import json
        with open(str(model_path) + '_metadata.json', 'r') as f:
            metadata = json.load(f)
            self.sequence_length = metadata['sequence_length']
            self.n_features = metadata['n_features']
            self.encoding_dim = metadata['encoding_dim']

def group_consecutive_anomalies(anomaly_indices, timestamps, min_gap_minutes=5):
    """Group consecutive anomaly points into windows."""
    if not anomaly_indices:
        return []
    
    windows = []
    current_start = anomaly_indices[0]
    current_end = anomaly_indices[0]
    
    for i in range(1, len(anomaly_indices)):
        curr_idx = anomaly_indices[i]
        prev_idx = anomaly_indices[i-1]
        
        # Check time gap
        time_gap = (timestamps[curr_idx] - timestamps[prev_idx]).total_seconds() / 60
        
        if time_gap <= min_gap_minutes:
            # Extend current window
            current_end = curr_idx
        else:
            # Close current window and start new one
            windows.append((timestamps[current_start], timestamps[current_end]))
            current_start = curr_idx
            current_end = curr_idx
    
    # Add final window
    windows.append((timestamps[current_start], timestamps[current_end]))
    
    return windows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-id", type=int, required=True)
    parser.add_argument("--start", required=True, help="ISO timestamp")
    parser.add_argument("--end", required=True, help="ISO timestamp")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--encoding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threshold-percentile", type=float, default=95)
    parser.add_argument("--retrain", action="store_true", help="Force retrain model")
    args = parser.parse_args()
    
    init_db()
    
    with SessionLocal() as db:
        # Load telemetry data
        q = (db.query(Telemetry)
               .filter(Telemetry.channel_id == args.channel_id,
                       Telemetry.ts >= args.start,
                       Telemetry.ts <= args.end)
               .order_by(Telemetry.ts))
        rows = q.all()
        
        if len(rows) < args.sequence_length * 2:
            print(f"Not enough data. Need at least {args.sequence_length * 2} points.")
            return
        
        # Prepare data
        timestamps = [r.ts for r in rows]
        values = np.array([r.value for r in rows])
        
        # Initialize autoencoder
        autoencoder = LSTMAutoencoder(
            sequence_length=args.sequence_length,
            encoding_dim=args.encoding_dim
        )
        
        # Model path
        model_path = Path(f"models/lstm_autoencoder_ch{args.channel_id}")
        
        # Train or load model
        if args.retrain or not (model_path.parent / f"{model_path.name}_model.h5").exists():
            print("Training LSTM autoencoder...")
            
            # Split data (use first 70% for training)
            split_idx = int(0.7 * len(values))
            train_data = values[:split_idx]
            
            # Train model
            history = autoencoder.fit(train_data, epochs=args.epochs)
            
            # Save model
            autoencoder.save_model(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("Loading existing model...")
            autoencoder.load_model(model_path)
        
        # Detect anomalies on full dataset
        print("Detecting anomalies...")
        anomaly_indices, reconstruction_errors, threshold = autoencoder.detect_anomalies(
            values, threshold_percentile=args.threshold_percentile
        )
        
        print(f"Found {len(anomaly_indices)} anomalous points (threshold: {threshold:.4f})")
        
        # Group into windows
        anomaly_windows = group_consecutive_anomalies(anomaly_indices, timestamps)
        
        # Save to database
        anomaly_objs = []
        for window_start, window_end in anomaly_windows:
            # Calculate max reconstruction error in window
            start_idx = next(i for i, ts in enumerate(timestamps) if ts >= window_start)
            end_idx = next(i for i, ts in enumerate(timestamps) if ts >= window_end)
            
            # Find max error in the sequence range
            seq_start = max(0, start_idx - args.sequence_length // 2)
            seq_end = min(len(reconstruction_errors), end_idx - args.sequence_length // 2)
            
            if seq_start < seq_end and seq_start < len(reconstruction_errors):
                max_error = float(np.max(reconstruction_errors[seq_start:seq_end + 1]))
            else:
                max_error = float(threshold * 1.1)  # Slightly above threshold
            
            anomaly_objs.append(Anomaly(
                channel_id=args.channel_id,
                window_start=window_start,
                window_end=window_end,
                score=max_error,
                label=True,
                method="lstm_autoencoder",
                params={
                    "sequence_length": args.sequence_length,
                    "encoding_dim": args.encoding_dim,
                    "threshold_percentile": args.threshold_percentile,
                    "threshold": float(threshold)
                }
            ))
        
        if anomaly_objs:
            # Clear existing LSTM detections
            db.query(Anomaly).filter_by(
                channel_id=args.channel_id, 
                method="lstm_autoencoder"
            ).delete()
            
            db.bulk_save_objects(anomaly_objs)
            db.commit()
        
        print(f"Inserted {len(anomaly_objs)} anomaly windows.")

if __name__ == "__main__":
    main()