"""
Advanced ML Models for Satellite Anomaly Detection
Implements state-of-the-art algorithms for real-time anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import List, Dict, Tuple
import joblib
from datetime import datetime, timedelta

class AdvancedAnomalyDetector:
    """Advanced ML models for satellite telemetry anomaly detection"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'temperature', 'pressure', 'voltage', 'current', 
            'altitude', 'velocity', 'acceleration', 'gyro_x', 'gyro_y', 'gyro_z'
        ]
    
    def build_transformer_model(self, sequence_length: int = 50):
        """Build Transformer-based anomaly detection model"""
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        
        inputs = tf.keras.Input(shape=(sequence_length, len(self.feature_names)))
        
        # Multi-head attention layers
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=64
        )(inputs, inputs)
        
        attention_output = LayerNormalization()(attention_output + inputs)
        
        # Feed forward network
        ffn_output = Dense(256, activation='relu')(attention_output)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(len(self.feature_names))(ffn_output)
        
        outputs = LayerNormalization()(ffn_output + attention_output)
        
        # Final layers for anomaly scoring
        x = tf.keras.layers.GlobalAveragePooling1D()(outputs)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        anomaly_score = Dense(1, activation='sigmoid', name='anomaly_score')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=anomaly_score)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_variational_autoencoder(self, latent_dim: int = 32):
        """Build Variational Autoencoder for unsupervised anomaly detection"""
        input_dim = len(self.feature_names)
        
        # Encoder
        encoder_inputs = tf.keras.Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(encoder_inputs)
        x = Dense(64, activation='relu')(x)
        
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_inputs = tf.keras.Input(shape=(latent_dim,))
        x = Dense(64, activation='relu')(decoder_inputs)
        x = Dense(128, activation='relu')(x)
        decoder_outputs = Dense(input_dim, activation='linear')(x)
        
        # Models
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z])
        decoder = tf.keras.Model(decoder_inputs, decoder_outputs)
        
        # VAE
        vae_outputs = decoder(encoder(encoder_inputs)[2])
        vae = tf.keras.Model(encoder_inputs, vae_outputs)
        
        # Loss function
        def vae_loss(y_true, y_pred):
            reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
            )
            return reconstruction_loss + kl_loss
        
        vae.compile(optimizer='adam', loss=vae_loss)
        return vae, encoder, decoder
    
    def ensemble_prediction(self, data: np.ndarray) -> Dict:
        """Combine predictions from multiple models"""
        predictions = {}
        
        # Traditional models
        if 'isolation_forest' in self.models:
            predictions['isolation_forest'] = self.models['isolation_forest'].predict(data)
        
        if 'lstm' in self.models:
            # Reshape for LSTM if needed
            lstm_data = data.reshape((data.shape[0], 1, data.shape[1]))
            predictions['lstm'] = self.models['lstm'].predict(lstm_data)
        
        if 'transformer' in self.models:
            # Reshape for Transformer
            seq_data = data.reshape((data.shape[0], 1, data.shape[1]))
            predictions['transformer'] = self.models['transformer'].predict(seq_data)
        
        # Ensemble voting
        anomaly_scores = []
        for model_name, preds in predictions.items():
            if len(preds.shape) > 1:
                anomaly_scores.append(preds.flatten())
            else:
                anomaly_scores.append(preds)
        
        if anomaly_scores:
            ensemble_score = np.mean(anomaly_scores, axis=0)
            confidence = 1 - np.std(anomaly_scores, axis=0)
        else:
            ensemble_score = np.zeros(len(data))
            confidence = np.zeros(len(data))
        
        return {
            'anomaly_scores': ensemble_score.tolist(),
            'confidence': confidence.tolist(),
            'individual_predictions': predictions,
            'consensus_anomalies': (ensemble_score > 0.5).tolist()
        }
    
    def real_time_stream_detection(self, data_stream: List[Dict]) -> Dict:
        """Process streaming telemetry data in real-time"""
        results = []
        
        for data_point in data_stream:
            # Extract features
            features = [data_point.get(feature, 0) for feature in self.feature_names]
            features_array = np.array([features])
            
            # Scale features
            if 'scaler' in self.scalers:
                features_scaled = self.scalers['scaler'].transform(features_array)
            else:
                features_scaled = features_array
            
            # Get predictions
            prediction = self.ensemble_prediction(features_scaled)
            
            result = {
                'timestamp': data_point.get('timestamp', datetime.utcnow().isoformat()),
                'channel': data_point.get('channel', 'unknown'),
                'raw_values': features,
                'anomaly_score': prediction['anomaly_scores'][0],
                'confidence': prediction['confidence'][0],
                'is_anomaly': prediction['consensus_anomalies'][0],
                'model_breakdown': prediction['individual_predictions']
            }
            
            results.append(result)
        
        return {
            'processed_points': len(results),
            'anomalies_detected': sum(r['is_anomaly'] for r in results),
            'average_confidence': np.mean([r['confidence'] for r in results]),
            'results': results
        }

# Real-time processing pipeline
class RealTimeTelemetryProcessor:
    """Process satellite telemetry in real-time with ML predictions"""
    
    def __init__(self):
        self.detector = AdvancedAnomalyDetector()
        self.buffer = []
        self.buffer_size = 100
        self.alert_threshold = 0.8
    
    def process_telemetry_point(self, telemetry: Dict) -> Dict:
        """Process a single telemetry point"""
        # Add to buffer
        self.buffer.append(telemetry)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Process current point
        result = self.detector.real_time_stream_detection([telemetry])
        
        # Check for alerts
        current_result = result['results'][0]
        if current_result['anomaly_score'] > self.alert_threshold:
            alert = {
                'level': 'HIGH' if current_result['anomaly_score'] > 0.9 else 'MEDIUM',
                'message': f"Anomaly detected in {current_result['channel']}",
                'score': current_result['anomaly_score'],
                'timestamp': current_result['timestamp'],
                'recommended_action': self._get_recommended_action(current_result)
            }
            current_result['alert'] = alert
        
        return current_result
    
    def _get_recommended_action(self, result: Dict) -> str:
        """Get recommended action based on anomaly type"""
        score = result['anomaly_score']
        channel = result['channel']
        
        if 'temperature' in channel.lower():
            if score > 0.9:
                return "CRITICAL: Check thermal systems immediately"
            return "Monitor thermal conditions closely"
        elif 'power' in channel.lower() or 'voltage' in channel.lower():
            if score > 0.9:
                return "CRITICAL: Check power systems and battery health"
            return "Monitor power consumption patterns"
        elif 'attitude' in channel.lower() or 'gyro' in channel.lower():
            if score > 0.9:
                return "CRITICAL: Check attitude control systems"
            return "Monitor spacecraft orientation stability"
        else:
            if score > 0.9:
                return "CRITICAL: Investigate system anomaly immediately"
            return "Continue monitoring system parameters"

# Performance monitoring
class ModelPerformanceMonitor:
    """Monitor ML model performance and trigger retraining"""
    
    def __init__(self):
        self.performance_history = []
        self.drift_threshold = 0.1
    
    def evaluate_model_drift(self, recent_predictions: List[Dict]) -> Dict:
        """Detect if model performance is degrading"""
        if len(recent_predictions) < 100:
            return {"drift_detected": False, "confidence": "insufficient_data"}
        
        # Calculate recent accuracy metrics
        recent_scores = [p['confidence'] for p in recent_predictions]
        avg_confidence = np.mean(recent_scores)
        confidence_std = np.std(recent_scores)
        
        # Compare with historical performance
        if self.performance_history:
            historical_avg = np.mean([h['avg_confidence'] for h in self.performance_history])
            drift_magnitude = abs(avg_confidence - historical_avg)
            
            drift_detected = drift_magnitude > self.drift_threshold
        else:
            drift_detected = False
            drift_magnitude = 0
        
        self.performance_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'sample_size': len(recent_predictions)
        })
        
        return {
            'drift_detected': drift_detected,
            'drift_magnitude': drift_magnitude,
            'current_confidence': avg_confidence,
            'recommendation': 'retrain_model' if drift_detected else 'continue_monitoring'
        }