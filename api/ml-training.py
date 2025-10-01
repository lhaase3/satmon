from http.server import BaseHTTPRequestHandler
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import random

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            # Parse request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            action = request_data.get('action', 'retrain')
            
            if action == 'retrain':
                result = self.retrain_models()
            elif action == 'deploy':
                result = self.deploy_model(request_data.get('model_id'))
            elif action == 'compare':
                result = self.compare_models()
            else:
                raise ValueError(f"Unknown action: {action}")
            
            self.wfile.write(json.dumps(result, indent=2).encode('utf-8'))
            
        except Exception as e:
            error_response = {
                "error": str(e),
                "status": "error",
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            # Return current training status and model versions
            current_time = time.time()
            
            response_data = {
                "timestamp": current_time,
                "status": "success",
                "training_pipeline": {
                    "current_model_version": "v2.1.4",
                    "last_retrain": current_time - 3600,  # 1 hour ago
                    "next_scheduled": current_time + 7200,  # 2 hours from now
                    "training_status": "idle",
                    "model_drift_detected": False,
                    "retrain_trigger": "scheduled"
                },
                "model_versions": [
                    {
                        "version": "v2.1.4",
                        "status": "production",
                        "accuracy": 94.7,
                        "f1_score": 0.912,
                        "deployed_at": current_time - 3600,
                        "training_samples": 15847
                    },
                    {
                        "version": "v2.1.3", 
                        "status": "archived",
                        "accuracy": 93.2,
                        "f1_score": 0.889,
                        "deployed_at": current_time - 86400,  # 1 day ago
                        "training_samples": 14203
                    },
                    {
                        "version": "v2.2.0-beta",
                        "status": "testing",
                        "accuracy": 95.1,
                        "f1_score": 0.923,
                        "deployed_at": None,
                        "training_samples": 16492
                    }
                ],
                "a_b_testing": {
                    "active_test": True,
                    "champion": "v2.1.4",
                    "challenger": "v2.2.0-beta",
                    "traffic_split": {"champion": 80, "challenger": 20},
                    "test_start": current_time - 172800,  # 2 days ago
                    "statistical_significance": 0.89,
                    "winner_threshold": 0.95
                },
                "data_quality": {
                    "completeness": 99.2,
                    "consistency": 97.8,
                    "timeliness": 98.5,
                    "accuracy": 96.1,
                    "issues_detected": 3,
                    "last_check": current_time - 300  # 5 minutes ago
                }
            }
            
            self.wfile.write(json.dumps(response_data, indent=2).encode('utf-8'))
            
        except Exception as e:
            error_response = {
                "error": str(e),
                "status": "error",
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def retrain_models(self):
        """Simulate model retraining with real data"""
        try:
            current_time = time.time()
            
            # Load and prepare data (similar to ml-analytics.py)
            data_dir = Path(__file__).parent.parent / "data"
            demo_file = data_dir / "demo_temp.csv"
            
            if not demo_file.exists():
                raise FileNotFoundError("Training data not found")
            
            df = pd.read_csv(demo_file)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts').reset_index(drop=True)
            
            # Feature engineering
            df['value_lag1'] = df['value'].shift(1)
            df['value_lag2'] = df['value'].shift(2)
            df['rolling_mean'] = df['value'].rolling(window=10, min_periods=1).mean()
            df['rolling_std'] = df['value'].rolling(window=10, min_periods=1).std()
            df['rate_change'] = df['value'].diff()
            df = df.dropna()
            
            # Prepare features
            feature_cols = ['value', 'value_lag1', 'value_lag2', 'rolling_mean', 'rolling_std', 'rate_change']
            X = df[feature_cols].values
            
            # Create ground truth with realistic anomaly patterns
            y = np.zeros(len(df), dtype=bool)
            
            # Inject different types of anomalies
            # 1. Spike anomalies
            spike_indices = np.random.choice(len(df), size=5, replace=False)
            y[spike_indices] = True
            
            # 2. Drift anomalies (gradual changes)
            drift_start = len(df) // 2
            y[drift_start:drift_start+10] = True
            
            # Split data for training/testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple model variants
            models = {}
            
            # Model 1: Conservative (low contamination)
            iso_conservative = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
            iso_conservative.fit(X_train_scaled)
            pred_conservative = (iso_conservative.predict(X_test_scaled) == -1)
            
            # Model 2: Standard (medium contamination)
            iso_standard = IsolationForest(contamination=0.1, random_state=42, n_estimators=150)
            iso_standard.fit(X_train_scaled)
            pred_standard = (iso_standard.predict(X_test_scaled) == -1)
            
            # Model 3: Aggressive (high contamination)
            iso_aggressive = IsolationForest(contamination=0.15, random_state=42, n_estimators=200)
            iso_aggressive.fit(X_train_scaled)
            pred_aggressive = (iso_aggressive.predict(X_test_scaled) == -1)
            
            # Calculate metrics for each model
            models['conservative'] = {
                "name": "Conservative IsoForest",
                "contamination": 0.05,
                "precision": float(precision_score(y_test, pred_conservative, zero_division=0)),
                "recall": float(recall_score(y_test, pred_conservative, zero_division=0)),
                "f1_score": float(f1_score(y_test, pred_conservative, zero_division=0)),
                "accuracy": float(accuracy_score(y_test, pred_conservative)),
                "anomalies_detected": int(np.sum(pred_conservative))
            }
            
            models['standard'] = {
                "name": "Standard IsoForest",
                "contamination": 0.1,
                "precision": float(precision_score(y_test, pred_standard, zero_division=0)),
                "recall": float(recall_score(y_test, pred_standard, zero_division=0)),
                "f1_score": float(f1_score(y_test, pred_standard, zero_division=0)),
                "accuracy": float(accuracy_score(y_test, pred_standard)),
                "anomalies_detected": int(np.sum(pred_standard))
            }
            
            models['aggressive'] = {
                "name": "Aggressive IsoForest",
                "contamination": 0.15,
                "precision": float(precision_score(y_test, pred_aggressive, zero_division=0)),
                "recall": float(recall_score(y_test, pred_aggressive, zero_division=0)),
                "f1_score": float(f1_score(y_test, pred_aggressive, zero_division=0)),
                "accuracy": float(accuracy_score(y_test, pred_aggressive)),
                "anomalies_detected": int(np.sum(pred_aggressive))
            }
            
            # Determine best model
            best_model = max(models.keys(), key=lambda k: models[k]['f1_score'])
            
            return {
                "status": "success",
                "timestamp": current_time,
                "training_completed": True,
                "training_duration_seconds": 45.7,
                "dataset_info": {
                    "total_samples": len(df),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": len(feature_cols),
                    "anomaly_rate": float(np.mean(y))
                },
                "model_candidates": models,
                "recommended_model": {
                    "type": best_model,
                    "reason": f"Best F1-score: {models[best_model]['f1_score']:.3f}",
                    "performance": models[best_model]
                },
                "next_steps": [
                    "Deploy recommended model to staging",
                    "Run A/B test against current production model", 
                    "Monitor performance for 24 hours",
                    "Promote to production if metrics improve"
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Training failed: {str(e)}",
                "timestamp": time.time()
            }

    def deploy_model(self, model_id):
        """Simulate model deployment"""
        current_time = time.time()
        
        return {
            "status": "success",
            "timestamp": current_time,
            "deployment": {
                "model_id": model_id or "v2.2.0",
                "deployment_type": "canary",
                "traffic_percentage": 5,
                "rollout_schedule": "gradual_over_2_hours",
                "health_checks": {
                    "api_latency": "passing",
                    "error_rate": "passing", 
                    "memory_usage": "passing",
                    "accuracy_threshold": "monitoring"
                },
                "rollback_plan": "automatic_if_error_rate_exceeds_1_percent"
            }
        }

    def compare_models(self):
        """Compare model performance"""
        current_time = time.time()
        
        return {
            "status": "success", 
            "timestamp": current_time,
            "comparison": {
                "timeframe": "last_7_days",
                "models": [
                    {
                        "version": "v2.1.4",
                        "requests": 145892,
                        "avg_latency_ms": 23.4,
                        "accuracy": 94.7,
                        "false_positive_rate": 2.1,
                        "uptime": 99.97
                    },
                    {
                        "version": "v2.2.0-beta",
                        "requests": 36473,
                        "avg_latency_ms": 21.8,
                        "accuracy": 95.1,
                        "false_positive_rate": 1.8,
                        "uptime": 99.95
                    }
                ],
                "winner": "v2.2.0-beta",
                "improvement": {
                    "accuracy": "+0.4%",
                    "latency": "-1.6ms",
                    "false_positives": "-0.3%"
                }
            }
        }

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()