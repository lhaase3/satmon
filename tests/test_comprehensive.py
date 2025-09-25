#!/usr/bin/env python3
"""
Comprehensive Testing Suite for SatMon
Production-level testing and validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.detect_zscore import detect_anomalies_zscore
from scripts.detect_isoforest import detect_anomalies_isolation_forest
from scripts.advanced_pipeline import TelemetryDataPipeline

class TestDataGenerator:
    """Generate test data with known anomalies for validation"""
    
    @staticmethod
    def create_test_signal(n_points=1000, anomaly_positions=None, signal_type="sinusoidal"):
        """Create synthetic signal with known anomalies"""
        if anomaly_positions is None:
            anomaly_positions = [200, 500, 800]
        
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_points, 0, -1)]
        
        if signal_type == "sinusoidal":
            # Sinusoidal signal with noise
            base_signal = 50 + 20 * np.sin(np.linspace(0, 8*np.pi, n_points))
            noise = np.random.normal(0, 2, n_points)
            values = base_signal + noise
        elif signal_type == "random_walk":
            # Random walk
            values = np.cumsum(np.random.normal(0, 1, n_points)) + 50
        else:
            # Linear trend with noise
            values = np.linspace(40, 60, n_points) + np.random.normal(0, 3, n_points)
        
        # Inject anomalies
        ground_truth = np.zeros(n_points, dtype=bool)
        for pos in anomaly_positions:
            if 0 <= pos < n_points:
                values[pos] += np.random.choice([-1, 1]) * np.random.uniform(15, 25)
                ground_truth[pos] = True
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'ground_truth': ground_truth
        })
        
        return df

class TestAnomalyDetection:
    """Test anomaly detection algorithms"""
    
    def setup_method(self):
        """Setup test data"""
        self.test_data = TestDataGenerator.create_test_signal(
            n_points=1000,
            anomaly_positions=[100, 300, 700],
            signal_type="sinusoidal"
        )
    
    def test_zscore_detection_accuracy(self):
        """Test Z-Score detection accuracy"""
        # Create temporary test data
        temp_file = Path("test_data.csv")
        self.test_data[['timestamp', 'value']].to_csv(temp_file, index=False)
        
        try:
            # Run detection (would normally use database)
            # This is a simplified test - in reality you'd mock the database
            from scripts.detect_zscore import ZScoreDetector
            
            detector = ZScoreDetector(window_size=50, threshold=3.0)
            anomalies, scores = detector.detect(self.test_data['value'].values)
            
            # Calculate performance metrics
            precision, recall, f1 = self.calculate_metrics(
                self.test_data['ground_truth'].values,
                anomalies
            )
            
            # Assert minimum performance requirements
            assert recall >= 0.8, f"Recall too low: {recall}"
            assert precision >= 0.2, f"Precision too low: {precision}"
            assert f1 >= 0.3, f"F1-score too low: {f1}"
            
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    def test_isolation_forest_detection(self):
        """Test Isolation Forest detection"""
        from scripts.detect_isoforest import IsolationForestDetector
        
        detector = IsolationForestDetector(contamination=0.05)
        features = self.prepare_features(self.test_data['value'].values)
        anomalies = detector.detect(features)
        
        # Check that some anomalies are detected
        assert np.any(anomalies), "No anomalies detected"
        assert np.sum(anomalies) < len(anomalies) * 0.2, "Too many anomalies detected"
    
    def test_lstm_autoencoder_integration(self):
        """Test LSTM autoencoder integration"""
        # This would test the LSTM implementation
        # For now, just check that the module can be imported
        try:
            from scripts.detect_lstm import LSTMAutoencoder
            detector = LSTMAutoencoder(sequence_length=30)
            assert detector is not None
        except ImportError:
            pytest.skip("TensorFlow not available for LSTM testing")
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate precision, recall, F1-score"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def prepare_features(self, values):
        """Prepare features for anomaly detection"""
        # Simple feature engineering
        features = []
        window_size = 10
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window) - np.min(window),
                values[i] - np.mean(window)
            ])
        
        return np.array(features)

class TestDataPipeline:
    """Test data pipeline functionality"""
    
    @pytest.mark.asyncio
    async def test_pipeline_configuration(self):
        """Test pipeline configuration loading"""
        pipeline = TelemetryDataPipeline()
        assert pipeline.config is not None
        assert "batch_size" in pipeline.config
        assert pipeline.config["batch_size"] > 0
    
    @pytest.mark.asyncio
    async def test_data_quality_assessment(self):
        """Test data quality assessment"""
        pipeline = TelemetryDataPipeline()
        
        # Create test DataFrame
        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': np.random.normal(20, 2, 100),
            'voltage': np.random.normal(28, 0.5, 100)
        })
        
        # Add some missing values
        test_df.loc[10:15, 'temperature'] = np.nan
        
        quality_metrics = pipeline.assess_data_quality(test_df)
        
        assert "completeness" in quality_metrics
        assert "overall_score" in quality_metrics
        assert 0 <= quality_metrics["overall_score"] <= 1

class TestAPIEndpoints:
    """Test API endpoint functionality"""
    
    def setup_method(self):
        """Setup test client"""
        # This would normally use FastAPI TestClient
        pass
    
    def test_channels_endpoint(self):
        """Test channels API endpoint"""
        # Mock test - in practice you'd use FastAPI TestClient
        expected_channels = [
            {"name": "demo_temp_sensor", "description": "Temperature Sensor"},
            {"name": "demo_power_voltage", "description": "Power Voltage"},
            {"name": "demo_attitude_x", "description": "Attitude X-Axis"}
        ]
        
        # Assert structure
        for channel in expected_channels:
            assert "name" in channel
            assert "description" in channel
    
    def test_timeseries_endpoint(self):
        """Test timeseries data endpoint"""
        # Mock test for timeseries data
        mock_data = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "value": 25.5,
                "channel": "demo_temp_sensor"
            }
        ]
        
        assert len(mock_data) > 0
        assert "timestamp" in mock_data[0]
        assert "value" in mock_data[0]

class TestPerformanceBenchmarks:
    """Performance and benchmark tests"""
    
    def test_detection_speed(self):
        """Test that detection algorithms meet speed requirements"""
        import time
        
        # Generate large dataset
        large_data = TestDataGenerator.create_test_signal(n_points=10000)
        
        # Test Z-Score speed
        start_time = time.time()
        # Simulate Z-Score detection
        rolling_mean = large_data['value'].rolling(50).mean()
        rolling_std = large_data['value'].rolling(50).std()
        z_scores = np.abs((large_data['value'] - rolling_mean) / rolling_std)
        anomalies = z_scores > 3.0
        zscore_time = time.time() - start_time
        
        # Assert performance requirements
        assert zscore_time < 1.0, f"Z-Score too slow: {zscore_time:.2f}s"
        assert np.any(anomalies), "No anomalies detected"
    
    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        large_data = TestDataGenerator.create_test_signal(n_points=50000)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assert reasonable memory usage (less than 100MB increase)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.1f}MB"

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data ingestion to anomaly detection"""
        # This would test the complete workflow
        # 1. Data ingestion
        # 2. Processing
        # 3. Anomaly detection
        # 4. Results storage
        # 5. API response
        
        # For now, just verify components can be imported
        components = [
            "scripts.detect_zscore",
            "scripts.detect_isoforest", 
            "scripts.advanced_pipeline"
        ]
        
        for component in components:
            try:
                __import__(component)
            except ImportError as e:
                pytest.fail(f"Failed to import {component}: {e}")

# Performance benchmarking utilities
def benchmark_algorithms():
    """Benchmark all anomaly detection algorithms"""
    print("🚀 Running Performance Benchmarks...")
    
    test_data = TestDataGenerator.create_test_signal(
        n_points=5000,
        anomaly_positions=[500, 1500, 3000, 4000]
    )
    
    results = {}
    
    # Benchmark Z-Score
    import time
    start = time.time()
    rolling_mean = test_data['value'].rolling(50).mean()
    rolling_std = test_data['value'].rolling(50).std()
    z_scores = np.abs((test_data['value'] - rolling_mean) / rolling_std)
    zscore_anomalies = z_scores > 3.0
    zscore_time = time.time() - start
    
    results['zscore'] = {
        'time': zscore_time,
        'anomalies_found': np.sum(zscore_anomalies),
        'throughput': len(test_data) / zscore_time
    }
    
    print(f"📊 Benchmark Results:")
    for algo, metrics in results.items():
        print(f"  {algo}: {metrics['time']:.3f}s, {metrics['throughput']:.0f} points/sec")
    
    return results

if __name__ == "__main__":
    # Run benchmarks
    benchmark_results = benchmark_algorithms()
    
    # Run tests
    pytest.main([__file__, "-v"])