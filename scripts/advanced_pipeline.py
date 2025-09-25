#!/usr/bin/env python3
"""
Advanced Data Pipeline for SatMon
Demonstrates production-level data engineering capabilities
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import aiofiles
import asyncpg
from pydantic import BaseModel

class TelemetryDataPipeline:
    """Production-grade data pipeline for satellite telemetry"""
    
    def __init__(self, config_path: str = "pipeline_config.json"):
        self.config = self.load_config(config_path)
        self.processed_files = set()
        
    def load_config(self, path: str) -> Dict:
        """Load pipeline configuration"""
        default_config = {
            "batch_size": 1000,
            "max_concurrent_files": 5,
            "quality_thresholds": {
                "min_points_per_hour": 50,
                "max_gap_minutes": 10,
                "outlier_zscore_threshold": 5.0
            },
            "data_sources": {
                "nasa_datasets": True,
                "synthetic_generation": True,
                "real_time_stream": False
            }
        }
        
        try:
            with open(path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            return default_config
    
    async def process_data_batch(self, files: List[Path]) -> Dict:
        """Process multiple data files concurrently"""
        semaphore = asyncio.Semaphore(self.config["max_concurrent_files"])
        
        async def process_single_file(file_path: Path):
            async with semaphore:
                return await self.process_telemetry_file(file_path)
        
        tasks = [process_single_file(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        return {
            "processed": len(successful),
            "failed": len(failed),
            "total_points": sum(r.get("points", 0) for r in successful),
            "quality_score": np.mean([r.get("quality", 0) for r in successful])
        }
    
    async def process_telemetry_file(self, file_path: Path) -> Dict:
        """Process individual telemetry file with quality checks"""
        try:
            # Read file asynchronously
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            
            # Parse data (assume CSV format)
            df = pd.read_csv(io.StringIO(content))
            
            # Data quality assessment
            quality_metrics = self.assess_data_quality(df)
            
            # Clean and validate data
            cleaned_df = self.clean_telemetry_data(df)
            
            # Extract features for anomaly detection
            features = self.extract_features(cleaned_df)
            
            return {
                "file": str(file_path),
                "points": len(cleaned_df),
                "quality": quality_metrics["overall_score"],
                "features": features,
                "processing_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Comprehensive data quality assessment"""
        metrics = {}
        
        # Completeness
        metrics["completeness"] = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        # Temporal consistency
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_gaps = df['timestamp'].diff().dt.total_seconds() / 60  # minutes
            metrics["max_gap_minutes"] = time_gaps.max()
            metrics["avg_gap_minutes"] = time_gaps.mean()
        
        # Value consistency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            metrics[f"{col}_outlier_rate"] = (z_scores > 5).mean()
        
        # Overall quality score (0-1)
        completeness_score = metrics["completeness"]
        gap_score = max(0, 1 - metrics.get("max_gap_minutes", 0) / 60)  # penalize gaps > 1 hour
        outlier_score = 1 - np.mean([metrics.get(f"{col}_outlier_rate", 0) for col in numeric_cols])
        
        metrics["overall_score"] = np.mean([completeness_score, gap_score, outlier_score])
        
        return metrics
    
    def clean_telemetry_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess telemetry data"""
        cleaned = df.copy()
        
        # Remove duplicates
        cleaned = cleaned.drop_duplicates()
        
        # Handle missing values
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Forward fill small gaps, interpolate larger ones
            cleaned[col] = cleaned[col].fillna(method='ffill', limit=3)
            cleaned[col] = cleaned[col].interpolate(method='linear', limit=10)
        
        # Remove extreme outliers (beyond 6 sigma)
        for col in numeric_cols:
            mean_val = cleaned[col].mean()
            std_val = cleaned[col].std()
            mask = np.abs(cleaned[col] - mean_val) < 6 * std_val
            cleaned = cleaned[mask]
        
        return cleaned
    
    def extract_features(self, df: pd.DataFrame) -> Dict:
        """Extract statistical features for anomaly detection"""
        features = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = df[col].dropna()
            
            features[f"{col}_mean"] = float(values.mean())
            features[f"{col}_std"] = float(values.std())
            features[f"{col}_min"] = float(values.min())
            features[f"{col}_max"] = float(values.max())
            features[f"{col}_skew"] = float(values.skew())
            features[f"{col}_kurtosis"] = float(values.kurtosis())
            
            # Trend analysis
            if len(values) > 10:
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                features[f"{col}_trend_slope"] = float(slope)
        
        return features

class RealTimeStreamProcessor:
    """Real-time telemetry stream processing"""
    
    def __init__(self):
        self.buffer = []
        self.window_size = 100
        
    async def process_stream(self, data_generator):
        """Process streaming telemetry data"""
        async for data_point in data_generator:
            # Add to buffer
            self.buffer.append(data_point)
            
            # Process when buffer is full
            if len(self.buffer) >= self.window_size:
                await self.process_buffer()
                self.buffer = []
    
    async def process_buffer(self):
        """Process buffered data points"""
        df = pd.DataFrame(self.buffer)
        
        # Quick anomaly detection on streaming data
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col]
            z_scores = np.abs((values - values.mean()) / values.std())
            anomalies = z_scores > 3
            
            if anomalies.any():
                print(f"🚨 Real-time anomaly detected in {col}")
                # Trigger alerts, store in database, etc.

# Usage example and configuration
async def main():
    """Example usage of the data pipeline"""
    pipeline = TelemetryDataPipeline()
    
    # Process batch of files
    data_files = list(Path("data").glob("*.csv"))
    if data_files:
        results = await pipeline.process_data_batch(data_files)
        print(f"Pipeline Results: {results}")
    
    # Real-time processing simulation
    processor = RealTimeStreamProcessor()
    
    async def mock_data_stream():
        """Mock real-time data stream"""
        for i in range(1000):
            yield {
                "timestamp": datetime.now().isoformat(),
                "temperature": 20 + 5 * np.sin(i * 0.1) + np.random.normal(0, 1),
                "voltage": 28 + np.random.normal(0, 0.5),
                "channel": "mock_sensor"
            }
            await asyncio.sleep(0.01)  # 100 Hz simulation
    
    # await processor.process_stream(mock_data_stream())

if __name__ == "__main__":
    asyncio.run(main())