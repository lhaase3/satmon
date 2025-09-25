#!/usr/bin/env python3
"""
NASA Data Integration for SatMon
Downloads and processes real spacecraft telemetry data
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class NASADataLoader:
    """Load real NASA spacecraft telemetry data"""
    
    def __init__(self, data_dir: Path = Path("data/nasa")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # NASA Open Data Portal endpoints
        self.endpoints = {
            "telemanom": "https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
            "msl_weather": "https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json",
            "iss_telemetry": "http://api.open-notify.org/iss-now.json"
        }
    
    def download_telemanom_dataset(self) -> Path:
        """Download NASA's Telemanom anomaly detection dataset"""
        print("🛰️ Downloading NASA Telemanom dataset...")
        
        zip_path = self.data_dir / "telemanom.zip"
        extract_path = self.data_dir / "telemanom"
        
        if not extract_path.exists():
            # Download dataset
            response = requests.get(self.endpoints["telemanom"], stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            print(f"✅ Dataset extracted to {extract_path}")
        
        return extract_path
    
    def load_spacecraft_channels(self, dataset_path: Path) -> List[Dict]:
        """Load real spacecraft telemetry channels"""
        channels = []
        
        # Look for telemetry files
        for file_path in dataset_path.rglob("*.csv"):
            if "train" in file_path.name:
                channel_info = self.analyze_channel_file(file_path)
                if channel_info:
                    channels.append(channel_info)
        
        return channels
    
    def analyze_channel_file(self, file_path: Path) -> Optional[Dict]:
        """Analyze individual channel file"""
        try:
            df = pd.read_csv(file_path)
            
            if len(df) < 100:  # Skip small files
                return None
            
            # Extract channel metadata
            channel_name = file_path.stem
            
            # Infer channel type from name patterns
            channel_type = self.infer_channel_type(channel_name)
            
            # Calculate statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            primary_col = numeric_cols[0]  # Assume first numeric column is primary
            values = df[primary_col].dropna()
            
            return {
                "name": channel_name,
                "type": channel_type,
                "file_path": str(file_path),
                "data_points": len(values),
                "mean_value": float(values.mean()),
                "std_value": float(values.std()),
                "min_value": float(values.min()),
                "max_value": float(values.max()),
                "primary_column": primary_col,
                "spacecraft": self.infer_spacecraft(channel_name)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return None
    
    def infer_channel_type(self, channel_name: str) -> str:
        """Infer channel type from naming patterns"""
        name_lower = channel_name.lower()
        
        if any(temp in name_lower for temp in ['temp', 'thermal', 'heat']):
            return "temperature"
        elif any(power in name_lower for power in ['power', 'volt', 'curr', 'batt']):
            return "power"
        elif any(att in name_lower for att in ['att', 'gyro', 'orient', 'angle']):
            return "attitude"
        elif any(prop in name_lower for prop in ['prop', 'fuel', 'thrust']):
            return "propulsion"
        elif any(comm in name_lower for comm in ['comm', 'radio', 'antenna']):
            return "communication"
        else:
            return "unknown"
    
    def infer_spacecraft(self, channel_name: str) -> str:
        """Infer spacecraft from channel naming"""
        name_lower = channel_name.lower()
        
        if 'msl' in name_lower or 'curiosity' in name_lower:
            return "Mars Science Laboratory"
        elif 'iss' in name_lower:
            return "International Space Station"
        elif 'smap' in name_lower:
            return "Soil Moisture Active Passive"
        else:
            return "Unknown Spacecraft"
    
    def create_channel_catalog(self, channels: List[Dict]) -> pd.DataFrame:
        """Create a catalog of available channels"""
        df = pd.DataFrame(channels)
        
        # Add derived metrics
        df['data_quality'] = df.apply(self.calculate_data_quality, axis=1)
        df['anomaly_potential'] = df.apply(self.estimate_anomaly_potential, axis=1)
        
        # Sort by data quality and size
        df = df.sort_values(['data_quality', 'data_points'], ascending=[False, False])
        
        return df
    
    def calculate_data_quality(self, row: pd.Series) -> float:
        """Calculate data quality score for a channel"""
        # Based on data completeness, reasonable ranges, etc.
        points_score = min(1.0, row['data_points'] / 10000)  # Prefer more data
        range_score = 1.0 if row['std_value'] > 0 else 0.0   # Must have variation
        
        return (points_score + range_score) / 2
    
    def estimate_anomaly_potential(self, row: pd.Series) -> str:
        """Estimate how likely this channel is to have interesting anomalies"""
        if row['std_value'] / abs(row['mean_value']) > 0.1:  # High coefficient of variation
            return "high"
        elif row['data_points'] > 5000:  # Lots of data increases chance
            return "medium"
        else:
            return "low"

def load_real_nasa_data():
    """Main function to load and catalog NASA data"""
    loader = NASADataLoader()
    
    try:
        # Download and extract dataset
        dataset_path = loader.download_telemanom_dataset()
        
        # Load all channels
        print("📊 Analyzing spacecraft channels...")
        channels = loader.load_spacecraft_channels(dataset_path)
        
        if not channels:
            print("❌ No valid channels found")
            return
        
        # Create catalog
        catalog = loader.create_channel_catalog(channels)
        
        # Save catalog
        catalog_path = loader.data_dir / "nasa_channel_catalog.csv"
        catalog.to_csv(catalog_path, index=False)
        
        print(f"✅ Found {len(channels)} spacecraft telemetry channels")
        print(f"📋 Catalog saved to {catalog_path}")
        
        # Show top channels
        print("\n🏆 Top 5 Recommended Channels:")
        for _, row in catalog.head().iterrows():
            print(f"  • {row['name']} ({row['type']}) - {row['data_points']} points - {row['spacecraft']}")
        
        return catalog
        
    except Exception as e:
        print(f"❌ Error loading NASA data: {e}")
        return None

if __name__ == "__main__":
    load_real_nasa_data()