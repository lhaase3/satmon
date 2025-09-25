"""
PostgreSQL Time-Series Database Integration
High-performance telemetry data storage with TimescaleDB
"""

import asyncpg
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import numpy as np

class SatelliteTelemetryDB:
    """PostgreSQL/TimescaleDB integration for satellite telemetry"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize_db(self):
        """Initialize database connection pool and create tables"""
        self.pool = await asyncpg.create_pool(self.connection_string)
        
        async with self.pool.acquire() as conn:
            # Create main telemetry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    spacecraft_id VARCHAR(50) NOT NULL,
                    channel_name VARCHAR(100) NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    unit VARCHAR(20),
                    subsystem VARCHAR(50),
                    quality_flag INTEGER DEFAULT 1,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for time-series optimization (TimescaleDB)
            try:
                await conn.execute("""
                    SELECT create_hypertable('telemetry_data', 'timestamp', 
                                           if_not_exists => TRUE);
                """)
            except Exception as e:
                print(f"Note: TimescaleDB extension not available: {e}")
            
            # Create anomaly detection results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    spacecraft_id VARCHAR(50) NOT NULL,
                    channel_name VARCHAR(100) NOT NULL,
                    anomaly_score DOUBLE PRECISION NOT NULL,
                    confidence DOUBLE PRECISION,
                    detection_method VARCHAR(50),
                    severity VARCHAR(20),
                    description TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMPTZ,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create spacecraft configuration table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS spacecraft_config (
                    id SERIAL PRIMARY KEY,
                    spacecraft_id VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    mission_type VARCHAR(50),
                    launch_date DATE,
                    status VARCHAR(20) DEFAULT 'ACTIVE',
                    orbital_parameters JSONB,
                    telemetry_channels JSONB,
                    thresholds JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp 
                ON telemetry_data (timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_spacecraft_channel 
                ON telemetry_data (spacecraft_id, channel_name, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp 
                ON anomaly_detections (timestamp DESC);
            """)
    
    async def insert_telemetry_batch(self, telemetry_data: List[Dict]) -> int:
        """Insert batch of telemetry data efficiently"""
        if not telemetry_data:
            return 0
        
        async with self.pool.acquire() as conn:
            # Prepare data for batch insert
            values = []
            for data in telemetry_data:
                values.append((
                    data['timestamp'],
                    data['spacecraft_id'],
                    data['channel_name'],
                    data['value'],
                    data.get('unit'),
                    data.get('subsystem'),
                    data.get('quality_flag', 1),
                    json.dumps(data.get('metadata', {}))
                ))
            
            # Batch insert
            result = await conn.executemany("""
                INSERT INTO telemetry_data 
                (timestamp, spacecraft_id, channel_name, value, unit, subsystem, quality_flag, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, values)
            
            return len(values)
    
    async def insert_anomaly(self, anomaly_data: Dict) -> int:
        """Insert anomaly detection result"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                INSERT INTO anomaly_detections 
                (timestamp, spacecraft_id, channel_name, anomaly_score, confidence, 
                 detection_method, severity, description, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """, 
                anomaly_data['timestamp'],
                anomaly_data['spacecraft_id'],
                anomaly_data['channel_name'],
                anomaly_data['anomaly_score'],
                anomaly_data.get('confidence'),
                anomaly_data.get('detection_method'),
                anomaly_data.get('severity'),
                anomaly_data.get('description'),
                json.dumps(anomaly_data.get('metadata', {}))
            )
            return result
    
    async def get_telemetry_range(self, spacecraft_id: str, channel_name: str, 
                                 start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get telemetry data for specific time range"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT timestamp, value, unit, quality_flag, metadata
                FROM telemetry_data
                WHERE spacecraft_id = $1 AND channel_name = $2
                AND timestamp BETWEEN $3 AND $4
                ORDER BY timestamp ASC
            """, spacecraft_id, channel_name, start_time, end_time)
            
            return [dict(row) for row in rows]
    
    async def get_recent_anomalies(self, spacecraft_id: str, hours: int = 24) -> List[Dict]:
        """Get recent anomalies for a spacecraft"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT timestamp, channel_name, anomaly_score, confidence,
                       detection_method, severity, description, resolved
                FROM anomaly_detections
                WHERE spacecraft_id = $1 AND timestamp >= $2
                ORDER BY timestamp DESC
            """, spacecraft_id, cutoff_time)
            
            return [dict(row) for row in rows]
    
    async def get_telemetry_statistics(self, spacecraft_id: str, 
                                     time_window: timedelta = timedelta(hours=1)) -> Dict:
        """Get telemetry statistics for monitoring dashboard"""
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        async with self.pool.acquire() as conn:
            # Data points per channel
            channel_stats = await conn.fetch("""
                SELECT channel_name, 
                       COUNT(*) as data_points,
                       AVG(value) as avg_value,
                       MIN(value) as min_value,
                       MAX(value) as max_value,
                       STDDEV(value) as std_value
                FROM telemetry_data
                WHERE spacecraft_id = $1 AND timestamp BETWEEN $2 AND $3
                GROUP BY channel_name
                ORDER BY data_points DESC
            """, spacecraft_id, start_time, end_time)
            
            # Anomaly summary
            anomaly_stats = await conn.fetchrow("""
                SELECT COUNT(*) as total_anomalies,
                       COUNT(*) FILTER (WHERE severity = 'HIGH') as high_severity,
                       COUNT(*) FILTER (WHERE severity = 'MEDIUM') as medium_severity,
                       COUNT(*) FILTER (WHERE resolved = false) as unresolved
                FROM anomaly_detections
                WHERE spacecraft_id = $1 AND timestamp BETWEEN $2 AND $3
            """, spacecraft_id, start_time, end_time)
            
            # Data quality metrics
            quality_stats = await conn.fetchrow("""
                SELECT COUNT(*) as total_points,
                       COUNT(*) FILTER (WHERE quality_flag = 1) as good_quality,
                       COUNT(*) FILTER (WHERE quality_flag = 0) as poor_quality,
                       (COUNT(*) FILTER (WHERE quality_flag = 1) * 100.0 / COUNT(*)) as quality_percentage
                FROM telemetry_data
                WHERE spacecraft_id = $1 AND timestamp BETWEEN $2 AND $3
            """, spacecraft_id, start_time, end_time)
            
            return {
                'time_window': str(time_window),
                'channels': [dict(row) for row in channel_stats],
                'anomalies': dict(anomaly_stats) if anomaly_stats else {},
                'data_quality': dict(quality_stats) if quality_stats else {}
            }
    
    async def create_spacecraft(self, spacecraft_config: Dict) -> int:
        """Create new spacecraft configuration"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                INSERT INTO spacecraft_config 
                (spacecraft_id, name, mission_type, launch_date, orbital_parameters, 
                 telemetry_channels, thresholds)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """,
                spacecraft_config['spacecraft_id'],
                spacecraft_config['name'],
                spacecraft_config.get('mission_type'),
                spacecraft_config.get('launch_date'),
                json.dumps(spacecraft_config.get('orbital_parameters', {})),
                json.dumps(spacecraft_config.get('telemetry_channels', {})),
                json.dumps(spacecraft_config.get('thresholds', {}))
            )
            return result
    
    async def get_spacecraft_list(self) -> List[Dict]:
        """Get list of all configured spacecraft"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT spacecraft_id, name, mission_type, launch_date, status,
                       (SELECT COUNT(*) FROM telemetry_data t 
                        WHERE t.spacecraft_id = sc.spacecraft_id 
                        AND t.timestamp >= NOW() - INTERVAL '24 hours') as recent_data_points,
                       (SELECT COUNT(*) FROM anomaly_detections a 
                        WHERE a.spacecraft_id = sc.spacecraft_id 
                        AND a.timestamp >= NOW() - INTERVAL '24 hours'
                        AND a.resolved = false) as active_anomalies
                FROM spacecraft_config sc
                ORDER BY name
            """)
            
            return [dict(row) for row in rows]

# Data pipeline for continuous ingestion
class TelemetryDataPipeline:
    """Continuous data pipeline for satellite telemetry"""
    
    def __init__(self, db: SatelliteTelemetryDB):
        self.db = db
        self.batch_size = 1000
        self.buffer = []
    
    async def process_data_stream(self, data_stream):
        """Process continuous stream of telemetry data"""
        async for data_point in data_stream:
            self.buffer.append(data_point)
            
            if len(self.buffer) >= self.batch_size:
                await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush accumulated data to database"""
        if self.buffer:
            await self.db.insert_telemetry_batch(self.buffer)
            self.buffer.clear()
    
    async def generate_synthetic_data(self, spacecraft_id: str, duration_hours: int = 24):
        """Generate synthetic telemetry data for testing"""
        channels = [
            {'name': 'bus_temperature', 'unit': 'C', 'subsystem': 'thermal'},
            {'name': 'solar_voltage', 'unit': 'V', 'subsystem': 'power'},
            {'name': 'battery_current', 'unit': 'A', 'subsystem': 'power'},
            {'name': 'attitude_x', 'unit': 'deg', 'subsystem': 'adcs'},
            {'name': 'attitude_y', 'unit': 'deg', 'subsystem': 'adcs'},
            {'name': 'altitude', 'unit': 'km', 'subsystem': 'navigation'},
            {'name': 'velocity', 'unit': 'km/s', 'subsystem': 'navigation'},
        ]
        
        start_time = datetime.utcnow() - timedelta(hours=duration_hours)
        
        data_points = []
        for i in range(duration_hours * 3600):  # One point per second
            timestamp = start_time + timedelta(seconds=i)
            
            for channel in channels:
                # Generate realistic telemetry values with occasional anomalies
                base_value = self._get_channel_baseline(channel['name'])
                noise = np.random.normal(0, base_value * 0.05)  # 5% noise
                
                # Inject anomalies (1% chance)
                if np.random.random() < 0.01:
                    anomaly_multiplier = np.random.choice([0.5, 2.0])  # 50% drop or 200% spike
                    value = base_value * anomaly_multiplier + noise
                else:
                    value = base_value + noise
                
                data_points.append({
                    'timestamp': timestamp,
                    'spacecraft_id': spacecraft_id,
                    'channel_name': channel['name'],
                    'value': round(value, 3),
                    'unit': channel['unit'],
                    'subsystem': channel['subsystem'],
                    'quality_flag': 1 if abs(noise) < base_value * 0.1 else 0,
                    'metadata': {
                        'generated': True,
                        'baseline': base_value
                    }
                })
        
        # Insert in batches
        for i in range(0, len(data_points), self.batch_size):
            batch = data_points[i:i + self.batch_size]
            await self.db.insert_telemetry_batch(batch)
        
        return len(data_points)
    
    def _get_channel_baseline(self, channel_name: str) -> float:
        """Get baseline values for different telemetry channels"""
        baselines = {
            'bus_temperature': 25.0,
            'solar_voltage': 28.5,
            'battery_current': 2.3,
            'attitude_x': 0.0,
            'attitude_y': 0.0,
            'altitude': 408.0,
            'velocity': 7.66
        }
        return baselines.get(channel_name, 50.0)