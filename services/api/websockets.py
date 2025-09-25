#!/usr/bin/env python3
"""
WebSocket endpoint for real-time telemetry streaming.
Simulates live spacecraft telemetry for demo purposes.
"""

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed, remove it
                self.active_connections.remove(connection)

manager = ConnectionManager()

def generate_live_telemetry_point(channel_type: str = "P", base_time: datetime = None):
    """Generate a single realistic telemetry point."""
    if base_time is None:
        base_time = datetime.now(timezone.utc)
    
    # Base patterns by channel type
    patterns = {
        'P': {'mean': 0.0, 'std': 0.1, 'range': [-0.8, 0.8]},  # Power
        'S': {'mean': 0.2, 'std': 0.08, 'range': [-0.6, 1.0]}, # Solar
        'T': {'mean': 0.1, 'std': 0.06, 'range': [-0.5, 0.7]}, # Thermal
        'E': {'mean': -0.1, 'std': 0.12, 'range': [-1.0, 0.8]} # Electrical
    }
    
    pattern = patterns.get(channel_type, patterns['P'])
    
    # Add some time-based variation (simulating orbital cycles)
    time_factor = np.sin(2 * np.pi * base_time.minute / 60.0) * 0.1
    
    # Generate value with occasional anomalies (5% chance)
    if np.random.random() < 0.05:  # 5% anomaly chance
        # Generate anomaly
        anomaly_magnitude = np.random.uniform(0.3, 0.7) * np.random.choice([-1, 1])
        value = pattern['mean'] + anomaly_magnitude + time_factor
    else:
        # Normal value
        value = np.random.normal(pattern['mean'] + time_factor, pattern['std'])
    
    # Clip to realistic range
    value = np.clip(value, pattern['range'][0], pattern['range'][1])
    
    return {
        'timestamp': base_time.isoformat(),
        'value': float(value),
        'channel_type': channel_type
    }

async def websocket_endpoint(websocket: WebSocket, channel_id: int = 1):
    """WebSocket endpoint for real-time telemetry streaming."""
    await manager.connect(websocket)
    
    # Get channel info to determine type
    channel_type = "P"  # Default to Power, could lookup from DB
    
    try:
        while True:
            # Generate new telemetry point
            point = generate_live_telemetry_point(channel_type)
            
            # Send to client
            await manager.send_personal_message(json.dumps(point), websocket)
            
            # Wait before next point (simulate real-time 1Hz data)
            await asyncio.sleep(1.0)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected for channel {channel_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)