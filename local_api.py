#!/usr/bin/env python3
"""
Local development API server for space data
This mimics the Vercel serverless functions for local testing
"""

import json
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time
import math
import random
import datetime

class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/space-data':
            self.handle_space_data()
        elif path == '/api/channels':
            self.handle_channels()
        elif path.startswith('/api/timeseries'):
            self.handle_timeseries(parsed_path)
        elif path == '/api/anomalies':
            self.handle_anomalies()
        elif path == '/api/run-detection':
            self.handle_run_detection()
        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def handle_space_data(self):
        try:
            # Fetch live ISS position
            iss_data = self.fetch_iss_data()
            
            # Fetch astronaut count
            astros_data = self.fetch_astros_data()
            
            # Fetch SpaceX data
            spacex_data = self.fetch_spacex_data()
            
            # Mars rover data (using demo data since NASA API requires key)
            mars_data = {
                "sol": 950,
                "earth_date": "2024-09-15",
                "status": "active",
                "mission_duration_days": 1317,
                "sample_count": 24,
                "total_photos": 15,
                "latest_photo_date": "2024-09-15"
            }
            
            response_data = {
                "status": "success",
                "timestamp": int(time.time()),
                "iss": {
                    "latitude": float(iss_data.get("latitude", 25.5)),
                    "longitude": float(iss_data.get("longitude", -80.3)),
                    "altitude_km": 408,
                    "velocity_kmh": 27600,
                    "crew_count": astros_data.get("number", 7),
                    "last_updated": int(time.time()),
                    "position": {
                        "latitude": float(iss_data.get("latitude", 25.5)),
                        "longitude": float(iss_data.get("longitude", -80.3))
                    }
                },
                "spacex": {
                    "mission_name": spacex_data.get("name", "Starship IFT-5"),
                    "flight_number": spacex_data.get("flight_number", 200),
                    "launch_date": spacex_data.get("date_utc", "2024-09-01T12:00:00.000Z"),
                    "success": spacex_data.get("success", True),
                    "cores_recovered": 2,
                    "rocket": spacex_data.get("rocket", "Falcon Heavy")
                },
                "mars": mars_data
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            response_json = json.dumps(response_data)
            self.wfile.write(response_json.encode('utf-8'))
            
        except Exception as e:
            print(f"Error in space data API: {e}")
            # Return fallback data
            fallback_data = {
                "status": "fallback",
                "error": str(e),
                "iss": {
                    "latitude": 25.5,
                    "longitude": -80.3,
                    "altitude_km": 408,
                    "velocity_kmh": 27600,
                    "crew_count": 7,
                    "last_updated": int(time.time()),
                    "position": {"latitude": 25.5, "longitude": -80.3}
                },
                "spacex": {
                    "mission_name": "Starship IFT-5",
                    "flight_number": 200,
                    "launch_date": "2024-09-01T12:00:00.000Z",
                    "success": True,
                    "cores_recovered": 2,
                    "rocket": "Falcon Heavy"
                },
                "mars": {
                    "sol": 950,
                    "earth_date": "2024-09-15",
                    "status": "active",
                    "mission_duration_days": 1317,
                    "sample_count": 24,
                    "total_photos": 15
                }
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            response_json = json.dumps(fallback_data)
            self.wfile.write(response_json.encode('utf-8'))
    
    def fetch_iss_data(self):
        try:
            with urllib.request.urlopen('http://api.open-notify.org/iss-now.json', timeout=5) as response:
                data = json.loads(response.read().decode())
                return data.get('iss_position', {})
        except:
            return {"latitude": 25.5, "longitude": -80.3}
    
    def fetch_astros_data(self):
        try:
            with urllib.request.urlopen('http://api.open-notify.org/astros.json', timeout=5) as response:
                data = json.loads(response.read().decode())
                return data
        except:
            return {"number": 7}
    
    def fetch_spacex_data(self):
        try:
            with urllib.request.urlopen('https://api.spacexdata.com/v4/launches/latest', timeout=5) as response:
                data = json.loads(response.read().decode())
                return data
        except:
            return {
                "name": "Starship IFT-5",
                "flight_number": 200,
                "date_utc": "2024-09-01T12:00:00.000Z",
                "success": True,
                "rocket": "Falcon Heavy"
            }
    
    def handle_channels(self):
        channels = [
            {"name": "demo_channel_1", "description": "Temperature Sensor", "source": "Satellite Alpha"},
            {"name": "demo_channel_2", "description": "Pressure Sensor", "source": "Satellite Beta"},
            {"name": "demo_channel_3", "description": "Voltage Monitor", "source": "Satellite Gamma"}
        ]
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(channels).encode())
    
    def handle_timeseries(self, parsed_path):
        # Generate demo timeseries data
        now = datetime.datetime.now()
        data = []
        
        for i in range(100):
            timestamp = now - datetime.timedelta(minutes=i)
            value = 50 + random.gauss(0, 10) + 5 * math.sin(i * 0.1)
            if random.random() < 0.05:  # 5% chance of anomaly
                value += random.choice([-30, 30])
            
            data.append({
                "timestamp": timestamp.isoformat(),
                "value": round(value, 2),
                "anomaly": abs(value - 50) > 25
            })
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps({"data": list(reversed(data))}).encode())
    
    def handle_anomalies(self):
        anomalies = [
            {
                "timestamp": "2024-09-26T20:30:00",
                "channel": "demo_channel_1",
                "value": 85.2,
                "severity": "high",
                "type": "spike"
            },
            {
                "timestamp": "2024-09-26T19:45:00",
                "channel": "demo_channel_2", 
                "value": 12.1,
                "severity": "medium",
                "type": "drop"
            }
        ]
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps({"anomalies": anomalies}).encode())
    
    def handle_run_detection(self):
        result = {
            "status": "success",
            "algorithm": "isolation_forest",
            "anomalies_detected": 3,
            "processing_time": 1.23,
            "timestamp": int(time.time())
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

def run_server():
    server = HTTPServer(('localhost', 3001), APIHandler)
    print("🚀 Local API server running on http://localhost:3001")
    print("📡 Space data endpoint: http://localhost:3001/api/space-data")
    server.serve_forever()

if __name__ == '__main__':
    run_server()