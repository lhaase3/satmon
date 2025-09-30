#!/usr/bin/env python3
"""
Simple local API server for testing
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.request
import time

class SimpleAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/space-data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Simple test data
            data = {
                "status": "success",
                "iss": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "altitude_km": 408,
                    "velocity_kmh": 27600,
                    "crew_count": 7,
                    "last_updated": int(time.time()),
                    "position": {
                        "latitude": 40.7128,
                        "longitude": -74.0060
                    }
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
            
            self.wfile.write(json.dumps(data).encode())
            
        elif self.path == '/api/channels':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            channels = [
                {"name": "demo_channel_1", "description": "Temperature Sensor"},
                {"name": "demo_channel_2", "description": "Pressure Sensor"}
            ]
            
            self.wfile.write(json.dumps(channels).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('localhost', 3001), SimpleAPIHandler)
    print("🚀 Simple API server running on http://localhost:3001")
    server.serve_forever()