from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        channels = [
            {"id": 1, "name": "iss_position", "description": "ISS Real-Time Position", "unit": "Lat/Lon", "source": "NASA ISS API"},
            {"id": 2, "name": "iss_altitude", "description": "ISS Orbital Altitude", "unit": "Kilometers", "source": "NASA ISS API"},
            {"id": 3, "name": "spacex_launches", "description": "SpaceX Launch Data", "unit": "Count", "source": "SpaceX API"},
            {"id": 4, "name": "satellite_tle", "description": "Satellite Orbital Elements", "unit": "TLE", "source": "CelesTrak"},
            {"id": 5, "name": "demo_temp_sensor", "description": "Synthetic Temperature Data", "unit": "Celsius", "source": "Demo System"}
        ]
        self.wfile.write(json.dumps(channels, indent=2).encode('utf-8'))