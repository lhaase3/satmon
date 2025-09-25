import json
import random
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Enable CORS
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Parse query parameters
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            channel = query_params.get('channel', ['voltage'])[0]
            limit = int(query_params.get('limit', ['100'])[0])
            
            # Generate realistic telemetry data
            data = []
            base_time = datetime.now() - timedelta(hours=2)
            
            # Define channel characteristics
            channel_configs = {
                'voltage': {'base': 12.0, 'variance': 0.5, 'unit': 'V'},
                'temperature': {'base': 45.0, 'variance': 8.0, 'unit': '°C'},
                'pressure': {'base': 101.3, 'variance': 2.5, 'unit': 'kPa'},
                'current': {'base': 2.5, 'variance': 0.3, 'unit': 'A'},
                'fuel_level': {'base': 75.0, 'variance': 1.0, 'unit': '%'},
                'battery_temp': {'base': 25.0, 'variance': 5.0, 'unit': '°C'},
                'solar_power': {'base': 150.0, 'variance': 20.0, 'unit': 'W'},
                'attitude_x': {'base': 0.0, 'variance': 0.1, 'unit': 'deg'},
                'attitude_y': {'base': 0.0, 'variance': 0.1, 'unit': 'deg'},
                'attitude_z': {'base': 0.0, 'variance': 0.1, 'unit': 'deg'},
                'satellite_temp': {'base': -10.0, 'variance': 15.0, 'unit': '°C'},
                'signal_strength': {'base': -85.0, 'variance': 5.0, 'unit': 'dBm'}
            }
            
            config = channel_configs.get(channel, channel_configs['voltage'])
            
            # Generate time series data
            for i in range(limit):
                timestamp = base_time + timedelta(minutes=i * 2)
                
                # Add some realistic trends and noise
                trend = 0
                if channel == 'fuel_level':
                    trend = -i * 0.02  # Fuel decreases over time
                elif channel == 'battery_temp':
                    trend = random.uniform(-0.1, 0.1) * i  # Temperature drift
                
                # Add occasional anomalies (5% chance)
                anomaly_factor = 1.0
                if random.random() < 0.05:
                    anomaly_factor = random.uniform(0.3, 3.0)
                
                value = (config['base'] + trend + 
                        random.gauss(0, config['variance']) * anomaly_factor)
                
                # Ensure realistic bounds
                if channel == 'fuel_level':
                    value = max(0, min(100, value))
                elif channel in ['temperature', 'battery_temp', 'satellite_temp']:
                    value = max(-50, min(150, value))
                
                data.append({
                    'timestamp': timestamp.isoformat(),
                    'value': round(value, 3),
                    'channel': channel,
                    'unit': config['unit']
                })
            
            response_data = {
                'data': data,
                'channel': channel,
                'count': len(data),
                'generated_at': datetime.now().isoformat()
            }
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            error_response = {
                'error': f'Failed to generate timeseries data: {str(e)}',
                'fallback_data': [{
                    'timestamp': datetime.now().isoformat(),
                    'value': 12.0,
                    'channel': 'voltage',
                    'unit': 'V'
                }]
            }
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()