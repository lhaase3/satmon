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
            
            # Generate some realistic anomalies
            anomalies = []
            base_time = datetime.now() - timedelta(hours=1)
            
            # Generate 3-8 anomalies for demo purposes
            num_anomalies = random.randint(3, 8)
            
            # Define anomaly types and their characteristics
            anomaly_types = [
                {'type': 'spike', 'severity': 'HIGH', 'description': 'Sudden voltage spike detected'},
                {'type': 'drift', 'severity': 'MEDIUM', 'description': 'Gradual parameter drift'},
                {'type': 'dropout', 'severity': 'HIGH', 'description': 'Signal dropout detected'},
                {'type': 'noise', 'severity': 'LOW', 'description': 'Increased noise level'},
                {'type': 'oscillation', 'severity': 'MEDIUM', 'description': 'Abnormal oscillation pattern'}
            ]
            
            for i in range(num_anomalies):
                anomaly = random.choice(anomaly_types)
                timestamp = base_time + timedelta(minutes=random.randint(5, 60))
                
                # Generate confidence score based on severity
                confidence_ranges = {
                    'HIGH': (0.85, 0.98),
                    'MEDIUM': (0.65, 0.84),
                    'LOW': (0.50, 0.74)
                }
                conf_range = confidence_ranges[anomaly['severity']]
                confidence = random.uniform(conf_range[0], conf_range[1])
                
                anomalies.append({
                    'id': f'anomaly_{i+1}',
                    'timestamp': timestamp.isoformat(),
                    'channel': channel,
                    'type': anomaly['type'],
                    'severity': anomaly['severity'],
                    'confidence': round(confidence, 3),
                    'description': anomaly['description'],
                    'value': round(random.uniform(8.0, 18.0), 2),
                    'expected_range': [10.5, 13.5],
                    'algorithm': random.choice(['isolation_forest', 'z_score', 'lstm_autoencoder']),
                    'detected_at': datetime.now().isoformat()
                })
            
            # Sort by timestamp
            anomalies.sort(key=lambda x: x['timestamp'])
            
            response_data = {
                'anomalies': anomalies,
                'channel': channel,
                'total_count': len(anomalies),
                'detection_window': '1 hour',
                'analysis_completed_at': datetime.now().isoformat()
            }
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')  
            self.end_headers()
            
            error_response = {
                'error': f'Failed to retrieve anomalies: {str(e)}',
                'anomalies': [],
                'fallback': True
            }
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()