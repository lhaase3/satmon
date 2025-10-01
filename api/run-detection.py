import json
import random
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Enable CORS
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body_data = {}
            if content_length > 0:
                try:
                    body = self.rfile.read(content_length).decode('utf-8')
                    body_data = json.loads(body)
                except:
                    pass
            
            # Parse query parameters
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            # Get parameters from either body or query
            channel = body_data.get('channel') or query_params.get('channel', ['voltage'])[0]
            algorithm = body_data.get('algorithm') or query_params.get('algorithm', ['isolation_forest'])[0]
            
            # Simulate detection processing time
            processing_time = random.uniform(1.5, 3.5)
            time.sleep(min(processing_time, 2.0))  # Cap at 2 seconds for serverless
            
            # Generate detection results
            algorithms_info = {
                'isolation_forest': {
                    'name': 'Isolation Forest',
                    'description': 'Tree-based anomaly detection',
                    'typical_accuracy': 0.87
                },
                'z_score': {
                    'name': 'Z-Score Analysis', 
                    'description': 'Statistical threshold detection',
                    'typical_accuracy': 0.72
                },
                'lstm_autoencoder': {
                    'name': 'LSTM Autoencoder',
                    'description': 'Deep learning reconstruction error',
                    'typical_accuracy': 0.91
                }
            }
            
            algo_info = algorithms_info.get(algorithm, algorithms_info['isolation_forest'])
            
            # Generate realistic results
            accuracy = random.uniform(algo_info['typical_accuracy'] - 0.05, 
                                    algo_info['typical_accuracy'] + 0.08)
            precision = random.uniform(0.75, 0.95)
            recall = random.uniform(0.70, 0.90)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            anomalies_detected = random.randint(2, 12)
            total_datapoints = random.randint(450, 550)
            
            response_data = {
                'status': 'completed',
                'job_id': f'detection_{int(time.time())}',
                'channel': channel,
                'algorithm': {
                    'name': algo_info['name'],
                    'type': algorithm,
                    'description': algo_info['description']
                },
                'results': {
                    'anomalies_detected': anomalies_detected,
                    'total_datapoints': total_datapoints,
                    'anomaly_rate': round(anomalies_detected / total_datapoints * 100, 2),
                    'processing_time': round(processing_time, 2),
                    'model_performance': {
                        'accuracy': round(accuracy, 3),
                        'precision': round(precision, 3),
                        'recall': round(recall, 3),
                        'f1_score': round(f1_score, 3)
                    }
                },
                'execution_details': {
                    'started_at': datetime.now().isoformat(),
                    'completed_at': datetime.now().isoformat(),
                    'data_window': '2 hours',
                    'model_version': '2.1.0'
                },
                'next_steps': {
                    'view_anomalies': f'/api/anomalies?channel={channel}',
                    'download_report': f'/api/reports?job_id=detection_{int(time.time())}'
                }
            }
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            error_response = {
                'status': 'failed',
                'error': f'Detection failed: {str(e)}',
                'fallback_message': 'Please try again or contact support'
            }
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()