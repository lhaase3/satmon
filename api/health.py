from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "status": "healthy", 
            "service": "SatMon", 
            "deployment": "Vercel", 
            "version": "1.0.0",
            "timestamp": "2025-09-25T00:00:00Z"
        }
        self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))