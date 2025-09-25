#!/usr/bin/env python3
"""
Vercel-compatible entry point for SatMon
"""

from demo_server import app

# Vercel serverless handler
def handler(request, response):
    return app(request, response)

# For Vercel, we export the app
application = app

if __name__ == "__main__":
    # This runs locally, not on Vercel
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting SatMon Demo Server...")
    print(f"📊 Dashboard: http://localhost:{port}")
    print(f"🔧 API Docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)