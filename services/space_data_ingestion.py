"""
NASA/SpaceX Data Integration Module
Fetches real satellite telemetry data from public APIs
"""

import requests
import json
from datetime import datetime, timedelta
import pandas as pd

class SpaceDataIngestion:
    """Ingest real satellite data from NASA and SpaceX APIs"""
    
    def __init__(self):
        self.nasa_api_key = "DEMO_KEY"  # Replace with real API key
        self.spacex_api = "https://api.spacexdata.com/v4"
        self.nasa_api = "https://api.nasa.gov"
    
    async def fetch_iss_telemetry(self):
        """Fetch ISS current position and telemetry"""
        try:
            # ISS Current Location API
            response = requests.get("http://api.open-notify.org/iss-now.json")
            iss_data = response.json()
            
            # ISS Astronauts API
            astronauts_response = requests.get("http://api.open-notify.org/astros.json")
            astronauts_data = astronauts_response.json()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "latitude": float(iss_data["iss_position"]["latitude"]),
                "longitude": float(iss_data["iss_position"]["longitude"]),
                "altitude": 408000,  # ISS average altitude in meters
                "crew_count": len([p for p in astronauts_data["people"] if p["craft"] == "ISS"]),
                "orbital_velocity": 7.66,  # km/s
                "source": "ISS_Real_Time"
            }
        except Exception as e:
            print(f"Error fetching ISS data: {e}")
            return None
    
    async def fetch_spacex_launches(self):
        """Fetch recent SpaceX launch telemetry"""
        try:
            response = requests.get(f"{self.spacex_api}/launches/latest")
            launch_data = response.json()
            
            return {
                "mission_name": launch_data["name"],
                "flight_number": launch_data["flight_number"],
                "launch_date": launch_data["date_utc"],
                "success": launch_data["success"],
                "rocket": launch_data["rocket"],
                "cores_recovery": len([c for c in launch_data.get("cores", []) if c.get("landing_success")]),
                "source": "SpaceX_API"
            }
        except Exception as e:
            print(f"Error fetching SpaceX data: {e}")
            return None
    
    async def fetch_satellite_tle_data(self):
        """Fetch Two-Line Element (TLE) data for satellites"""
        try:
            # Using CelesTrak API for satellite orbital data
            response = requests.get("https://celestrak.com/NORAD/elements/stations.txt")
            tle_data = response.text.strip().split('\n')
            
            satellites = []
            for i in range(0, len(tle_data), 3):
                if i + 2 < len(tle_data):
                    satellites.append({
                        "name": tle_data[i].strip(),
                        "line1": tle_data[i + 1],
                        "line2": tle_data[i + 2],
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            return satellites[:10]  # Return first 10 satellites
        except Exception as e:
            print(f"Error fetching TLE data: {e}")
            return []

# Usage in your API endpoints
async def get_real_space_data():
    """Endpoint to serve real space data"""
    ingestion = SpaceDataIngestion()
    
    data = {
        "iss": await ingestion.fetch_iss_telemetry(),
        "spacex": await ingestion.fetch_spacex_launches(),
        "satellites": await ingestion.fetch_satellite_tle_data()
    }
    
    return data