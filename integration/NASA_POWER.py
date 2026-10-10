import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RAW_DATA_DIR, WEATHER_FILE

os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Region coordinates and start/end dates (for example, latitude 30, longitude 31, year 2025)
lat = 30.0
lon = 31.0
start_date = "20250101"
end_date = "20250131"

# NASA POWER API endpoint for daily data requests (Community Agriculture/Climate)
url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=RH2M,PRECTOTCORR,GWETTOP&community=AG&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"

print("Fetching climate and rainfall data from NASA POWER...")
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    # Extract daily parameter values
    properties = data['properties']['parameter']

    # Convert the data to a DataFrame for viewing or storage
    df = pd.DataFrame(properties)
    print("Data retrieved successfully!")
    print(df.head())

    # Save the file into the configured data directory
    df.to_csv(WEATHER_FILE, index=False)
    print(f"Saved weather data to: {WEATHER_FILE}")
else:
    print(f"Download failed, error code: {response.status_code}")