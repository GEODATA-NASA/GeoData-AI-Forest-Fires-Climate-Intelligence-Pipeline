import logging
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RAW_DATA_DIR, WEATHER_FILE

LOGGER = logging.getLogger("geodata.integration.nasa_power")


def download_weather_data():
    """Download NASA POWER weather data unless the configured raw file exists."""
    if os.path.isfile(WEATHER_FILE):
        if os.path.getsize(WEATHER_FILE) == 0:
            raise ValueError(f"Existing NASA POWER file is empty: {WEATHER_FILE}")
        LOGGER.info("NASA POWER weather data is already downloaded: %s", WEATHER_FILE)
        return WEATHER_FILE

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Region coordinates and start/end dates for the daily NASA POWER request.
    latitude = 30.0
    longitude = 31.0
    start_date = "20250101"
    end_date = "20250131"
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=RH2M,PRECTOTCORR,GWETTOP&community=AG"
        f"&longitude={longitude}&latitude={latitude}"
        f"&start={start_date}&end={end_date}&format=JSON"
    )

    LOGGER.info("Fetching climate and rainfall data from NASA POWER.")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    properties = data["properties"]["parameter"]

    df = pd.DataFrame(properties)
    df.index.name = "date"
    df = df.reset_index()
    df["latitude"] = latitude
    df["longitude"] = longitude

    df.to_csv(WEATHER_FILE, index=False)
    LOGGER.info("NASA POWER weather data saved to %s", WEATHER_FILE)
    return WEATHER_FILE


if __name__ == "__main__":
    download_weather_data()