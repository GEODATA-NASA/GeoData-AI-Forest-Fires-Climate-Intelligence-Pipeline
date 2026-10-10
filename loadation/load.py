import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DB_CONFIG, CLEANED_DATA_DIR


def normalize_fire_chunk(chunk):
    chunk = chunk.copy()

    if "confidence" in chunk.columns:
        chunk["confidence"] = (
            chunk["confidence"].astype(str).str.strip().str.lower().replace({"h": 100, "n": 50, "l": 25})
            .apply(lambda value: pd.to_numeric(value, errors="coerce"))
        )
        chunk["confidence"] = pd.to_numeric(chunk["confidence"], errors="coerce").astype("Int64")

    for column in ["type", "acq_time"]:
        if column in chunk.columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Int64")

    return chunk


def get_connection_url():
    # Configure the PostgreSQL connection using variables imported from config.py
    return f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

def load_data_to_postgres():
    print("Connecting to the PostgreSQL database...")
    engine = create_engine(get_connection_url())
    try:
        with engine.connect():
            pass
    except OperationalError as e:
        print(f"Could not connect to database '{DB_CONFIG['dbname']}': {e.orig}")
        print(
            "Make sure PostgreSQL is running, the database exists, and the "
            "connection settings in .env are correct."
        )
        engine.dispose()
        return

    # 1. Load the cleaned fire data using CLEANED_DATA_DIR from config
    fires_file = os.path.join(CLEANED_DATA_DIR, "cleaned_master_fires.csv")
    if os.path.exists(fires_file):
        print("Uploading the fire table (fire_events) to the database in chunks...")
        
        chunk_size = 50000
        first_chunk = True
        chunk_count = 0
        
        for chunk in pd.read_csv(fires_file, chunksize=chunk_size, low_memory=False):
            chunk = normalize_fire_chunk(chunk)
            mode = "replace" if first_chunk else "append"
            chunk.to_sql("fire_events", engine, if_exists=mode, index=False)
            first_chunk = False
            chunk_count += 1
            print(f"{chunk_count}: Uploaded a chunk of {chunk_size} rows")
            
        print("Fire table uploaded successfully!")
    else:
        print(f"Clean fire file not found at path: {fires_file}")

    # 2. Load the cleaned weather data
    weather_file = os.path.join(CLEANED_DATA_DIR, "cleaned_weather.csv")
    if os.path.exists(weather_file):
        print("Uploading the weather table (weather_data) to the database...")
        df_weather = pd.read_csv(weather_file)
        df_weather.to_sql("weather_data", engine, if_exists="replace", index=False)
        print("Weather table uploaded successfully!")
    else:
        print(f"Clean weather file not found at path: {weather_file}")

    print("The Load stage completed successfully, and the data is ready inside the database!")

if __name__ == "__main__":
    load_data_to_postgres()