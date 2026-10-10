import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import RAW_DATA_DIR, CLEANED_DATA_DIR, FIRE_FILES, WEATHER_FILE

FIRE_CHUNK_SIZE = 100_000


def transform_fire_data(collect_data=True):
    print("Processing and cleaning NASA fire data...")

    cleaned_frames = []
    output_path = os.path.join(CLEANED_DATA_DIR, "cleaned_master_fires.csv")
    output_written = False

    for source_name, file_path in FIRE_FILES.items():
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        for df in pd.read_csv(file_path, chunksize=FIRE_CHUNK_SIZE):
            df.columns = [col.strip().lower() for col in df.columns]
            df["satellite_source"] = source_name

            if "acq_date" in df.columns:
                df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

            if "latitude" in df.columns and "longitude" in df.columns:
                df = df.dropna(subset=["latitude", "longitude"])

            df.to_csv(output_path, mode="w" if not output_written else "a",
                      header=not output_written, index=False)
            output_written = True
            if collect_data:
                cleaned_frames.append(df)

        print(f"Data for {source_name} cleaned successfully.")

    if output_written:
        print(f"Cleaned fire data saved to: {output_path}")

    if cleaned_frames:
        return pd.concat(cleaned_frames, ignore_index=True)
    return None


def transform_weather_data():
    print("Processing and cleaning weather and climate data (NASA POWER)...")

    if not os.path.exists(WEATHER_FILE):
        print(f"Weather file not found at: {WEATHER_FILE}")
        return None

    weather_df = pd.read_csv(WEATHER_FILE)

    # Clean weather data (remove missing values and sort dates)
    weather_df = weather_df.dropna()

    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(CLEANED_DATA_DIR, "cleaned_weather.csv")
    weather_df.to_csv(output_path, index=False)
    print(f"Weather data cleaned and saved to: {output_path}")
    return weather_df


if __name__ == "__main__":
    # Run transformation and cleaning operations
    transform_fire_data(collect_data=False)
    weather_df = transform_weather_data()
    print("The Transform stage completed successfully and the data is ready for Load!")