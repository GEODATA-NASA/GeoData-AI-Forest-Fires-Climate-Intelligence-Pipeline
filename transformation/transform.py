import os
import sys
from pathlib import Path
import tempfile

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CLEANED_DATA_DIR, FIRE_FILES, WEATHER_FILE
from utils.quality_checks import validate_fire_data

FIRE_CHUNK_SIZE = 100_000

FLAG_FILE_PATH = Path(CLEANED_DATA_DIR) / ".fire_validated.flag"


def _validate_existing_fire_file(file_path, collect_data):
    """Validate a cleaned fire CSV only if it hasn't been validated before."""
    
    # Check if the flag file exists and is newer than the cleaned fire data file
    if FLAG_FILE_PATH.is_file() and FLAG_FILE_PATH.stat().st_mtime >= Path(file_path).stat().st_mtime:
        print(f"Skipping quality checks: Cleaned fire data is already validated and unchanged: {file_path}")
        if collect_data:
            pass
        return None

    print("Running initial quality checks on existing cleaned fire file...")
    cleaned_frames = []
    input_rows = 0
    valid_rows = 0
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            dir=Path(file_path).parent,
            delete=False,
            encoding="utf-8",
            newline="",
        ) as temp_file:
            temp_path = temp_file.name
            wrote_rows = False
            for chunk in pd.read_csv(
                file_path, chunksize=FIRE_CHUNK_SIZE, low_memory=False
            ):
                input_rows += len(chunk)
                validated = validate_fire_data(chunk, raise_if_empty=False)
                valid_rows += len(validated)
                if collect_data and not validated.empty:
                    cleaned_frames.append(validated)
                if not validated.empty:
                    validated.to_csv(
                        temp_file,
                        index=False,
                        header=not wrote_rows,
                    )
                    wrote_rows = True

        if input_rows == 0 or valid_rows == 0:
            raise ValueError(f"Cleaned fire data contains no valid rows: {file_path}")

        if valid_rows < input_rows:
            if not wrote_rows:
                raise ValueError(f"Cleaned fire data contains no valid rows: {file_path}")
            os.replace(temp_path, file_path)
            temp_path = None
        else:
            os.remove(temp_path)
            temp_path = None
    except Exception:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    # Create the flag file to indicate that the fire data has been validated
    FLAG_FILE_PATH.touch()

    print(
        f"Existing cleaned fire data validated: {valid_rows} valid rows "
        f"({input_rows - valid_rows} removed)."
    )
    if cleaned_frames:
        return pd.concat(cleaned_frames, ignore_index=True)
    return None


def normalize_fire_dataframe(df, source_name):
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    df["satellite_source"] = source_name

    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude", "longitude"])

    if "confidence" in df.columns:
        df["confidence"] = (
            df["confidence"].astype(str).str.strip().str.lower().replace({"h": 100, "n": 50, "l": 25})
            .apply(lambda value: pd.to_numeric(value, errors="coerce"))
        )
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").astype("Int64")

    for column in ["type", "acq_time"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    return df


def transform_fire_data(collect_data=True):
    output_path = os.path.join(CLEANED_DATA_DIR, "cleaned_master_fires.csv")

    if os.path.isfile(output_path):
        print(f"Using existing cleaned fire data: {output_path}")
        return _validate_existing_fire_file(output_path, collect_data)

    print("Processing and cleaning NASA fire data...")
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    cleaned_frames = []
    output_written = False

    for source_name, file_path in FIRE_FILES.items():
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        for df in pd.read_csv(file_path, chunksize=FIRE_CHUNK_SIZE):
            df = normalize_fire_dataframe(df, source_name)
            df = validate_fire_data(df, raise_if_empty=False)
            if df.empty:
                continue

            df.to_csv(output_path, mode="w" if not output_written else "a",
                      header=not output_written, index=False)
            output_written = True
            if collect_data:
                cleaned_frames.append(df)

        print(f"Data for {source_name} cleaned successfully.")

    if output_written:
        print(f"Cleaned fire data saved to: {output_path}")
        # Create the flag file to indicate that the fire data has been validated
        FLAG_FILE_PATH.touch()
    else:
        raise ValueError("No valid fire rows were found in the configured raw files.")

    if cleaned_frames:
        return pd.concat(cleaned_frames, ignore_index=True)
    return None


def transform_weather_data():
    output_path = os.path.join(CLEANED_DATA_DIR, "cleaned_weather.csv")
    if os.path.isfile(output_path):
        print(f"Using existing cleaned weather data: {output_path}")
        weather_df = pd.read_csv(output_path)
        if weather_df.empty:
            raise ValueError(f"Cleaned weather data contains no rows: {output_path}")
        return weather_df

    print("Processing and cleaning weather and climate data (NASA POWER)...")
    if not os.path.exists(WEATHER_FILE):
        raise FileNotFoundError(f"Weather file not found at: {WEATHER_FILE}")

    weather_df = pd.read_csv(WEATHER_FILE)

    # Clean weather data (remove missing values and sort dates)
    weather_df = weather_df.dropna()
    if weather_df.empty:
        raise ValueError(f"NASA POWER weather data contains no usable rows: {WEATHER_FILE}")

    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    weather_df.to_csv(output_path, index=False)
    print(f"Weather data cleaned and saved to: {output_path}")
    return weather_df


if __name__ == "__main__":
    transform_fire_data(collect_data=False)
    transform_weather_data()
    print("The Transform stage completed successfully and the data is ready for Load!")