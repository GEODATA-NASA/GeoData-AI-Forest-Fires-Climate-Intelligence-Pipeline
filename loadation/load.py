import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, create_engine, inspect, text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DB_CONFIG, CLEANED_DATA_DIR

LOGGER = logging.getLogger("geodata.load")


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
    """Build the PostgreSQL connection URL from the project's configuration."""
    return URL.create(
        "postgresql+psycopg2",
        username=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=int(DB_CONFIG["port"]),
        database=DB_CONFIG["dbname"],
    )


def table_contains_rows(connection, table_name):
    """Return whether a table in the active schema exists and contains rows."""
    inspector = inspect(connection)
    if not inspector.has_table(table_name):
        return False

    preparer = connection.dialect.identifier_preparer
    quoted_table = preparer.quote(table_name)
    row_count = connection.execute(
        text(f"SELECT COUNT(*) FROM {quoted_table}")
    ).scalar_one()
    return row_count > 0


def load_data_to_postgres():
    """Upload cleaned datasets, skipping tables that already contain data."""
    engine = create_engine(get_connection_url(), pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            fire_is_loaded = table_contains_rows(connection, "fire_events")
            weather_is_loaded = table_contains_rows(connection, "weather_data")

        fires_file = os.path.join(CLEANED_DATA_DIR, "cleaned_master_fires.csv")
        if fire_is_loaded:
            LOGGER.info("Table fire_events already contains rows; skipping upload.")
        elif os.path.isfile(fires_file):
            LOGGER.info("Uploading cleaned fire data to fire_events.")
            first_chunk = True
            uploaded_rows = 0

            for chunk in pd.read_csv(fires_file, chunksize=50_000, low_memory=False):
                chunk = normalize_fire_chunk(chunk)
                chunk.to_sql(
                    "fire_events",
                    engine,
                    if_exists="replace" if first_chunk else "append",
                    index=False,
                )
                first_chunk = False
                uploaded_rows += len(chunk)
                LOGGER.info("Uploaded %s fire rows so far.", uploaded_rows)

            if first_chunk:
                raise ValueError(f"Cleaned fire file contains no rows: {fires_file}")
            LOGGER.info("Uploaded %s rows to fire_events.", uploaded_rows)
        else:
            raise FileNotFoundError(f"Cleaned fire file not found: {fires_file}")

        weather_file = os.path.join(CLEANED_DATA_DIR, "cleaned_weather.csv")
        if weather_is_loaded:
            LOGGER.info("Table weather_data already contains rows; skipping upload.")
        elif os.path.isfile(weather_file):
            LOGGER.info("Uploading cleaned weather data to weather_data.")
            weather_df = pd.read_csv(weather_file)
            if weather_df.empty:
                raise ValueError(f"Cleaned weather file contains no rows: {weather_file}")
            weather_df.to_sql("weather_data", engine, if_exists="replace", index=False)
            LOGGER.info("Uploaded %s rows to weather_data.", len(weather_df))
        else:
            raise FileNotFoundError(f"Cleaned weather file not found: {weather_file}")

        LOGGER.info("Load stage completed successfully.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_data_to_postgres()
