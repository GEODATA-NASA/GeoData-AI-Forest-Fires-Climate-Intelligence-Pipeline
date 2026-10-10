import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "geodata_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "raw_data")

CLEANED_DATA_DIR = os.path.join(os.path.dirname(__file__), "cleaned_data")

FIRE_FILES = {
    "modis": os.path.join(RAW_DATA_DIR, "fire_archive_M-C61_820251.csv"),
    "viirs_j1": os.path.join(RAW_DATA_DIR, "fire_archive_J1V-C2_820252.csv"),
    "viirs_suomi": os.path.join(RAW_DATA_DIR, "fire_archive_SV-C2_820253.csv")
}

WEATHER_FILE = os.path.join(RAW_DATA_DIR, "nasa_power_weather.csv")