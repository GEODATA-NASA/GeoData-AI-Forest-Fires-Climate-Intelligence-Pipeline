import logging
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "geodata_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Data directories default to the repository layout but can be pointed
# elsewhere (for example a shared drive) through environment variables.
RAW_DATA_DIR = os.path.abspath(
    os.getenv("RAW_DATA_DIR", os.path.join(PROJECT_ROOT, "raw_data"))
)

CLEANED_DATA_DIR = os.path.abspath(
    os.getenv("CLEANED_DATA_DIR", os.path.join(PROJECT_ROOT, "cleaned_data"))
)

FIRE_FILES = {
    "modis": os.path.join(RAW_DATA_DIR, "fire_archive_M-C61_820251.csv"),
    "viirs_j1": os.path.join(RAW_DATA_DIR, "fire_archive_J1V-C2_820252.csv"),
    "viirs_suomi": os.path.join(RAW_DATA_DIR, "fire_archive_SV-C2_820253.csv")
}

WEATHER_FILE = os.path.join(RAW_DATA_DIR, "nasa_power_weather.csv")

os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "pipeline.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeoDataAI")