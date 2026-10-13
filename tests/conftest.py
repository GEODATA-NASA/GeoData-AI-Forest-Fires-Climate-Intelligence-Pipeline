"""Shared fixtures: a tiny fire/weather dataset served through the CSV backend."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402

FIRE_ROWS = [
    # lat, lon, brightness, scan, track, date, time, sat, instr, conf, version, t31, frp, dn, type, source
    (30.10, 31.10, 330.0, 1.0, 1.0, "2025-01-02", 1030, "Terra", "MODIS", 90, 61.03, 290.0, 45.0, "D", 0, "modis"),
    (30.20, 31.30, 305.0, 1.0, 1.0, "2025-01-02", 2230, "Aqua", "MODIS", 40, 61.03, 285.0, 4.0, "N", 0, "modis"),
    (30.00, 31.00, 340.0, 0.4, 0.4, "2025-01-05", 1100, "N20", "VIIRS", 100, 2, 295.0, 120.0, "D", 0, "viirs_j1"),
    (-33.9, 23.40, 318.0, 0.4, 0.4, "2025-03-15", 1300, "SNPP", "VIIRS", 50, 2, 288.0, 2.0, "D", 0, "viirs_suomi"),
    (45.00, -120.0, 360.0, 1.1, 1.0, "2025-03-15", 2000, "Terra", "MODIS", 95, 61.03, 300.0, 250.0, "N", 0, "modis"),
    (45.10, -120.1, 310.0, 1.1, 1.0, "2025-03-16", 2010, "Terra", "MODIS", 20, 61.03, 290.0, 1.5, "N", 2, "modis"),
]
FIRE_COLUMNS = [
    "latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time", "satellite",
    "instrument", "confidence", "version", "bright_t31", "frp", "daynight", "type", "satellite_source",
]
WEATHER_ROWS = [
    (20250101, 71.96, 0.01, 0.16, 30.0, 31.0),
    (20250102, 71.20, 0.03, 0.16, 30.0, 31.0),
    (20250105, 48.95, 0.00, 0.16, 30.0, 31.0),
]
WEATHER_COLUMNS = ["date", "RH2M", "PRECTOTCORR", "GWETTOP", "latitude", "longitude"]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("data")
    pd.DataFrame(FIRE_ROWS, columns=FIRE_COLUMNS).to_csv(directory / "fires.csv", index=False)
    pd.DataFrame(WEATHER_ROWS, columns=WEATHER_COLUMNS).to_csv(directory / "weather.csv", index=False)
    return directory


@pytest.fixture(scope="session")
def settings(data_dir, tmp_path_factory) -> Settings:
    return Settings(
        data_backend="csv",
        fires_csv=data_dir / "fires.csv",
        fires_sample_csv=data_dir / "missing_sample.csv",
        weather_csv=data_dir / "weather.csv",
        log_file=tmp_path_factory.mktemp("logs") / "api.log",
        serve_frontend=True,
    )


@pytest.fixture(scope="session")
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
