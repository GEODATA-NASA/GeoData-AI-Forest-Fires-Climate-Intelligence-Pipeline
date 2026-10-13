"""Pipeline stages wrapped as task functions.

Each function runs one stage of ``main.py``, raises on failure (so the
orchestrator can retry it) and returns a small JSON-serialisable summary
(which Airflow stores as the task's XCom and shows in the UI). Nothing here
imports Airflow, so the functions are unit-testable and reusable from any
scheduler.
"""

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LOGGER = logging.getLogger("geodata.orchestration")


def _file_summary(path) -> dict:
    path = Path(path)
    return {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}


def download_weather() -> dict:
    from config import WEATHER_FILE
    from integration.NASA_POWER import MANIFEST_FILE, download_weather_data

    download_weather_data()
    summary = _file_summary(WEATHER_FILE)
    if MANIFEST_FILE.is_file():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        summary.update({key: manifest.get(key) for key in ("mode", "start", "end", "points", "tiles", "rows")})
    return summary


def transform_fires() -> dict:
    from config import CLEANED_DATA_DIR
    from transformation.transform import transform_fire_data

    transform_fire_data(collect_data=False)
    return _file_summary(Path(CLEANED_DATA_DIR) / "cleaned_master_fires.csv")


def transform_weather() -> dict:
    from config import CLEANED_DATA_DIR
    from transformation.transform import transform_weather_data

    frame = transform_weather_data()
    return {**_file_summary(Path(CLEANED_DATA_DIR) / "cleaned_weather.csv"), "rows": int(len(frame))}


def check_cleaned_outputs() -> dict:
    """Fail fast before touching the database if a transform produced nothing."""
    from config import CLEANED_DATA_DIR

    files = {
        "fires": Path(CLEANED_DATA_DIR) / "cleaned_master_fires.csv",
        "weather": Path(CLEANED_DATA_DIR) / "cleaned_weather.csv",
    }
    problems = [f"{name}: {path}" for name, path in files.items() if not path.is_file() or path.stat().st_size == 0]
    if problems:
        raise FileNotFoundError("Cleaned outputs are missing or empty: " + ", ".join(problems))
    return {name: _file_summary(path) for name, path in files.items()}


def load_database() -> dict:
    from loadation.load import load_data_to_postgres

    load_data_to_postgres()
    return {"loaded": True}


def train_model(force: bool | None = None) -> dict:
    """Train the classifier. ``force`` (or FORCE_RETRAIN=1) retrains over an existing artifact."""
    import modeling.train_model as training

    if force is not None:
        training.FORCE_RETRAIN = bool(force)
    status = training.main()
    if status != 0:
        raise RuntimeError(f"Model training failed with status {status}; see logs/pipeline.log.")
    import joblib

    artifact = joblib.load(training.MODEL_ARTIFACT_PATH)
    return {
        "artifact": str(training.MODEL_ARTIFACT_PATH),
        "metrics": artifact.get("metrics"),
        "training_rows": (artifact.get("training_extent") or {}).get("rows"),
    }


def refresh_api(api_url: str | None = None, timeout: float = 30.0) -> dict:
    """Tell a running API to rebuild its caches. Never fails the pipeline: the
    API may simply not be running, and its caches also expire on their own."""
    api_url = (api_url if api_url is not None else os.getenv("GEODATA_API_URL", "")).rstrip("/")
    if not api_url:
        LOGGER.info("GEODATA_API_URL is not set; skipping the API cache refresh.")
        return {"refreshed": False, "reason": "GEODATA_API_URL not set"}
    request = urllib.request.Request(f"{api_url}/api/v1/refresh", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        LOGGER.info("API caches refreshed: %s", payload.get("fire_date_range"))
        return {"refreshed": True, **payload}
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        LOGGER.warning("Could not refresh the API at %s: %s", api_url, error)
        return {"refreshed": False, "reason": str(error)}
