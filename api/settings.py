"""Runtime settings for the API, loaded from environment variables / ``.env``."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Importing the project config also loads ``.env`` and sets up data directories.
from config import CLEANED_DATA_DIR, DB_CONFIG, PROJECT_ROOT

ROOT_DIR = Path(PROJECT_ROOT)


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass
class Settings:
    data_backend: str = "auto"  # auto | postgres | csv
    db_config: dict = field(default_factory=lambda: dict(DB_CONFIG))
    db_schema: str = "public"
    fires_csv: Path = Path(CLEANED_DATA_DIR) / "cleaned_master_fires.csv"
    fires_sample_csv: Path = Path(CLEANED_DATA_DIR) / "fires_sample.csv"
    weather_csv: Path = Path(CLEANED_DATA_DIR) / "cleaned_weather.csv"
    csv_max_rows: int = 500_000
    model_path: Path = ROOT_DIR / "modeling" / "artifacts" / "high_frp_classifier.joblib"
    high_frp_threshold_mw: float = 10.0
    max_weather_distance_km: float = 100.0
    decision_threshold: float = 0.5
    default_page_size: int = 500
    max_page_size: int = 5_000
    max_geojson_features: int = 50_000
    max_batch_predictions: int = 5_000
    cache_ttl_seconds: int = 300  # weather index and date range are rebuilt after this
    assistant_enabled: bool = True
    assistant_model: str = "claude-opus-5-5"
    assistant_effort: str = "medium"
    assistant_max_rounds: int = 8
    assistant_fallbacks: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    log_file: Path = ROOT_DIR / "logs" / "api.log"
    log_level: str = "INFO"
    frontend_dir: Path = ROOT_DIR / "frontend"
    serve_frontend: bool = True


def load_settings() -> Settings:
    """Build settings from the environment (``.env`` is loaded by ``config``)."""
    defaults = Settings()
    return Settings(
        data_backend=os.getenv("DATA_BACKEND", defaults.data_backend).strip().lower(),
        db_schema=os.getenv("DB_SCHEMA", defaults.db_schema),
        fires_csv=Path(os.getenv("FIRES_CSV_FILE", defaults.fires_csv)),
        fires_sample_csv=Path(os.getenv("FIRES_SAMPLE_FILE", defaults.fires_sample_csv)),
        weather_csv=Path(os.getenv("WEATHER_CSV_FILE", defaults.weather_csv)),
        csv_max_rows=int(os.getenv("CSV_MAX_ROWS", defaults.csv_max_rows)),
        model_path=Path(os.getenv("MODEL_PATH", defaults.model_path)),
        high_frp_threshold_mw=float(os.getenv("HIGH_FRP_THRESHOLD", defaults.high_frp_threshold_mw)),
        max_weather_distance_km=float(
            os.getenv("MAX_WEATHER_DISTANCE_KM", defaults.max_weather_distance_km)
        ),
        decision_threshold=float(os.getenv("DECISION_THRESHOLD", defaults.decision_threshold)),
        default_page_size=int(os.getenv("API_DEFAULT_PAGE_SIZE", defaults.default_page_size)),
        max_page_size=int(os.getenv("API_MAX_PAGE_SIZE", defaults.max_page_size)),
        max_geojson_features=int(
            os.getenv("API_MAX_GEOJSON_FEATURES", defaults.max_geojson_features)
        ),
        max_batch_predictions=int(
            os.getenv("API_MAX_BATCH_PREDICTIONS", defaults.max_batch_predictions)
        ),
        cache_ttl_seconds=int(os.getenv("API_CACHE_TTL_SECONDS", defaults.cache_ttl_seconds)),
        assistant_enabled=os.getenv("ASSISTANT_ENABLED", "1").lower() not in {"0", "false", "no"},
        assistant_model=os.getenv("ASSISTANT_MODEL", defaults.assistant_model),
        assistant_effort=os.getenv("ASSISTANT_EFFORT", defaults.assistant_effort),
        assistant_max_rounds=int(os.getenv("ASSISTANT_MAX_ROUNDS", defaults.assistant_max_rounds)),
        assistant_fallbacks=os.getenv("ASSISTANT_FALLBACKS", "1").lower() not in {"0", "false", "no"},
        cors_origins=_env_list("CORS_ORIGINS", "*"),
        log_file=Path(os.getenv("API_LOG_FILE", defaults.log_file)),
        log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
        frontend_dir=Path(os.getenv("FRONTEND_DIR", defaults.frontend_dir)),
        serve_frontend=os.getenv("SERVE_FRONTEND", "1").lower() not in {"0", "false", "no"},
    )
