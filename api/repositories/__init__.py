"""Storage backends for fire and weather data."""

import logging

from api.repositories.base import FIRE_COLUMNS, FireRepository
from api.repositories.csv_backend import CsvFireRepository
from api.repositories.postgres import PostgresFireRepository
from api.settings import Settings

LOGGER = logging.getLogger("geodata.api.repository")

__all__ = [
    "FIRE_COLUMNS",
    "FireRepository",
    "CsvFireRepository",
    "PostgresFireRepository",
    "create_repository",
]


def create_repository(settings: Settings) -> FireRepository:
    """Pick the backend from ``DATA_BACKEND``: postgres, csv, or auto (postgres then csv)."""
    backend = settings.data_backend
    if backend not in {"auto", "postgres", "csv"}:
        raise ValueError(f"Unknown DATA_BACKEND '{backend}'. Use auto, postgres or csv.")

    if backend in {"auto", "postgres"}:
        try:
            repository = PostgresFireRepository(settings.db_config, settings.db_schema)
            LOGGER.info(
                "Connected to PostgreSQL database '%s' on %s:%s (schema %s).",
                settings.db_config.get("dbname"),
                settings.db_config.get("host"),
                settings.db_config.get("port"),
                settings.db_schema,
            )
            return repository
        except Exception as error:  # noqa: BLE001 - we want to report any connection problem
            if backend == "postgres":
                raise
            reason = (str(error).splitlines() or [type(error).__name__])[0]
            LOGGER.warning("PostgreSQL is unavailable (%s). Falling back to the CSV backend.", reason)

    repository = CsvFireRepository(
        fires_file=settings.fires_csv,
        sample_file=settings.fires_sample_csv,
        weather_file=settings.weather_csv,
        max_rows=settings.csv_max_rows,
    )
    LOGGER.info("Serving %s fire rows from %s.", len(repository), repository.source_file)
    return repository
