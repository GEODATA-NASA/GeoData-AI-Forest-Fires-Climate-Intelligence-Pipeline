"""Runtime state shared by the routers: data backend, model, and refreshable caches.

The weather index and the fire date range are rebuilt from the backend when
they are older than ``Settings.cache_ttl_seconds`` or when ``refresh()`` is
called (``POST /api/v1/refresh``), so a pipeline run that loads new rows shows
up without restarting the API.
"""

import logging
import threading
import time
from datetime import date

from api.predictor import SeverityPredictor
from api.repositories.base import FireRepository
from api.settings import Settings
from api.weather_index import WeatherIndex

LOGGER = logging.getLogger("geodata.api.runtime")


class RuntimeData:
    def __init__(self, settings: Settings, repository: FireRepository | None, predictor: SeverityPredictor | None):
        self.settings = settings
        self.repository = repository
        self.predictor = predictor
        self._lock = threading.Lock()
        self._weather_index: WeatherIndex | None = None
        self._weather_built_at: float | None = None
        self._weather_error: str | None = None
        self._date_range: tuple[date | None, date | None] | None = None
        self._date_range_at: float | None = None
        self._date_range_error: str | None = None

    # ------------------------------------------------------------- helpers
    def _stale(self, built_at: float | None) -> bool:
        ttl = self.settings.cache_ttl_seconds
        return built_at is None or (ttl > 0 and time.monotonic() - built_at > ttl)

    # ------------------------------------------------------------- weather
    def weather_index(self, force: bool = False) -> WeatherIndex | None:
        if self.repository is None:
            return None
        if not force and not self._stale(self._weather_built_at):
            return self._weather_index
        with self._lock:
            if force or self._stale(self._weather_built_at):
                try:
                    index = WeatherIndex(self.repository.weather(), self.settings.max_weather_distance_km)
                    self._weather_index = index
                    self._weather_error = None
                    LOGGER.info("Weather index ready: %s", index.summary())
                except Exception as error:  # noqa: BLE001
                    self._weather_error = str(error)
                    LOGGER.exception("Weather data could not be indexed; predictions will use imputed weather.")
                self._weather_built_at = time.monotonic()
        return self._weather_index

    @property
    def weather_error(self) -> str | None:
        return self._weather_error

    # ---------------------------------------------------------- date range
    def date_range(self, force: bool = False) -> tuple[date | None, date | None]:
        if self.repository is None:
            return (None, None)
        if not force and not self._stale(self._date_range_at):
            return self._date_range or (None, None)
        with self._lock:
            if force or self._stale(self._date_range_at):
                try:
                    self._date_range = self.repository.date_range()
                    self._date_range_error = None
                except Exception as error:  # noqa: BLE001
                    self._date_range_error = str(error)
                    LOGGER.warning("Could not compute the fire date range: %s", error)
                    self._date_range = (None, None)
                self._date_range_at = time.monotonic()
        return self._date_range or (None, None)

    @property
    def date_range_error(self) -> str | None:
        return self._date_range_error

    # ------------------------------------------------------------- refresh
    def refresh(self) -> dict:
        """Rebuild every cache now; returns a summary for the refresh endpoint."""
        index = self.weather_index(force=True)
        start, end = self.date_range(force=True)
        return {
            "weather": index.summary() if index is not None else {"observations": 0, "error": self._weather_error},
            "fire_date_range": {"start_date": start.isoformat() if start else None, "end_date": end.isoformat() if end else None},
        }

    def close(self) -> None:
        if self.repository is not None:
            self.repository.close()
