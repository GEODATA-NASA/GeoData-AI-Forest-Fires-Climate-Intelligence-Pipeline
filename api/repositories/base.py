"""Abstract data access interface used by the API routers."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from api.filters import FireFilters

# Columns of the cleaned FIRMS dataset (and the ``fire_events`` table).
FIRE_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
    "type",
    "satellite_source",
]

NUMERIC_FIRE_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_time",
    "confidence",
    "bright_t31",
    "frp",
    "type",
]


class FireRepository(ABC):
    """Read-only access to fire detections and weather observations."""

    name: str = "abstract"

    @abstractmethod
    def health(self) -> dict:
        """Cheap connectivity/status check used by ``/health``."""

    @abstractmethod
    def fires(self, filters: FireFilters) -> pd.DataFrame:
        """Return a page of fire detections (columns from ``FIRE_COLUMNS``)."""

    @abstractmethod
    def count_fires(self, filters: FireFilters) -> int:
        """Total rows matching the filters (ignores limit/offset)."""

    @abstractmethod
    def fire_stats(self, filters: FireFilters, high_frp_threshold: float) -> dict:
        """Aggregate counts by day, satellite source and day/night."""

    @abstractmethod
    def date_range(self) -> tuple[date | None, date | None]:
        """Earliest and latest acquisition dates available."""

    @abstractmethod
    def weather(self) -> pd.DataFrame:
        """All weather observations (small: one row per location per day)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections or memory."""


def empty_stats() -> dict:
    return {
        "total": 0,
        "avg_frp": None,
        "max_frp": None,
        "high_frp_count": 0,
        "high_frp_share": None,
        "by_date": [],
        "by_source": [],
        "by_daynight": [],
    }
