"""Filter object shared by the fire endpoints and both storage backends."""

from dataclasses import dataclass
from datetime import date

ORDER_OPTIONS = ("acq_date_desc", "acq_date_asc", "frp_desc", "frp_asc")


@dataclass
class FireFilters:
    start_date: date | None = None
    end_date: date | None = None
    min_lat: float | None = None
    max_lat: float | None = None
    min_lon: float | None = None
    max_lon: float | None = None
    sources: list[str] | None = None
    min_confidence: int | None = None
    min_frp: float | None = None
    max_frp: float | None = None
    daynight: str | None = None
    limit: int = 500
    offset: int = 0
    order: str = "acq_date_desc"

    @property
    def has_bbox(self) -> bool:
        return None not in (self.min_lat, self.max_lat, self.min_lon, self.max_lon)

    def describe(self) -> dict:
        """Echo the active filters back to API clients (None values dropped)."""
        payload = {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "bbox": [self.min_lon, self.min_lat, self.max_lon, self.max_lat] if self.has_bbox else None,
            "sources": self.sources,
            "min_confidence": self.min_confidence,
            "min_frp": self.min_frp,
            "max_frp": self.max_frp,
            "daynight": self.daynight,
            "order": self.order,
        }
        return {key: value for key, value in payload.items() if value is not None}
