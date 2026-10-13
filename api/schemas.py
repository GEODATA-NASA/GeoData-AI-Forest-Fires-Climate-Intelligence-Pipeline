"""Pydantic models describing API requests and responses."""

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FireObservation(BaseModel):
    """A single (possibly hypothetical) fire detection to score."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "latitude": 30.05, "longitude": 31.2, "acq_date": "2025-01-15", "acq_time": 1130,
            "brightness": 325.4, "confidence": 80, "scan": 1.1, "track": 1.0, "type": 0,
        }
    })

    latitude: float = Field(..., ge=-90, le=90, description="Decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Decimal degrees")
    acq_date: date = Field(..., description="Acquisition date (YYYY-MM-DD)")
    acq_time: int | None = Field(None, ge=0, le=2359, description="Acquisition time HHMM (UTC)")
    brightness: float | None = Field(None, gt=0, description="Brightness temperature (Kelvin)")
    confidence: int | None = Field(None, ge=0, le=100, description="Detection confidence 0-100")
    scan: float | None = Field(None, gt=0, description="Along-scan pixel size (km)")
    track: float | None = Field(None, gt=0, description="Along-track pixel size (km)")
    type: int | None = Field(None, ge=0, le=3, description="FIRMS inferred hot-spot type")
    rh2m: float | None = Field(None, ge=0, le=100, description="Relative humidity at 2 m (%)")
    prectotcorr: float | None = Field(None, ge=0, description="Precipitation (mm/day)")
    gwettop: float | None = Field(None, ge=0, le=1, description="Surface soil wetness (0-1)")


class BatchPredictionRequest(BaseModel):
    observations: list[FireObservation] = Field(..., min_length=1)


class WeatherUsed(BaseModel):
    matched: bool = Field(..., description="A same-day station within the radius was found")
    supplied: bool = Field(False, description="Weather values were provided by the caller")
    distance_km: float | None = None
    rh2m: float | None = None
    prectotcorr: float | None = None
    gwettop: float | None = None


class Prediction(BaseModel):
    high_frp_probability: float
    is_high_frp: bool
    predicted_label: str
    decision_threshold: float
    high_frp_threshold_mw: float
    weather: WeatherUsed
    input: dict[str, Any] | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[Prediction]


class FireListResponse(BaseModel):
    items: list[dict[str, Any]]
    count: int
    limit: int
    offset: int
    total: int | None = None
    filters: dict[str, Any]
    backend: str


class DateRange(BaseModel):
    start_date: date | None
    end_date: date | None


class ByDateStat(BaseModel):
    date: str
    count: int
    avg_frp: float | None = None
    max_frp: float | None = None
    high_frp_count: int = 0


class BySourceStat(BaseModel):
    source: str | None
    count: int
    avg_frp: float | None = None
    high_frp_count: int = 0


class ByDayNightStat(BaseModel):
    daynight: str | None
    count: int


class FireStatsResponse(BaseModel):
    total: int
    avg_frp: float | None
    max_frp: float | None
    high_frp_count: int
    high_frp_share: float | None
    high_frp_threshold_mw: float
    by_date: list[ByDateStat]
    by_source: list[BySourceStat]
    by_daynight: list[ByDayNightStat]
    filters: dict[str, Any]
    backend: str


class WeatherListResponse(BaseModel):
    items: list[dict[str, Any]]
    count: int
    features: list[str]
    max_distance_km: float


class ModelInfo(BaseModel):
    loaded: bool
    model_path: str
    estimator: str | None
    n_estimators: int | None
    feature_columns: list[str]
    weather_features: list[str]
    target: str | None
    high_frp_threshold_mw: float
    decision_threshold: float


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: str
    data: dict[str, Any]
    model: dict[str, Any]
    weather: dict[str, Any]
    fire_date_range: DateRange
    cache_ttl_seconds: int


class RefreshResponse(BaseModel):
    weather: dict[str, Any]
    fire_date_range: DateRange
