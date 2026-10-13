"""Fire detection endpoints: paginated list, GeoJSON for maps, and aggregate statistics."""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from api.deps import FiltersDep, OptionalPredictorDep, RepositoryDep, SettingsDep, WeatherIndexDep
from api.schemas import DateRange, FireListResponse, FireStatsResponse
from api.serialization import frame_to_records

LOGGER = logging.getLogger("geodata.api.fires")
router = APIRouter(prefix="/api/v1/fires", tags=["fires"])

GEOJSON_PROPERTIES = [
    "acq_date", "acq_time", "brightness", "bright_t31", "frp", "confidence",
    "scan", "track", "daynight", "type", "satellite", "instrument", "satellite_source",
]
PREDICTION_PROPERTIES = ("high_frp_probability", "is_high_frp", "predicted_label", "weather_matched")


def _require_predictor(predictor):
    if predictor is None:
        raise HTTPException(status_code=503, detail="The model is not loaded, so predictions are unavailable. Retry with predict=false.")
    return predictor


def _attach_predictions(frame, predictor, weather_index, decision_threshold):
    if frame.empty:
        return frame
    return frame.join(predictor.predict_frame(frame, weather_index, decision_threshold))


def _fetch_page(repository, filters, cap):
    """Fetch one row more than requested so `truncated` is exact, then trim."""
    filters.limit = cap + 1
    frame = repository.fires(filters)
    filters.limit = cap
    truncated = len(frame) > cap
    return frame.iloc[:cap], truncated


@router.get("", response_model=FireListResponse, summary="List fire detections")
def list_fires(
    filters: FiltersDep,
    repository: RepositoryDep,
    weather_index: WeatherIndexDep,
    predictor: OptionalPredictorDep,
    predict: Annotated[bool, Query(description="Attach model predictions to each row")] = False,
    include_total: Annotated[bool, Query(description="Also count all matching rows (slower on large tables)")] = False,
    decision_threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> FireListResponse:
    if predict:
        predictor = _require_predictor(predictor)
    frame = repository.fires(filters)
    if predict:
        frame = _attach_predictions(frame, predictor, weather_index, decision_threshold)
    total = repository.count_fires(filters) if include_total else None
    return FireListResponse(
        items=frame_to_records(frame),
        count=len(frame),
        limit=filters.limit,
        offset=filters.offset,
        total=total,
        filters=filters.describe(),
        backend=repository.name,
    )


@router.get("/geojson", summary="Fire detections as a GeoJSON FeatureCollection")
def fires_geojson(
    filters: FiltersDep,
    repository: RepositoryDep,
    settings: SettingsDep,
    weather_index: WeatherIndexDep,
    predictor: OptionalPredictorDep,
    predict: Annotated[bool, Query(description="Attach predicted severity to each feature")] = True,
    limit: Annotated[int | None, Query(ge=1, description="Maximum number of features")] = None,
    decision_threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> dict:
    if predict:
        predictor = _require_predictor(predictor)
    # GeoJSON responses may be larger than a normal page; cap separately.
    cap = min(limit or 10_000, settings.max_geojson_features)
    frame, truncated = _fetch_page(repository, filters, cap)
    if predict:
        frame = _attach_predictions(frame, predictor, weather_index, decision_threshold)

    records = frame_to_records(frame)
    property_names = [name for name in GEOJSON_PROPERTIES if name in frame.columns]
    prediction_names = [name for name in PREDICTION_PROPERTIES if name in frame.columns]
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [record["longitude"], record["latitude"]]},
            "properties": {name: record.get(name) for name in property_names + prediction_names},
        }
        for record in records
    ]
    high = sum(1 for record in records if record.get("is_high_frp")) if predict else None
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "count": len(features),
            "limit": cap,
            "truncated": truncated,
            "predicted_high_frp": high,
            "filters": filters.describe(),
            "backend": repository.name,
        },
    }


@router.get("/stats", response_model=FireStatsResponse, summary="Aggregate statistics for the filtered fires")
def fire_stats(filters: FiltersDep, repository: RepositoryDep, settings: SettingsDep) -> FireStatsResponse:
    stats = repository.fire_stats(filters, settings.high_frp_threshold_mw)
    return FireStatsResponse(
        **stats,
        high_frp_threshold_mw=settings.high_frp_threshold_mw,
        filters=filters.describe(),
        backend=repository.name,
    )


@router.get("/date-range", response_model=DateRange, summary="Earliest and latest acquisition dates")
def fire_date_range(repository: RepositoryDep) -> DateRange:
    try:
        start, end = repository.date_range()
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Failed to compute date range")
        raise HTTPException(status_code=500, detail=f"Could not compute date range: {error}") from error
    return DateRange(start_date=start, end_date=end)
