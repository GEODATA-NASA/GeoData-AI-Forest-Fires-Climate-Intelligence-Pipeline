"""Fire detection endpoints: paginated list, GeoJSON for maps, and aggregate statistics."""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from api.deps import FiltersDep, PredictorDep, RepositoryDep, SettingsDep, WeatherIndexDep
from api.schemas import DateRange, FireListResponse, FireStatsResponse
from api.serialization import frame_to_records

LOGGER = logging.getLogger("geodata.api.fires")
router = APIRouter(prefix="/api/v1/fires", tags=["fires"])

GEOJSON_PROPERTIES = [
    "acq_date", "acq_time", "brightness", "bright_t31", "frp", "confidence",
    "scan", "track", "daynight", "type", "satellite", "instrument", "satellite_source",
]


def _attach_predictions(frame, predictor, weather_index, decision_threshold):
    if frame.empty:
        return frame
    predictions = predictor.predict_frame(frame, weather_index, decision_threshold)
    return frame.join(predictions)


@router.get("", response_model=FireListResponse, summary="List fire detections")
def list_fires(
    filters: FiltersDep,
    repository: RepositoryDep,
    settings: SettingsDep,
    weather_index: WeatherIndexDep,
    predictor: PredictorDep,
    predict: Annotated[bool, Query(description="Attach model predictions to each row")] = False,
    include_total: Annotated[bool, Query(description="Also count all matching rows (slower on large tables)")] = False,
    decision_threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> FireListResponse:
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
    predictor: PredictorDep,
    predict: Annotated[bool, Query(description="Attach predicted severity to each feature")] = True,
    limit: Annotated[int | None, Query(ge=1, description="Maximum number of features")] = None,
    decision_threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> dict:
    # GeoJSON responses may be larger than a normal page; cap separately.
    filters.limit = min(limit or 10_000, settings.max_geojson_features)
    frame = repository.fires(filters)
    if predict:
        frame = _attach_predictions(frame, predictor, weather_index, decision_threshold)

    records = frame_to_records(frame)
    property_names = [name for name in GEOJSON_PROPERTIES if name in frame.columns]
    prediction_names = [
        name for name in ("high_frp_probability", "is_high_frp", "predicted_label", "weather_matched")
        if name in frame.columns
    ]
    features = []
    for record in records:
        properties = {name: record.get(name) for name in property_names + prediction_names}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [record["longitude"], record["latitude"]]},
                "properties": properties,
            }
        )
    high = sum(1 for record in records if record.get("is_high_frp")) if predict else None
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "count": len(features),
            "limit": filters.limit,
            "truncated": len(features) >= filters.limit,
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
