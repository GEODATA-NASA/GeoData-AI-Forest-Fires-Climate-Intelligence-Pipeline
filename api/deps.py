"""FastAPI dependencies: application state accessors and the shared fire filter parser."""

from datetime import date
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request

from api.filters import ORDER_OPTIONS, FireFilters
from api.predictor import SeverityPredictor
from api.repositories.base import FireRepository
from api.settings import Settings
from api.weather_index import WeatherIndex


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repository(request: Request) -> FireRepository:
    repository = getattr(request.app.state, "repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="Data backend is not available.")
    return repository


def get_predictor(request: Request) -> SeverityPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return predictor


def get_weather_index(request: Request) -> WeatherIndex | None:
    return getattr(request.app.state, "weather_index", None)


def fire_filters(
    settings: Annotated[Settings, Depends(get_settings)],
    start_date: Annotated[date | None, Query(description="Earliest acquisition date (inclusive)")] = None,
    end_date: Annotated[date | None, Query(description="Latest acquisition date (inclusive)")] = None,
    bbox: Annotated[
        str | None,
        Query(description="Bounding box 'min_lon,min_lat,max_lon,max_lat'", examples=["24,22,37,32"]),
    ] = None,
    source: Annotated[
        list[str] | None,
        Query(description="Satellite source(s): modis, viirs_j1, viirs_suomi (repeatable)"),
    ] = None,
    min_confidence: Annotated[int | None, Query(ge=0, le=100)] = None,
    min_frp: Annotated[float | None, Query(ge=0, description="Minimum fire radiative power (MW)")] = None,
    max_frp: Annotated[float | None, Query(ge=0)] = None,
    daynight: Annotated[str | None, Query(pattern="^[DN]$", description="D = day, N = night")] = None,
    limit: Annotated[int | None, Query(ge=1)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    order: Annotated[str, Query(description=f"One of {', '.join(ORDER_OPTIONS)}")] = "acq_date_desc",
) -> FireFilters:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date.")
    if order not in ORDER_OPTIONS:
        raise HTTPException(status_code=422, detail=f"order must be one of {', '.join(ORDER_OPTIONS)}.")

    min_lon = min_lat = max_lon = max_lat = None
    if bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = (float(part) for part in bbox.split(","))
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail="bbox must be 'min_lon,min_lat,max_lon,max_lat'."
            ) from error
        if not (-90 <= min_lat <= max_lat <= 90) or not (-180 <= min_lon <= max_lon <= 180):
            raise HTTPException(status_code=422, detail="bbox coordinates are out of range or inverted.")

    page_size = settings.default_page_size if limit is None else min(limit, settings.max_page_size)
    sources = [item.strip().lower() for item in source if item.strip()] if source else None
    return FireFilters(
        start_date=start_date,
        end_date=end_date,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        sources=sources or None,
        min_confidence=min_confidence,
        min_frp=min_frp,
        max_frp=max_frp,
        daynight=daynight,
        limit=page_size,
        offset=offset,
        order=order,
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
RepositoryDep = Annotated[FireRepository, Depends(get_repository)]
PredictorDep = Annotated[SeverityPredictor, Depends(get_predictor)]
WeatherIndexDep = Annotated[WeatherIndex | None, Depends(get_weather_index)]
FiltersDep = Annotated[FireFilters, Depends(fire_filters)]
