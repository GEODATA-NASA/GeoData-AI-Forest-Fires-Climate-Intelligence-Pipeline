"""NASA POWER weather observations used for feature enrichment."""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query

from api.deps import RepositoryDep, SettingsDep, WeatherIndexDep
from api.schemas import WeatherListResponse
from api.serialization import frame_to_records

router = APIRouter(prefix="/api/v1/weather", tags=["weather"])


@router.get("", response_model=WeatherListResponse, summary="List weather observations")
def list_weather(
    _repository: RepositoryDep,  # 503 when no data backend is available, like the fire endpoints
    weather_index: WeatherIndexDep,
    settings: SettingsDep,
    start_date: Annotated[date | None, Query()] = None,
    end_date: Annotated[date | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50_000)] = 5_000,
) -> WeatherListResponse:
    if weather_index is None or len(weather_index) == 0:
        return WeatherListResponse(items=[], count=0, features=[], max_distance_km=settings.max_weather_distance_km)
    frame = weather_index.rows(start_date, end_date, limit)
    return WeatherListResponse(
        items=frame_to_records(frame),
        count=len(frame),
        features=weather_index.features,
        max_distance_km=weather_index.max_distance_km,
    )
