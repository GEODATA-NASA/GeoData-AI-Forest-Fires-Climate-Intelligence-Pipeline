"""Exercise the SQL repository's query builder against an in-memory SQLite database.

The SQL it emits is standard enough (SUBSTR, BETWEEN, IN, GROUP BY, NULLS LAST)
to run on both PostgreSQL and SQLite, so this catches binding and syntax errors
without needing a live PostgreSQL server.
"""

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine

from api.filters import FireFilters
from api.repositories.postgres import PostgresFireRepository
from tests.conftest import FIRE_COLUMNS, FIRE_ROWS, WEATHER_COLUMNS, WEATHER_ROWS


@pytest.fixture(scope="module")
def repository():
    engine = create_engine("sqlite://")
    pd.DataFrame(FIRE_ROWS, columns=FIRE_COLUMNS).to_sql("fire_events", engine, index=False)
    pd.DataFrame(WEATHER_ROWS, columns=WEATHER_COLUMNS).to_sql("weather_data", engine, index=False)
    repo = PostgresFireRepository(
        {"user": "", "password": "", "host": "", "port": 0, "dbname": "memory"},
        schema="main",
        engine=engine,
    )
    yield repo
    repo.close()


def test_inspection_and_health(repository):
    assert repository.health()["ok"] is True
    assert repository.available_columns == FIRE_COLUMNS
    assert repository._date_is_text is True
    assert repository.has_weather is True


def test_fires_with_all_filters(repository):
    filters = FireFilters(
        start_date=date(2025, 1, 1), end_date=date(2025, 1, 31),
        min_lat=29, max_lat=31, min_lon=30, max_lon=32,
        sources=["modis", "viirs_j1"], min_confidence=50, min_frp=1, max_frp=500,
        daynight="D", limit=10, offset=0, order="frp_desc",
    )
    frame = repository.fires(filters)
    assert [round(value, 1) for value in frame["frp"]] == [120.0, 45.0]
    assert pd.api.types.is_datetime64_any_dtype(frame["acq_date"])
    assert repository.count_fires(filters) == 2


def test_pagination_and_ordering(repository):
    newest = repository.fires(FireFilters(limit=1, order="acq_date_desc"))
    assert newest.iloc[0]["acq_date"].date() == date(2025, 3, 16)
    oldest = repository.fires(FireFilters(limit=1, order="acq_date_asc"))
    assert oldest.iloc[0]["acq_date"].date() == date(2025, 1, 2)
    page_two = repository.fires(FireFilters(limit=2, offset=2, order="frp_asc"))
    assert len(page_two) == 2


def test_stats_and_date_range(repository):
    stats = repository.fire_stats(FireFilters(), high_frp_threshold=10)
    assert stats["total"] == 6
    assert stats["high_frp_count"] == 3
    assert [row["date"] for row in stats["by_date"]] == ["2025-01-02", "2025-01-05", "2025-03-15", "2025-03-16"]
    assert stats["by_source"][0] == {"source": "modis", "count": 4, "avg_frp": pytest.approx(75.125), "high_frp_count": 2}
    assert {row["daynight"] for row in stats["by_daynight"]} == {"D", "N"}
    assert repository.date_range() == (date(2025, 1, 2), date(2025, 3, 16))


def test_weather_table_is_read(repository):
    weather = repository.weather()
    assert len(weather) == 3
    assert "RH2M" in weather.columns
