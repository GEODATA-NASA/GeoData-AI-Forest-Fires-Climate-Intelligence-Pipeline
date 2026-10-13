"""Load stage: staging-table swap, skip logic and weather reload (SQLite stands in for PostgreSQL)."""

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from loadation import load
from tests.conftest import FIRE_COLUMNS, FIRE_ROWS, WEATHER_COLUMNS, WEATHER_ROWS


@pytest.fixture()
def sqlite_load(tmp_path, monkeypatch):
    database = tmp_path / "geodata.sqlite"
    url = f"sqlite:///{database.as_posix()}"
    monkeypatch.setattr(load, "get_connection_url", lambda: url)
    monkeypatch.setattr(load, "CLEANED_DATA_DIR", str(tmp_path))
    pd.DataFrame(FIRE_ROWS, columns=FIRE_COLUMNS).to_csv(tmp_path / "cleaned_master_fires.csv", index=False)
    pd.DataFrame(WEATHER_ROWS, columns=WEATHER_COLUMNS).to_csv(tmp_path / "cleaned_weather.csv", index=False)
    engine = create_engine(url)
    yield tmp_path, engine
    engine.dispose()


def _count(engine, table):
    with engine.connect() as connection:
        return connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()


def test_first_load_creates_both_tables_without_staging_leftovers(sqlite_load):
    _, engine = sqlite_load
    load.load_data_to_postgres()
    tables = set(inspect(engine).get_table_names())
    assert {"fire_events", "weather_data"} <= tables
    assert not {"fire_events_staging", "weather_data_staging"} & tables
    assert _count(engine, "fire_events") == len(FIRE_ROWS)
    assert _count(engine, "weather_data") == len(WEATHER_ROWS)
    indexes = {ix["name"] for ix in inspect(engine).get_indexes("weather_data")}
    assert "ix_weather_data_date" in indexes


def test_rerun_skips_current_tables_and_reloads_changed_weather(sqlite_load):
    folder, engine = sqlite_load
    load.load_data_to_postgres()
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO fire_events (latitude, longitude, acq_date, frp) VALUES (1, 2, '2025-05-05', 3)"))
    extended = pd.DataFrame(WEATHER_ROWS * 4, columns=WEATHER_COLUMNS)
    extended.to_csv(folder / "cleaned_weather.csv", index=False)

    load.load_data_to_postgres()
    assert _count(engine, "fire_events") == len(FIRE_ROWS) + 1  # fires are not reloaded
    assert _count(engine, "weather_data") == len(extended)        # weather is


def test_a_crashed_upload_never_replaces_the_live_table(sqlite_load, monkeypatch):
    _, engine = sqlite_load
    load.load_data_to_postgres()
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE weather_data"))
        connection.execute(text("CREATE TABLE weather_data (date INTEGER)"))
        connection.execute(text("INSERT INTO weather_data VALUES (20250101)"))

    original = pd.DataFrame.to_sql

    def failing_to_sql(self, name, *args, **kwargs):
        if name == "weather_data_staging":
            raise RuntimeError("connection lost")
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_sql", failing_to_sql)
    with pytest.raises(RuntimeError, match="connection lost"):
        load.load_data_to_postgres()
    assert _count(engine, "weather_data") == 1  # the previous table is untouched


def test_csv_row_count_ignores_the_header(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    assert load.csv_row_count(path) == 2
