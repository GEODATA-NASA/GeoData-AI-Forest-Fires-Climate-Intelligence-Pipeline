"""Airflow DAG for the GeoData AI pipeline.

    download_weather ─┬─ transform_fires ───┬─ check_cleaned_outputs ─ load_database ─ train_model ─ refresh_api
                      └─ transform_weather ─┘

Every stage is idempotent (it skips work whose output is already current), so
a daily run is cheap when nothing changed, and a failed task can be retried
on its own. Trigger with ``{"retrain": true}`` to retrain over an existing model.

Configuration (environment variables on the Airflow workers):
    GEODATA_HOME        repository root (defaults to two levels above this file)
    GEODATA_SCHEDULE    cron or preset, default "@daily"; "none" for manual only
    GEODATA_API_URL     API base URL to refresh after training (optional)
plus the pipeline's own variables (DB_*, WEATHER_*, ...) from .env.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

GEODATA_HOME = Path(os.getenv("GEODATA_HOME", Path(__file__).resolve().parents[2]))
if str(GEODATA_HOME) not in sys.path:
    sys.path.insert(0, str(GEODATA_HOME))

try:  # Airflow 3
    from airflow.sdk import dag, task, get_current_context
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task
    from airflow.operators.python import get_current_context

_schedule = os.getenv("GEODATA_SCHEDULE", "@daily").strip()
SCHEDULE = None if _schedule.lower() in {"", "none", "manual"} else _schedule

DEFAULT_ARGS = {
    "owner": "geodata",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


@dag(
    dag_id="geodata_pipeline",
    description="NASA FIRMS + NASA POWER ETL, PostGIS load and high-FRP model training",
    schedule=SCHEDULE,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    params={"retrain": False},
    tags=["geodata", "etl", "ml"],
    doc_md=__doc__,
)
def geodata_pipeline():
    @task(execution_timeout=timedelta(hours=3))
    def download_weather():
        from orchestration.tasks import download_weather as run
        return run()

    @task(execution_timeout=timedelta(hours=2))
    def transform_fires():
        from orchestration.tasks import transform_fires as run
        return run()

    @task(execution_timeout=timedelta(minutes=30))
    def transform_weather():
        from orchestration.tasks import transform_weather as run
        return run()

    @task
    def check_cleaned_outputs():
        from orchestration.tasks import check_cleaned_outputs as run
        return run()

    @task(execution_timeout=timedelta(hours=6))
    def load_database():
        from orchestration.tasks import load_database as run
        return run()

    @task(execution_timeout=timedelta(hours=2))
    def train_model():
        from orchestration.tasks import train_model as run
        retrain = bool(get_current_context()["params"].get("retrain", False))
        return run(force=retrain or None)

    @task(retries=0)
    def refresh_api():
        from orchestration.tasks import refresh_api as run
        return run()

    weather = download_weather()
    cleaned = [transform_fires(), transform_weather()]
    weather >> cleaned
    checked = check_cleaned_outputs()
    cleaned >> checked
    checked >> load_database() >> train_model() >> refresh_api()


pipeline_dag = geodata_pipeline()
