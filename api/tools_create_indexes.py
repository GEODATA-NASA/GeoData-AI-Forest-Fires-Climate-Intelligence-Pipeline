"""Create the PostgreSQL indexes the API relies on for fast filtering.

``loadation/load.py`` bulk-loads ``fire_events`` without indexes, which is
fine for one-off analysis but slow for interactive API queries over tens of
millions of rows. Run once (idempotent)::

    python -m api.tools_create_indexes
"""

import logging
import sys
from pathlib import Path

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.repositories.postgres import PostgresFireRepository  # noqa: E402
from api.settings import load_settings  # noqa: E402

LOGGER = logging.getLogger("geodata.api.indexes")

INDEXES = {
    "ix_fire_events_acq_date": ["acq_date"],
    "ix_fire_events_lat_lon": ["latitude", "longitude"],
    "ix_fire_events_source": ["satellite_source"],
    "ix_fire_events_frp": ["frp"],
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    repository = PostgresFireRepository(settings.db_config, settings.db_schema)
    table = repository._table(repository.fires_table)
    with repository.engine.begin() as connection:
        for name, columns in INDEXES.items():
            if any(column not in repository._columns for column in columns):
                LOGGER.info("Skipping %s: column(s) missing.", name)
                continue
            column_sql = ", ".join(repository._col(column) for column in columns)
            LOGGER.info("Creating index %s on (%s) ...", name, column_sql)
            connection.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column_sql})"))
        connection.execute(text(f"ANALYZE {table}"))
    repository.close()
    LOGGER.info("Indexes are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
