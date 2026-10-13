"""PostgreSQL/PostGIS backend reading the ``fire_events`` and ``weather_data`` tables."""

import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import URL, bindparam, create_engine, inspect, text
from sqlalchemy.engine import Engine

from api.filters import FireFilters
from api.repositories.base import FIRE_COLUMNS, FireRepository, empty_stats
from modeling.train_model import canonical_date

LOGGER = logging.getLogger("geodata.api.repository.postgres")


class PostgresFireRepository(FireRepository):
    name = "postgres"

    def __init__(
        self,
        db_config: dict,
        schema: str = "public",
        fires_table: str = "fire_events",
        weather_table: str = "weather_data",
        engine: Engine | None = None,
    ):
        self.db_config = db_config
        self.schema = schema
        self.fires_table = fires_table
        self.weather_table = weather_table
        self.engine = engine or create_engine(
            URL.create(
                "postgresql+psycopg2",
                username=db_config["user"],
                password=db_config["password"],
                host=db_config["host"],
                port=int(db_config["port"]),
                database=db_config["dbname"],
            ),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )
        self._preparer = self.engine.dialect.identifier_preparer
        self._inspect_tables()

    # ------------------------------------------------------------------ setup
    def _inspect_tables(self) -> None:
        inspector = inspect(self.engine)
        if not inspector.has_table(self.fires_table, schema=self.schema):
            raise RuntimeError(
                f"Table {self.schema}.{self.fires_table} does not exist. Run the pipeline first."
            )
        columns = inspector.get_columns(self.fires_table, schema=self.schema)
        self._columns = {column["name"].lower(): column["name"] for column in columns}
        missing = [name for name in ("latitude", "longitude", "acq_date", "frp") if name not in self._columns]
        if missing:
            raise RuntimeError(f"{self.fires_table} is missing required columns: {', '.join(missing)}")
        self.available_columns = [name for name in FIRE_COLUMNS if name in self._columns]

        date_type = next(c["type"] for c in columns if c["name"].lower() == "acq_date")
        try:
            self._date_is_text = date_type.python_type is str
        except NotImplementedError:
            self._date_is_text = True
        self.has_weather = inspector.has_table(self.weather_table, schema=self.schema)

    # --------------------------------------------------------------- helpers
    def _q(self, identifier: str) -> str:
        return self._preparer.quote(identifier)

    def _col(self, name: str) -> str:
        return self._q(self._columns[name])

    def _table(self, name: str) -> str:
        return f"{self._q(self.schema)}.{self._q(name)}"

    @property
    def _day_expr(self) -> str:
        """SQL expression yielding the acquisition day."""
        if self._date_is_text:
            return f"SUBSTR({self._col('acq_date')}, 1, 10)"
        return f"CAST({self._col('acq_date')} AS DATE)"

    def _date_param(self, value: date):
        return value.isoformat() if self._date_is_text else value

    def _where(self, filters: FireFilters) -> tuple[str, dict, bool]:
        clauses: list[str] = []
        params: dict = {}
        if filters.start_date:
            clauses.append(f"{self._col('acq_date')} >= :start_date")
            params["start_date"] = self._date_param(filters.start_date)
        if filters.end_date:
            # Exclusive upper bound tolerates timestamps / "YYYY-MM-DD hh:mm:ss" text.
            clauses.append(f"{self._col('acq_date')} < :end_date_exclusive")
            params["end_date_exclusive"] = self._date_param(filters.end_date + timedelta(days=1))
        if filters.has_bbox:
            clauses.append(f"{self._col('latitude')} BETWEEN :min_lat AND :max_lat")
            clauses.append(f"{self._col('longitude')} BETWEEN :min_lon AND :max_lon")
            params.update(
                min_lat=filters.min_lat, max_lat=filters.max_lat,
                min_lon=filters.min_lon, max_lon=filters.max_lon,
            )
        expanding = False
        if filters.sources and "satellite_source" in self._columns:
            clauses.append(f"{self._col('satellite_source')} IN :sources")
            params["sources"] = list(filters.sources)
            expanding = True
        if filters.min_confidence is not None and "confidence" in self._columns:
            clauses.append(f"{self._col('confidence')} >= :min_confidence")
            params["min_confidence"] = filters.min_confidence
        if filters.min_frp is not None:
            clauses.append(f"{self._col('frp')} >= :min_frp")
            params["min_frp"] = filters.min_frp
        if filters.max_frp is not None:
            clauses.append(f"{self._col('frp')} <= :max_frp")
            params["max_frp"] = filters.max_frp
        if filters.daynight and "daynight" in self._columns:
            clauses.append(f"{self._col('daynight')} = :daynight")
            params["daynight"] = filters.daynight
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params, expanding

    def _statement(self, sql: str, expanding: bool):
        statement = text(sql)
        if expanding:
            statement = statement.bindparams(bindparam("sources", expanding=True))
        return statement

    def _order_by(self, order: str) -> str:
        time_col = f", {self._col('acq_time')} DESC" if "acq_time" in self._columns else ""
        time_col_asc = f", {self._col('acq_time')} ASC" if "acq_time" in self._columns else ""
        return {
            "acq_date_desc": f" ORDER BY {self._col('acq_date')} DESC{time_col}",
            "acq_date_asc": f" ORDER BY {self._col('acq_date')} ASC{time_col_asc}",
            "frp_desc": f" ORDER BY {self._col('frp')} DESC NULLS LAST",
            "frp_asc": f" ORDER BY {self._col('frp')} ASC NULLS LAST",
        }.get(order, f" ORDER BY {self._col('acq_date')} DESC{time_col}")

    # ------------------------------------------------------------- interface
    def health(self) -> dict:
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "ok": True,
            "backend": self.name,
            "database": self.db_config.get("dbname"),
            "host": self.db_config.get("host"),
            "schema": self.schema,
            "fires_table": self.fires_table,
            "weather_table": self.weather_table if self.has_weather else None,
        }

    def fires(self, filters: FireFilters) -> pd.DataFrame:
        selected = ", ".join(f"{self._col(name)} AS {self._q(name)}" for name in self.available_columns)
        where, params, expanding = self._where(filters)
        sql = (
            f"SELECT {selected} FROM {self._table(self.fires_table)}{where}"
            f"{self._order_by(filters.order)} LIMIT :limit OFFSET :offset"
        )
        params.update(limit=int(filters.limit), offset=int(filters.offset))
        with self.engine.connect() as connection:
            frame = pd.read_sql_query(self._statement(sql, expanding), connection, params=params)
        if "acq_date" in frame.columns:
            frame["acq_date"] = canonical_date(frame["acq_date"])
        return frame

    def count_fires(self, filters: FireFilters) -> int:
        where, params, expanding = self._where(filters)
        sql = f"SELECT COUNT(*) FROM {self._table(self.fires_table)}{where}"
        with self.engine.connect() as connection:
            return int(connection.execute(self._statement(sql, expanding), params).scalar_one())

    def fire_stats(self, filters: FireFilters, high_frp_threshold: float) -> dict:
        where, params, expanding = self._where(filters)
        params["high_frp"] = float(high_frp_threshold)
        table = self._table(self.fires_table)
        frp = self._col("frp")
        high_count = f"SUM(CASE WHEN {frp} >= :high_frp THEN 1 ELSE 0 END)"
        stats = empty_stats()

        with self.engine.connect() as connection:
            total_row = connection.execute(
                self._statement(
                    f"SELECT COUNT(*) AS total, AVG({frp}) AS avg_frp, MAX({frp}) AS max_frp, "
                    f"{high_count} AS high_frp_count FROM {table}{where}",
                    expanding,
                ),
                params,
            ).mappings().one()
            total = int(total_row["total"] or 0)
            stats.update(
                total=total,
                avg_frp=_num(total_row["avg_frp"]),
                max_frp=_num(total_row["max_frp"]),
                high_frp_count=int(total_row["high_frp_count"] or 0),
            )
            stats["high_frp_share"] = (stats["high_frp_count"] / total) if total else None
            if total == 0:
                return stats

            by_date = connection.execute(
                self._statement(
                    f"SELECT {self._day_expr} AS day, COUNT(*) AS count, AVG({frp}) AS avg_frp, "
                    f"MAX({frp}) AS max_frp, {high_count} AS high_frp_count "
                    f"FROM {table}{where} GROUP BY 1 ORDER BY 1",
                    expanding,
                ),
                params,
            ).mappings().all()
            stats["by_date"] = [
                {
                    "date": _day_string(row["day"]),
                    "count": int(row["count"]),
                    "avg_frp": _num(row["avg_frp"]),
                    "max_frp": _num(row["max_frp"]),
                    "high_frp_count": int(row["high_frp_count"] or 0),
                }
                for row in by_date
            ]

            if "satellite_source" in self._columns:
                source = self._col("satellite_source")
                by_source = connection.execute(
                    self._statement(
                        f"SELECT {source} AS source, COUNT(*) AS count, AVG({frp}) AS avg_frp, "
                        f"{high_count} AS high_frp_count FROM {table}{where} GROUP BY 1 ORDER BY 2 DESC",
                        expanding,
                    ),
                    params,
                ).mappings().all()
                stats["by_source"] = [
                    {
                        "source": row["source"],
                        "count": int(row["count"]),
                        "avg_frp": _num(row["avg_frp"]),
                        "high_frp_count": int(row["high_frp_count"] or 0),
                    }
                    for row in by_source
                ]

            if "daynight" in self._columns:
                daynight = self._col("daynight")
                by_daynight = connection.execute(
                    self._statement(
                        f"SELECT {daynight} AS daynight, COUNT(*) AS count FROM {table}{where} "
                        "GROUP BY 1 ORDER BY 2 DESC",
                        expanding,
                    ),
                    params,
                ).mappings().all()
                stats["by_daynight"] = [
                    {"daynight": row["daynight"], "count": int(row["count"])} for row in by_daynight
                ]
        return stats

    def date_range(self) -> tuple[date | None, date | None]:
        column = self._col("acq_date")
        with self.engine.connect() as connection:
            row = connection.execute(
                text(f"SELECT MIN({column}), MAX({column}) FROM {self._table(self.fires_table)}")
            ).one()
        parsed = canonical_date(pd.Series(list(row)))
        start, end = parsed.iloc[0], parsed.iloc[1]
        return (
            None if pd.isna(start) else start.date(),
            None if pd.isna(end) else end.date(),
        )

    def weather(self) -> pd.DataFrame:
        if not self.has_weather:
            return pd.DataFrame()
        with self.engine.connect() as connection:
            return pd.read_sql_query(text(f"SELECT * FROM {self._table(self.weather_table)}"), connection)

    def close(self) -> None:
        self.engine.dispose()


def _num(value):
    return None if value is None or pd.isna(value) else float(value)


def _day_string(value) -> str:
    if isinstance(value, str):
        return value[:10]
    if hasattr(value, "isoformat"):
        return value.isoformat()[:10]
    return str(value)[:10]
