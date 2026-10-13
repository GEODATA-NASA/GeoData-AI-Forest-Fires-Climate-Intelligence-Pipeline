"""In-memory backend over the cleaned CSV files (used when PostgreSQL is unavailable).

It loads ``fires_sample.csv`` (see ``api/tools_make_sample.py``) when present,
otherwise the first ``max_rows`` rows of ``cleaned_master_fires.csv``.
"""

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from api.filters import FireFilters
from api.repositories.base import FIRE_COLUMNS, NUMERIC_FIRE_COLUMNS, FireRepository, empty_stats
from modeling.train_model import canonical_date

LOGGER = logging.getLogger("geodata.api.repository.csv")


class CsvFireRepository(FireRepository):
    name = "csv"

    def __init__(
        self,
        fires_file: Path,
        weather_file: Path,
        max_rows: int = 250_000,
        sample_file: Path | None = None,
        fires_frame: pd.DataFrame | None = None,
        weather_frame: pd.DataFrame | None = None,
    ):
        self.weather_file = Path(weather_file) if weather_file else None
        self._weather_frame = weather_frame
        self.max_rows = int(max_rows)

        if fires_frame is not None:
            self.source_file = "<in-memory>"
            frame = fires_frame.copy()
        else:
            candidates = [Path(path) for path in (sample_file, fires_file) if path]
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                raise FileNotFoundError(
                    "No fire data file found. Looked for: " + ", ".join(map(str, candidates))
                )
            self.source_file = str(source)
            LOGGER.info("Loading up to %s fire rows from %s", self.max_rows, source)
            frame = pd.read_csv(source, nrows=self.max_rows, low_memory=False)

        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing = [name for name in ("latitude", "longitude", "acq_date", "frp") if name not in frame.columns]
        if missing:
            raise ValueError(f"Fire data is missing required columns: {', '.join(missing)}")
        for column in NUMERIC_FIRE_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["acq_date"] = canonical_date(frame["acq_date"])
        frame = frame.dropna(subset=["latitude", "longitude", "acq_date"]).reset_index(drop=True)
        self.available_columns = [name for name in FIRE_COLUMNS if name in frame.columns]
        # The sample tool records how many rows it sampled from; use it to scale counts.
        self.sample_fraction = None
        meta_path = Path(str(self.source_file).replace(".csv", ".meta.json"))
        if fires_frame is None and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("source_rows"):
                    self.sample_fraction = round(len(frame) / float(meta["source_rows"]), 8)
            except (ValueError, OSError):
                LOGGER.warning("Could not read sample metadata %s", meta_path)
        self._fires = frame[self.available_columns]
        # True when we serve fewer rows than the full dataset (sample file or row cap hit).
        self.is_subset = fires_frame is None and (
            len(frame) >= self.max_rows or self.source_file == str(sample_file)
        )

    def __len__(self) -> int:
        return len(self._fires)

    # --------------------------------------------------------------- helpers
    def _mask(self, filters: FireFilters) -> pd.Series:
        fires = self._fires
        mask = pd.Series(True, index=fires.index)
        if filters.start_date:
            mask &= fires["acq_date"] >= pd.Timestamp(filters.start_date)
        if filters.end_date:
            mask &= fires["acq_date"] <= pd.Timestamp(filters.end_date)
        if filters.has_bbox:
            mask &= fires["latitude"].between(filters.min_lat, filters.max_lat)
            mask &= fires["longitude"].between(filters.min_lon, filters.max_lon)
        if filters.sources and "satellite_source" in fires.columns:
            mask &= fires["satellite_source"].isin(filters.sources)
        if filters.min_confidence is not None and "confidence" in fires.columns:
            mask &= fires["confidence"] >= filters.min_confidence
        if filters.min_frp is not None:
            mask &= fires["frp"] >= filters.min_frp
        if filters.max_frp is not None:
            mask &= fires["frp"] <= filters.max_frp
        if filters.daynight and "daynight" in fires.columns:
            mask &= fires["daynight"] == filters.daynight
        return mask

    def _sorted(self, frame: pd.DataFrame, order: str) -> pd.DataFrame:
        time_cols = ["acq_time"] if "acq_time" in frame.columns else []
        if order == "acq_date_asc":
            return frame.sort_values(["acq_date", *time_cols], ascending=True, kind="stable")
        if order == "frp_desc":
            return frame.sort_values("frp", ascending=False, na_position="last", kind="stable")
        if order == "frp_asc":
            return frame.sort_values("frp", ascending=True, na_position="last", kind="stable")
        return frame.sort_values(["acq_date", *time_cols], ascending=False, kind="stable")

    # ------------------------------------------------------------- interface
    def health(self) -> dict:
        return {
            "ok": True,
            "backend": self.name,
            "file": self.source_file,
            "rows": len(self._fires),
            "is_subset": bool(self.is_subset),
            "sample_fraction": self.sample_fraction,
        }

    def fires(self, filters: FireFilters) -> pd.DataFrame:
        selected = self._fires.loc[self._mask(filters)]
        selected = self._sorted(selected, filters.order)
        return selected.iloc[filters.offset: filters.offset + filters.limit].reset_index(drop=True)

    def count_fires(self, filters: FireFilters) -> int:
        return int(self._mask(filters).sum())

    def fire_stats(self, filters: FireFilters, high_frp_threshold: float) -> dict:
        selected = self._fires.loc[self._mask(filters)]
        stats = empty_stats()
        total = len(selected)
        if total == 0:
            return stats
        high = selected["frp"] >= high_frp_threshold
        stats.update(
            total=int(total),
            avg_frp=_num(selected["frp"].mean()),
            max_frp=_num(selected["frp"].max()),
            high_frp_count=int(high.sum()),
        )
        stats["high_frp_share"] = stats["high_frp_count"] / total

        grouped = selected.assign(_high=high.astype(int)).groupby(selected["acq_date"].dt.date, sort=True)
        stats["by_date"] = [
            {
                "date": day.isoformat(),
                "count": int(len(group)),
                "avg_frp": _num(group["frp"].mean()),
                "max_frp": _num(group["frp"].max()),
                "high_frp_count": int(group["_high"].sum()),
            }
            for day, group in grouped
        ]
        if "satellite_source" in selected.columns:
            by_source = selected.assign(_high=high.astype(int)).groupby("satellite_source", sort=False)
            stats["by_source"] = sorted(
                (
                    {
                        "source": source,
                        "count": int(len(group)),
                        "avg_frp": _num(group["frp"].mean()),
                        "high_frp_count": int(group["_high"].sum()),
                    }
                    for source, group in by_source
                ),
                key=lambda item: item["count"],
                reverse=True,
            )
        if "daynight" in selected.columns:
            counts = selected["daynight"].value_counts(dropna=True)
            stats["by_daynight"] = [
                {"daynight": str(key), "count": int(value)} for key, value in counts.items()
            ]
        return stats

    def date_range(self) -> tuple[date | None, date | None]:
        if self._fires.empty:
            return None, None
        return self._fires["acq_date"].min().date(), self._fires["acq_date"].max().date()

    def weather(self) -> pd.DataFrame:
        if self._weather_frame is not None:
            return self._weather_frame.copy()
        if self.weather_file and self.weather_file.is_file():
            return pd.read_csv(self.weather_file)
        LOGGER.warning("Weather file not found: %s", self.weather_file)
        return pd.DataFrame()


def _num(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value) else float(value)
