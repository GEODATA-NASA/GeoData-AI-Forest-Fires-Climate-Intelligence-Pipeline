"""Same-day nearest-neighbour weather lookup.

Mirrors the matching rule used in ``modeling/train_model.py``: a fire is paired
with the closest NASA POWER observation from the same date, provided it is
within ``max_distance_km`` (haversine distance). Keeping the rule identical at
training and serving time avoids train/serve skew.
"""

from datetime import date

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from modeling.train_model import EARTH_RADIUS_KM, WEATHER_ALIASES, canonical_date

WEATHER_FEATURES = ["rh2m", "prectotcorr", "gwettop"]
_DATE_NAMES = ("date", "acq_date", "time", "date_key")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon", "lng")


def normalize_weather_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise a weather table to date/latitude/longitude + feature columns."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "latitude", "longitude", *WEATHER_FEATURES])

    columns = {str(name).strip().lower(): name for name in raw.columns}
    date_name = next((columns[n] for n in _DATE_NAMES if n in columns), None)
    lat_name = next((columns[n] for n in _LAT_NAMES if n in columns), None)
    lon_name = next((columns[n] for n in _LON_NAMES if n in columns), None)
    if date_name is None or lat_name is None or lon_name is None:
        raise ValueError(
            "Weather data needs date, latitude and longitude columns; "
            f"found: {', '.join(map(str, raw.columns))}"
        )

    frame = pd.DataFrame(
        {
            "date": canonical_date(raw[date_name]),
            "latitude": pd.to_numeric(raw[lat_name], errors="coerce"),
            "longitude": pd.to_numeric(raw[lon_name], errors="coerce"),
        }
    )
    for lower_name, original in columns.items():
        canonical = WEATHER_ALIASES.get(lower_name)
        if canonical and canonical not in frame.columns:
            frame[canonical] = pd.to_numeric(raw[original], errors="coerce").replace(-999, np.nan)

    frame = frame.dropna(subset=["date", "latitude", "longitude"])
    frame = frame.loc[
        frame["latitude"].between(-90, 90) & frame["longitude"].between(-180, 180)
    ]
    return frame.reset_index(drop=True)


class WeatherIndex:
    """Per-day BallTree indexes over weather observations.

    Only compact per-day arrays are kept (coordinates in degrees and feature
    values as float32); the input frame is not retained, so a global year of
    weather (millions of rows) stays a few hundred megabytes at most.
    """

    def __init__(self, weather: pd.DataFrame, max_distance_km: float):
        frame = normalize_weather_frame(weather)
        self.max_distance_km = float(max_distance_km)
        self.features = [
            name
            for name in WEATHER_FEATURES
            if name in frame.columns and frame[name].notna().any()
        ]
        self._rows = len(frame)
        self._trees: dict[pd.Timestamp, tuple[BallTree, np.ndarray]] = {}
        self._points: dict[pd.Timestamp, np.ndarray] = {}
        frame = frame.sort_values("date", kind="stable")
        day_values = frame["date"].to_numpy()
        coordinates_deg = frame[["latitude", "longitude"]].to_numpy(dtype=float)
        values_all = frame[self.features].to_numpy(dtype=np.float32) if self.features else np.empty((len(frame), 0), np.float32)
        boundaries = np.flatnonzero(day_values[1:] != day_values[:-1]) + 1
        for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(frame)]):
            if stop <= start:
                continue
            day = pd.Timestamp(day_values[start])
            self._trees[day] = (BallTree(np.radians(coordinates_deg[start:stop]), metric="haversine"), values_all[start:stop])
            self._points[day] = coordinates_deg[start:stop].astype(np.float32)

    def __len__(self) -> int:
        return self._rows

    def rows(self, start: date | None = None, end: date | None = None, limit: int = 5_000) -> pd.DataFrame:
        """Observations between two dates (inclusive), oldest first, at most ``limit`` rows."""
        pieces, remaining = [], int(limit)
        for day, (_tree, values) in self._trees.items():
            if remaining <= 0:
                break
            if (start and day.date() < start) or (end and day.date() > end):
                continue
            points = self._points[day][:remaining]
            piece = pd.DataFrame({"date": day, "latitude": points[:, 0].astype(float).round(4), "longitude": points[:, 1].astype(float).round(4)})
            for column, feature in enumerate(self.features):
                piece[feature] = values[: len(points), column].astype(float).round(3)
            pieces.append(piece)
            remaining -= len(piece)
        if not pieces:
            return pd.DataFrame(columns=["date", "latitude", "longitude", *self.features])
        return pd.concat(pieces, ignore_index=True)

    @property
    def dates(self) -> list[pd.Timestamp]:
        return list(self._trees.keys())

    @property
    def date_range(self) -> tuple[date | None, date | None]:
        if not self._trees:
            return None, None
        days = self.dates
        return days[0].date(), days[-1].date()

    def summary(self) -> dict:
        start, end = self.date_range
        return {
            "observations": self._rows,
            "points_per_day": int(max((len(v) for v in self._points.values()), default=0)),
            "dates": len(self._trees),
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "features": self.features,
            "max_distance_km": self.max_distance_km,
        }

    def match(self, dates: pd.Series, latitudes: pd.Series, longitudes: pd.Series) -> pd.DataFrame:
        """Return matched feature values and distance for each row (NaN when unmatched)."""
        index = dates.index
        result = pd.DataFrame(index=index)
        for feature in self.features:
            result[feature] = np.nan
        result["weather_distance_km"] = np.nan

        # Already-parsed timestamps skip the expensive string round-trip.
        days = dates.dt.normalize() if pd.api.types.is_datetime64_any_dtype(dates) else canonical_date(dates)
        lat = pd.to_numeric(latitudes, errors="coerce")
        lon = pd.to_numeric(longitudes, errors="coerce")
        usable = days.notna() & lat.notna() & lon.notna()
        if not usable.any() or not self._trees:
            result["weather_matched"] = False
            return result

        max_radians = self.max_distance_km / EARTH_RADIUS_KM
        for day, positions in days[usable].groupby(days[usable], sort=False).groups.items():
            tree_entry = self._trees.get(pd.Timestamp(day))
            if tree_entry is None:
                continue
            tree, values = tree_entry
            coordinates = np.radians(
                np.column_stack([lat.loc[positions].to_numpy(dtype=float), lon.loc[positions].to_numpy(dtype=float)])
            )
            distances, nearest = tree.query(coordinates, k=1)
            distances = distances[:, 0]
            nearest = nearest[:, 0]
            within = distances <= max_radians
            if not within.any():
                continue
            matched_index = pd.Index(positions)[within]
            for column, feature in enumerate(self.features):
                result.loc[matched_index, feature] = np.round(values[nearest[within], column].astype(float), 4)
            result.loc[matched_index, "weather_distance_km"] = distances[within] * EARTH_RADIUS_KM

        result["weather_matched"] = result["weather_distance_km"].notna()
        return result

    def nearest(self, day: date, latitude: float, longitude: float) -> dict | None:
        """Convenience single-point lookup."""
        frame = self.match(
            pd.Series([pd.Timestamp(day)]), pd.Series([latitude]), pd.Series([longitude])
        )
        row = frame.iloc[0]
        if not bool(row["weather_matched"]):
            return None
        payload = {feature: float(row[feature]) for feature in self.features}
        payload["distance_km"] = float(row["weather_distance_km"])
        return payload
