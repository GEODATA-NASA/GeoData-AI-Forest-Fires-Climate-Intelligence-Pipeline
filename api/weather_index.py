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
    """Per-day BallTree indexes over weather observations."""

    def __init__(self, weather: pd.DataFrame, max_distance_km: float):
        self.weather = normalize_weather_frame(weather)
        self.max_distance_km = float(max_distance_km)
        self.features = [
            name
            for name in WEATHER_FEATURES
            if name in self.weather.columns and self.weather[name].notna().any()
        ]
        self._trees: dict[pd.Timestamp, tuple[BallTree, np.ndarray]] = {}
        for day, daily in self.weather.groupby("date", sort=True):
            coordinates = np.radians(daily[["latitude", "longitude"]].to_numpy(dtype=float))
            values = daily[self.features].to_numpy(dtype=float) if self.features else np.empty((len(daily), 0))
            self._trees[pd.Timestamp(day)] = (BallTree(coordinates, metric="haversine"), values)

    def __len__(self) -> int:
        return len(self.weather)

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
            "observations": len(self.weather),
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

        days = canonical_date(dates)
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
                result.loc[matched_index, feature] = values[nearest[within], column]
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
