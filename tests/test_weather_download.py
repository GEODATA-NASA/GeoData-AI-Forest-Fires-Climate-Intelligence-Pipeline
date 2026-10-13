"""Grid weather downloader: footprint, tiling and tile parsing (no network)."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import BallTree

from integration import NASA_POWER as power


def _fires(tmp_path, rows):
    path = tmp_path / "fires.csv"
    pd.DataFrame(rows, columns=["latitude", "longitude", "acq_date"]).to_csv(path, index=False)
    return path


def test_footprint_snaps_to_thinned_grid_and_reads_dates(tmp_path):
    path = _fires(tmp_path, [
        (30.2, 31.3, "2025-01-05"),
        (30.4, 31.1, "2025-03-01"),     # same thinned point as the first
        (-33.9, 23.4, "2025-12-31"),    # southern hemisphere
        (10.0, 179.9, "2025-06-01"),    # wraps to -180
    ])
    points, start, end, scanned = power.fire_footprint([path])
    assert scanned == 4
    assert (start, end) == ("2025-01-05", "2025-12-31")
    coords = set(zip(points.lat_idx * power.GRID_LAT_STEP, points.lon_idx * power.GRID_LON_STEP))
    assert coords == {(30.0, 31.25), (-34.0, 23.75), (10.0, -180.0)}


def test_every_fire_is_within_the_matching_radius(tmp_path):
    rng = np.random.default_rng(0)
    lat = rng.uniform(-60, 70, 5000)
    lon = rng.uniform(-180, 180, 5000)
    path = _fires(tmp_path, list(zip(lat, lon, ["2025-01-01"] * len(lat))))
    points, *_ = power.fire_footprint([path])
    tree = BallTree(np.radians(np.column_stack([points.lat_idx * power.GRID_LAT_STEP, points.lon_idx * power.GRID_LON_STEP])), metric="haversine")
    distance, _ = tree.query(np.radians(np.column_stack([lat, lon])), k=1)
    assert distance.max() * 6371.0088 < 100


def test_tiles_are_aligned_and_clamped():
    points = pd.DataFrame({"lat_idx": [30, 90, -90, 5], "lon_idx": [25, 143, -144, -1]})
    tiles = power.tiles_for(points)
    assert (30, 30) in tiles          # lat 30, lon 31.25
    assert (80, 170) in tiles         # lat 90 clamps into the last valid tile
    assert (-90, -180) in tiles
    assert all(lat % 10 == 0 and lon % 10 == 0 for lat, lon in tiles)
    assert all(-90 <= lat <= 80 and -180 <= lon <= 170 for lat, lon in tiles)


def test_year_chunks_split_multi_year_ranges():
    assert power._year_chunks("2024-11-15", "2026-02-01") == [
        ("20241115", "20241231"), ("20250101", "20251231"), ("20260101", "20260201"),
    ]


def test_tile_frame_keeps_only_footprint_points(tmp_path):
    payload = {"features": [
        {"geometry": {"coordinates": [31.25, 30.0, 20.0]}, "properties": {"parameter": {"RH2M": {"20250101": 50.0, "20250102": 51.5}}}},
        {"geometry": {"coordinates": [31.875, 30.0, 20.0]}, "properties": {"parameter": {"RH2M": {"20250101": 1.0, "20250102": 2.0}}}},  # off the thinned grid
        {"geometry": {"coordinates": [32.5, 30.0, 20.0]}, "properties": {"parameter": {"RH2M": {"20250101": 3.0, "20250102": 4.0}}}},   # on grid, no fires
    ]}
    path = tmp_path / "tile.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    keep = {(30, 25)}  # (30.0, 31.25)
    frame = power._tile_frame(path, "RH2M", keep)
    assert frame.to_dict("records") == [
        {"date": 20250101, "RH2M": 50.0, "latitude": 30.0, "longitude": 31.25},
        {"date": 20250102, "RH2M": 51.5, "latitude": 30.0, "longitude": 31.25},
    ]


def test_invalid_grid_steps_are_rejected(monkeypatch):
    monkeypatch.setattr(power, "GRID_LON_STEP", 1.0)
    with pytest.raises(ValueError, match="multiple of 0.625"):
        power._validate_steps()


def test_unknown_coverage_mode_is_rejected(monkeypatch):
    monkeypatch.setattr(power, "WEATHER_COVERAGE", "regional")
    with pytest.raises(ValueError, match="grid' or 'point"):
        power.download_weather_data()
