"""Download NASA POWER daily weather for the places and dates that have fires.

Two coverage modes, chosen with ``WEATHER_COVERAGE``:

``grid`` (default)
    Finds every place with fire detections, then downloads daily humidity
    (RH2M), precipitation (PRECTOTCORR) and surface soil wetness (GWETTOP) for
    those places over the full date range of the fire data. It uses the POWER
    *regional* endpoint, which returns a whole 10 x 10 degree tile on the native
    0.5 x 0.625 degree grid in one request, so the world's fire regions take a
    few hundred requests. Only grid points near fires are kept, thinned to
    ``WEATHER_GRID_LAT_STEP`` x ``WEATHER_GRID_LON_STEP`` degrees (default
    1.0 x 1.25, so every fire is within about 90 km of a point, inside the
    100 km matching radius used by training and the API).

    Raw tile responses are cached under ``raw_data/power_cache/`` so an
    interrupted download resumes where it stopped, and a manifest next to the
    output file records what was downloaded so later runs skip the work.

``point``
    The original behaviour: one location (``WEATHER_LATITUDE`` /
    ``WEATHER_LONGITUDE``) between ``WEATHER_START`` and ``WEATHER_END``.

The output keeps the original format (``date, RH2M, PRECTOTCORR, GWETTOP,
latitude, longitude``) so the rest of the pipeline is unchanged.
"""

import gzip
import json
import logging
import os
import shutil
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CLEANED_DATA_DIR, FIRE_FILES, RAW_DATA_DIR, WEATHER_FILE  # noqa: E402

LOGGER = logging.getLogger("geodata.integration.nasa_power")

PARAMETERS = ["RH2M", "PRECTOTCORR", "GWETTOP"]
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily"
NATIVE_LAT_STEP = 0.5
NATIVE_LON_STEP = 0.625
TILE_DEG = 10  # the regional endpoint accepts at most a 10 degree range per axis
FILL_VALUE = -999

WEATHER_COVERAGE = os.getenv("WEATHER_COVERAGE", "grid").strip().lower()
GRID_LAT_STEP = float(os.getenv("WEATHER_GRID_LAT_STEP", "1.0"))
GRID_LON_STEP = float(os.getenv("WEATHER_GRID_LON_STEP", "1.25"))
REQUEST_PAUSE_SECONDS = float(os.getenv("WEATHER_REQUEST_PAUSE", "1.0"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("WEATHER_REQUEST_TIMEOUT", "180"))
MAX_RETRIES = int(os.getenv("WEATHER_MAX_RETRIES", "6"))
CACHE_DIR = Path(os.getenv("WEATHER_CACHE_DIR", Path(RAW_DATA_DIR) / "power_cache"))
MANIFEST_FILE = Path(str(WEATHER_FILE).replace(".csv", ".manifest.json"))
LEGACY_BACKUP = Path(str(WEATHER_FILE).replace(".csv", "_point.csv"))
FIRE_SCAN_CHUNK = 1_000_000


# --------------------------------------------------------------------- helpers
def _compact(day: str | date) -> str:
    return pd.Timestamp(day).strftime("%Y%m%d")


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split a date range into calendar-year pieces (keeps each response small)."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    for year in range(start_ts.year, end_ts.year + 1):
        chunk_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        chunks.append((_compact(chunk_start), _compact(chunk_end)))
    return chunks


def _validate_steps() -> None:
    lat_ratio = GRID_LAT_STEP / NATIVE_LAT_STEP
    lon_ratio = GRID_LON_STEP / NATIVE_LON_STEP
    if GRID_LAT_STEP <= 0 or GRID_LON_STEP <= 0 or abs(lat_ratio - round(lat_ratio)) > 1e-9 or abs(lon_ratio - round(lon_ratio)) > 1e-9:
        raise ValueError(
            "WEATHER_GRID_LAT_STEP must be a multiple of 0.5 and WEATHER_GRID_LON_STEP a multiple of 0.625 "
            f"(got {GRID_LAT_STEP} and {GRID_LON_STEP})."
        )


def _fire_sources() -> list[Path]:
    cleaned = Path(CLEANED_DATA_DIR) / "cleaned_master_fires.csv"
    if cleaned.is_file():
        return [cleaned]
    raw = [Path(path) for path in FIRE_FILES.values() if Path(path).is_file()]
    if not raw:
        raise FileNotFoundError(
            "Grid weather needs the fire data to know where to download. "
            f"Neither {cleaned} nor the raw FIRMS archives were found."
        )
    return raw


def fire_footprint(paths: list[Path]) -> tuple[pd.DataFrame, str, str, int]:
    """Grid points nearest to any fire, plus the fire date range.

    Returns (points with columns lat_idx/lon_idx in thinned-grid units, start, end, rows scanned).
    """
    keys: set[int] = set()
    start = end = None
    scanned = 0
    for path in paths:
        LOGGER.info("Scanning fire locations in %s", path)
        for chunk in pd.read_csv(path, usecols=["latitude", "longitude", "acq_date"], chunksize=FIRE_SCAN_CHUNK):
            chunk = chunk.dropna()
            scanned += len(chunk)
            lat = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy()
            lon = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy()
            ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
            lat_idx = np.round(lat[ok] / GRID_LAT_STEP).astype(np.int64)
            lon_idx = np.round(lon[ok] / GRID_LON_STEP).astype(np.int64)
            # Wrap the antimeridian so +180 and -180 share a point.
            half_turn = int(round(180 / GRID_LON_STEP))
            lon_idx = np.where(lon_idx >= half_turn, lon_idx - 2 * half_turn, lon_idx)
            keys.update(np.unique(lat_idx * 100_000 + (lon_idx + 50_000)).tolist())
            dates = chunk["acq_date"].astype(str).str.slice(0, 10)
            chunk_min, chunk_max = dates.min(), dates.max()
            start = chunk_min if start is None or chunk_min < start else start
            end = chunk_max if end is None or chunk_max > end else end
            if scanned % 10_000_000 < FIRE_SCAN_CHUNK:
                LOGGER.info("  scanned %s rows, %s grid points so far", f"{scanned:,}", f"{len(keys):,}")
    if not keys:
        raise ValueError("No valid fire coordinates found.")
    packed = np.fromiter(keys, dtype=np.int64)
    lat_idx = np.floor_divide(packed, 100_000)
    lon_idx = packed - lat_idx * 100_000 - 50_000
    points = pd.DataFrame({"lat_idx": lat_idx, "lon_idx": lon_idx})
    return points, start, end, scanned


def tiles_for(points: pd.DataFrame) -> list[tuple[int, int]]:
    lat = points["lat_idx"] * GRID_LAT_STEP
    lon = points["lon_idx"] * GRID_LON_STEP
    tile_lat = np.clip(np.floor(lat / TILE_DEG) * TILE_DEG, -90, 90 - TILE_DEG).astype(int)
    tile_lon = np.clip(np.floor(lon / TILE_DEG) * TILE_DEG, -180, 180 - TILE_DEG).astype(int)
    return sorted(set(zip(tile_lat.tolist(), tile_lon.tolist())))


# -------------------------------------------------------------------- download
def _get_json(session: requests.Session, url: str, params: dict) -> dict:
    """GET with retries and exponential backoff on throttling and server errors."""
    delay = 5.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                LOGGER.warning("POWER answered %s (attempt %s/%s); waiting %.0f s.", response.status_code, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                delay = min(delay * 2, 300)
                continue
            raise RuntimeError(f"POWER request failed with HTTP {response.status_code}: {response.text[:300]}")
        except (requests.ConnectionError, requests.Timeout) as error:
            LOGGER.warning("POWER request error (attempt %s/%s): %s; waiting %.0f s.", attempt, MAX_RETRIES, error, delay)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError(f"POWER request kept failing after {MAX_RETRIES} attempts: {url} {params}")


def _tile_cache_path(parameter: str, tile: tuple[int, int], start: str, end: str) -> Path:
    return CACHE_DIR / parameter / f"{tile[0]:+04d}_{tile[1]:+05d}_{start}_{end}.json.gz"


def download_tile(session: requests.Session, parameter: str, tile: tuple[int, int], start: str, end: str) -> Path:
    path = _tile_cache_path(parameter, tile, start, end)
    if path.is_file() and path.stat().st_size > 0:
        return path
    params = {
        "latitude-min": tile[0], "latitude-max": tile[0] + TILE_DEG,
        "longitude-min": tile[1], "longitude-max": tile[1] + TILE_DEG,
        "parameters": parameter, "community": "AG", "start": start, "end": end,
        "format": "JSON", "time-standard": "UTC",
    }
    payload = _get_json(session, f"{POWER_BASE}/regional", params)
    if not payload.get("features"):
        raise RuntimeError(f"POWER returned no features for {parameter} tile {tile}: {str(payload)[:300]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(temp, path)
    time.sleep(REQUEST_PAUSE_SECONDS)
    return path


def _tile_frame(path: Path, parameter: str, keep: set[tuple[int, int]]) -> pd.DataFrame:
    """Rows for the kept grid points of one cached tile response."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = []
    for feature in payload["features"]:
        lon, lat = feature["geometry"]["coordinates"][:2]
        key = (int(round(lat / GRID_LAT_STEP)), int(round(lon / GRID_LON_STEP)))
        on_grid = abs(lat / GRID_LAT_STEP - key[0]) < 1e-6 and abs(lon / GRID_LON_STEP - key[1]) < 1e-6
        if not on_grid or key not in keep:
            continue
        series = feature["properties"]["parameter"][parameter]
        frames.append(pd.DataFrame({
            "date": np.fromiter((int(day) for day in series.keys()), dtype=np.int64, count=len(series)),
            parameter: np.fromiter(series.values(), dtype=np.float64, count=len(series)),
            "latitude": key[0] * GRID_LAT_STEP,
            "longitude": key[1] * GRID_LON_STEP,
        }))
    if not frames:
        return pd.DataFrame(columns=["date", parameter, "latitude", "longitude"])
    return pd.concat(frames, ignore_index=True)


def _manifest_matches(manifest: dict, expected: dict) -> bool:
    return all(manifest.get(key) == value for key, value in expected.items())


def download_grid_weather() -> str:
    _validate_steps()
    points, fire_start, fire_end, scanned = fire_footprint(_fire_sources())
    start = os.getenv("WEATHER_START", _compact(fire_start))
    end = os.getenv("WEATHER_END", _compact(fire_end))
    tiles = tiles_for(points)
    expected = {
        "mode": "grid", "parameters": PARAMETERS, "start": start, "end": end,
        "lat_step": GRID_LAT_STEP, "lon_step": GRID_LON_STEP, "points": int(len(points)),
    }

    if os.path.isfile(WEATHER_FILE) and MANIFEST_FILE.is_file():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if _manifest_matches(manifest, expected):
            LOGGER.info("Grid weather is already downloaded and matches the fire data: %s", WEATHER_FILE)
            return WEATHER_FILE
        LOGGER.info("Existing weather file covers a different area or period; downloading the difference.")

    if os.path.isfile(WEATHER_FILE) and not MANIFEST_FILE.is_file() and not LEGACY_BACKUP.exists():
        shutil.copy2(WEATHER_FILE, LEGACY_BACKUP)
        LOGGER.info("Kept the previous single-point weather file as %s", LEGACY_BACKUP)

    chunks = _year_chunks(start, end)
    total = len(tiles) * len(PARAMETERS) * len(chunks)
    LOGGER.info(
        "Grid weather: %s fire rows -> %s grid points in %s tiles, %s to %s; %s tile requests (cached ones are skipped).",
        f"{scanned:,}", f"{len(points):,}", len(tiles), start, end, total,
    )
    session = requests.Session()
    session.headers["User-Agent"] = "GeoData-AI-pipeline (+https://github.com/GEODATA-NASA)"
    done = 0
    started = time.monotonic()
    for tile in tiles:
        for parameter in PARAMETERS:
            for chunk_start, chunk_end in chunks:
                download_tile(session, parameter, tile, chunk_start, chunk_end)
                done += 1
                if done % 25 == 0 or done == total:
                    elapsed = time.monotonic() - started
                    LOGGER.info("  %s/%s tile requests done (%.0f s elapsed).", done, total, elapsed)

    LOGGER.info("Assembling the weather table from cached tiles.")
    keep = set(zip(points["lat_idx"].tolist(), points["lon_idx"].tolist()))
    pieces = []
    for tile in tiles:
        for chunk_start, chunk_end in chunks:
            merged = None
            for parameter in PARAMETERS:
                frame = _tile_frame(_tile_cache_path(parameter, tile, chunk_start, chunk_end), parameter, keep)
                merged = frame if merged is None else merged.merge(frame, on=["date", "latitude", "longitude"], how="outer")
            if merged is not None and not merged.empty:
                pieces.append(merged)
    weather = pd.concat(pieces, ignore_index=True)
    # Tiles share their edges; keep one row per point and day.
    weather = weather.drop_duplicates(subset=["date", "latitude", "longitude"]).sort_values(["date", "latitude", "longitude"])
    weather = weather[["date", *PARAMETERS, "latitude", "longitude"]]

    Path(WEATHER_FILE).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(WEATHER_FILE) + ".tmp")
    weather.to_csv(temp, index=False)
    os.replace(temp, WEATHER_FILE)
    manifest = {
        **expected,
        "tiles": len(tiles),
        "rows": int(len(weather)),
        "fire_rows_scanned": int(scanned),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{POWER_BASE}/regional",
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s weather rows for %s grid points to %s", f"{len(weather):,}", f"{len(points):,}", WEATHER_FILE)
    return WEATHER_FILE


def download_point_weather() -> str:
    """Original single-point download."""
    if os.path.isfile(WEATHER_FILE):
        if os.path.getsize(WEATHER_FILE) == 0:
            raise ValueError(f"Existing NASA POWER file is empty: {WEATHER_FILE}")
        LOGGER.info("NASA POWER weather data is already downloaded: %s", WEATHER_FILE)
        return WEATHER_FILE

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    latitude = float(os.getenv("WEATHER_LATITUDE", "30.0"))
    longitude = float(os.getenv("WEATHER_LONGITUDE", "31.0"))
    start_date = os.getenv("WEATHER_START", "20250101")
    end_date = os.getenv("WEATHER_END", "20250131")
    LOGGER.info("Fetching climate and rainfall data from NASA POWER for one point.")
    response = requests.get(
        f"{POWER_BASE}/point",
        params={
            "parameters": ",".join(PARAMETERS), "community": "AG",
            "longitude": longitude, "latitude": latitude,
            "start": start_date, "end": end_date, "format": "JSON",
        },
        timeout=60,
    )
    response.raise_for_status()
    df = pd.DataFrame(response.json()["properties"]["parameter"])
    df.index.name = "date"
    df = df.reset_index()
    df["latitude"] = latitude
    df["longitude"] = longitude
    df.to_csv(WEATHER_FILE, index=False)
    LOGGER.info("NASA POWER weather data saved to %s", WEATHER_FILE)
    return WEATHER_FILE


def download_weather_data():
    """Download NASA POWER weather according to WEATHER_COVERAGE (grid or point)."""
    if WEATHER_COVERAGE == "point":
        return download_point_weather()
    if WEATHER_COVERAGE != "grid":
        raise ValueError(f"WEATHER_COVERAGE must be 'grid' or 'point', not '{WEATHER_COVERAGE}'.")
    return download_grid_weather()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    download_weather_data()
