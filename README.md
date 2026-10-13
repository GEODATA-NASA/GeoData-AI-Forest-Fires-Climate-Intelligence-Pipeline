# 🌍 GeoData AI: Forest Fires & Climate Intelligence Pipeline

An automated ETL (Extract, Transform, Load) pipeline that ingests, cleans, and stores satellite-based active fire detections (NASA FIRMS) together with climate indicators (NASA POWER) in a PostgreSQL/PostGIS spatial database, and trains a GeoAI model that classifies wildfire severity (high FRP) to support wildfire analysis and early-warning systems.

---

## ✨ Features

- Merges multiple NASA FIRMS archives (MODIS and VIIRS) into a single standardized fire dataset containing millions of records.
- Pulls daily meteorological parameters (humidity, precipitation, soil moisture) from the NASA POWER API.
- Cleans, validates, and standardizes data into analysis-ready CSV files.
- Efficiently bulk-loads results into PostgreSQL/PostGIS (`fire_events` and `weather_data` tables) for spatial queries.
- Trains a `RandomForestClassifier` to classify high-severity fires, using a proper temporal split to avoid data leakage.
- Saves the trained pipeline as a reusable artifact (`high_frp_classifier.joblib`).
- Runs end to end with a single command (`python main.py`) through a pipeline orchestrator.
- Idempotent execution: each stage skips work that has already been done (existing files, tables, or a saved model).
- Data-quality validation (`validate_fire_data`) that runs once and is skipped on later runs via a flag file.
- Structured logging to `logs/pipeline.log` for progress, warnings, and errors.
- Keeps all credentials in environment variables, with centralized configuration.
- **FastAPI service** (`api/`) that queries the database and serves fire detections, GeoJSON, statistics and **severity predictions** from the trained model, with request logging to `logs/api.log`.
- **Interactive map frontend** (`frontend/`, Leaflet.js) showing fire locations coloured by predicted severity, daily statistics and a click-to-predict tool.

---

## 🛠️ Technology Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 – 3.14 |
| ETL / Processing | Pandas, Requests |
| Database & Spatial | PostgreSQL, PostGIS, SQLAlchemy, Psycopg2 |
| Machine Learning | scikit-learn (RandomForestClassifier), joblib |
| API | FastAPI, Uvicorn, Pydantic |
| Deployment | Docker, Docker Compose (PostGIS image) |
| Frontend | Leaflet.js, vanilla JavaScript (served by the API) |

**Data sources**

- **[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)** (NASA LANCE): MODIS and VIIRS active fire archives
- **NASA POWER API**: daily humidity, precipitation, and soil moisture

---

## 🔄 Pipeline Overview

```text
 ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
 │   Extract    │ ──▶ │    Transform    │ ──▶ │       Load       │ ──▶ │      Model       │
 │ NASA FIRMS   │     │ Clean, merge,   │     │ PostgreSQL +     │     │ RandomForest     │
 │ NASA POWER   │     │ standardize     │     │ PostGIS          │     │ high-FRP         │
 │              │     │ → cleaned_data/ │     │ fire_events,     │     │ classifier       │
 │              │     │                 │     │ weather_data     │     │ → .joblib        │
 └──────────────┘     └─────────────────┘     └──────────────────┘     └──────────────────┘
```

All stages are coordinated by `main.py`, which runs them in order: `NASA_POWER.py` ➔ `transform.py` ➔ `load.py` ➔ `train_model.py`.

---

## 📂 Project Structure

```text
GEODATA
├─ .env                  # Environment variables (DB credentials) – never commit
├─ .env.example          # Template for environment configuration
├─ main.py               # Pipeline orchestrator (single entry point)
├─ config.py             # Centralized configuration loader
├─ requirements.txt
├─ logs/                 # Pipeline logs
│  └─ pipeline.log
├─ raw_data/             # Original datasets downloaded from NASA
│  ├─ fire_archive_J1V-C2_820252.csv
│  ├─ fire_archive_M-C61_820251.csv
│  ├─ fire_archive_SV-C2_820253.csv
│  └─ nasa_power_weather.csv
├─ cleaned_data/         # Cleaned, structured output
│  ├─ cleaned_master_fires.csv
│  ├─ cleaned_weather.csv
│  └─ .fire_validated.flag   # Marks that data-quality validation already passed
├─ integration/          # Extraction scripts (API fetchers)
│  └─ NASA_POWER.py
├─ transformation/       # Cleaning and standardization
│  └─ transform.py
├─ loadation/            # Database loading
│  └─ load.py
├─ modeling/             # Model training
│  ├─ train_model.py
│  └─ artifacts/
│     └─ high_frp_classifier.joblib
├─ Dockerfile            # Image for the API and the pipeline
├─ docker-compose.yml    # PostGIS + API (+ pipeline / scheduler profiles)
├─ scheduler.py          # Reruns the pipeline on an interval
├─ api/                  # FastAPI service
│  ├─ main.py            # App factory (uvicorn api.main:app)
│  ├─ settings.py        # API settings from .env
│  ├─ logging_config.py  # Request logging -> logs/api.log
│  ├─ predictor.py       # Loads the model, builds features, matches weather
│  ├─ weather_index.py   # Same-day nearest-station lookup (as in training)
│  ├─ repositories/      # Data access: PostgreSQL or CSV fallback
│  ├─ routers/           # /health, /api/v1/fires, /api/v1/weather, /api/v1/predict
│  ├─ tools_make_sample.py     # Build cleaned_data/fires_sample.csv for the CSV backend
│  └─ tools_create_indexes.py  # Create PostgreSQL indexes used by the API
├─ frontend/             # Leaflet map UI (index.html, app.js, style.css)
├─ tests/                # pytest suite for the API
└─ README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 – 3.14
- PostgreSQL 14+ with the PostGIS extension installed

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd GEODATA

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Prepare the database

```sql
CREATE DATABASE geodata_db;
\c geodata_db
CREATE EXTENSION IF NOT EXISTS postgis;
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```env
DB_NAME=geodata_db
DB_USER=postgres
DB_PASSWORD=your_actual_password
DB_HOST=localhost
DB_PORT=5432
```

### 4. Run the pipeline

Run the whole pipeline with a single command:

```bash
python main.py
```

`main.py` executes the stages in order:

1. `integration/NASA_POWER.py`: fetch weather data from NASA POWER
2. `transformation/transform.py`: clean and standardize all datasets
3. `loadation/load.py`: load into PostgreSQL/PostGIS
4. `modeling/train_model.py`: train the high-FRP classification model

You can still run any stage on its own, for example `python modeling/train_model.py`.

### Smart execution (skip logic)

Each stage checks whether its output already exists and skips the work if so, which saves time and avoids repeating expensive steps:

| Stage | Skipped when |
|---|---|
| Extraction | The weather data file already exists |
| Transformation | The cleaned CSV files already exist |
| Loading | The target tables already exist in the database |
| Training | The saved model already exists at `modeling/artifacts/high_frp_classifier.joblib` |

To force a stage to run again, delete its output (file, table, or saved model) and rerun `python main.py`.

### Data-quality checks

`validate_fire_data` checks the fire dataset the first time the pipeline runs. After a successful validation it writes the flag file `cleaned_data/.fire_validated.flag`, and later runs use that flag to skip the heavy checks. Delete the flag file to force revalidation.

### Logging

Pipeline progress, warnings, and errors are written to `logs/pipeline.log` instead of relying on plain `print` output. Check this file first when a run fails.

---

## 🤖 GeoAI Model

The model is a `RandomForestClassifier` that classifies **high FRP (Fire Radiative Power) among already detected fire events**. It does not predict fire occurrence, because `fire_events` contains detections rather than labelled no-fire observations.

**Approach**

- **Temporal split**: training and test sets are separated by time (not randomly), so the model is evaluated on future data relative to what it was trained on.
- **Weather matching**: each fire is matched to the nearest same-day weather record within 100 km.
- **Output**: the trained pipeline is saved to `modeling/artifacts/high_frp_classifier.joblib`.

**Data requirements**

Weather rows must contain `date`, `latitude`, and `longitude` as well as one or more of `RH2M`, `PRECTOTCORR`, and `GWETTOP`. The NASA POWER extraction script preserves these join keys; rerun extraction, transformation, and loading after updating an existing weather table.

**Two training modes**

| Mode | Setting | Trains on | Use when |
|---|---|---|---|
| Weather-matched (default) | `REQUIRE_WEATHER_MATCH=1` | Only detections with a same-day NASA POWER record within the radius | Weather coverage matches the fire data |
| Global | `REQUIRE_WEATHER_MATCH=0` | Every detection; missing weather is imputed | Weather coverage is much smaller than fire coverage |

With the shipped weather file (one station near Cairo, January 2025) the
default mode trains on roughly 400 detections from one region, and the model
then produces almost the same score everywhere else on Earth, because its
latitude/longitude splits all lie inside that region. The global mode trained
on a 300 000-row sample reaches ROC-AUC 0.88 (precision 0.75, recall 0.61 for
severe fires) on a chronological hold-out, with brightness as the main driver.
`modeling/artifacts/high_frp_classifier_global.joblib` is that model; point
`MODEL_PATH` at it to serve it. Every artifact now records its training extent,
metrics and feature importances; the API returns `in_training_extent=false`
for scores outside that extent and the frontend shows a caution.

**Configuration**

The default high-FRP threshold is 10 MW. Configure the following with environment variables if needed:

| Variable | Purpose |
|---|---|
| `HIGH_FRP_THRESHOLD` | FRP value (MW) above which a fire is labelled high severity |
| `MAX_WEATHER_DISTANCE_KM` | Maximum distance for matching a fire to a weather record |
| `MAX_TRAINING_ROWS` | Cap on the number of rows used for training |
| `REQUIRE_WEATHER_MATCH` | 1 (default) keeps only weather-matched fires; 0 trains on all fires |
| `TRAINING_SOURCE` | `db` (default) or `csv` to train from `cleaned_data/` without PostgreSQL |
| `MODEL_ARTIFACT_PATH` | Where to save the trained model |
| `RF_N_ESTIMATORS`, `RF_MIN_SAMPLES_LEAF`, `RF_MAX_DEPTH` | Forest size (200 / 20 / 18 by default; depth 0 = unlimited). Small leaves on large samples produce gigabyte artifacts |
| `DB_SCHEMA` | Database schema containing the tables |

---

## 🌐 API (FastAPI)

The API connects to the database created by the pipeline and serves the trained
model. Start it with:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Then open:

- `http://localhost:8000/` – interactive map frontend
- `http://localhost:8000/docs` – Swagger UI (OpenAPI)
- `http://localhost:8000/health` – backend, model and data status

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service status: data backend, model, weather coverage, fire date range |
| GET | `/api/v1/model` | Model metadata: features, target, thresholds |
| GET | `/api/v1/fires` | Paginated fire detections. Filters: `start_date`, `end_date`, `bbox=min_lon,min_lat,max_lon,max_lat`, `source` (repeatable), `min_confidence`, `min_frp`, `max_frp`, `daynight`, `order`, `limit`, `offset`. Add `predict=true` to attach predictions and `include_total=true` for a total count |
| GET | `/api/v1/fires/geojson` | Same filters, returned as a GeoJSON `FeatureCollection` with predicted severity per point (for maps) |
| GET | `/api/v1/fires/stats` | Totals and per-day / per-satellite / day-night aggregates for the filtered fires |
| GET | `/api/v1/fires/date-range` | Earliest and latest acquisition dates |
| GET | `/api/v1/weather` | NASA POWER observations used for enrichment |
| POST | `/api/v1/predict` | Score one detection (JSON body, see below) |
| POST | `/api/v1/predict/batch` | Score up to 5 000 detections at once |
| POST | `/api/v1/refresh` | Rebuild the weather index and date range after the pipeline loaded new data (they also expire after `API_CACHE_TTL_SECONDS`, default 300) |

If the model or the data backend is unavailable the service still starts:
`/health` reports `degraded` with the reason, the affected endpoints answer
503, and everything else keeps working (for example the map with
`predict=false`).

Example prediction:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 30.05, "longitude": 31.2, "acq_date": "2025-01-15", "acq_time": 1130,
       "brightness": 325.4, "confidence": 80, "scan": 1.1, "track": 1.0, "type": 0}'
```

```json
{
  "high_frp_probability": 0.1432,
  "is_high_frp": false,
  "predicted_label": "normal_frp",
  "decision_threshold": 0.5,
  "high_frp_threshold_mw": 10.0,
  "weather": {"matched": true, "supplied": false, "distance_km": 20.04,
              "rh2m": 44.26, "prectotcorr": 0.0, "gwettop": 0.16}
}
```

Weather features are matched exactly as during training: the nearest same-day
NASA POWER record within `MAX_WEATHER_DISTANCE_KM`. If none exists the
response reports `"matched": false` and the model's median imputer fills the
gap. Callers may also pass `rh2m`, `prectotcorr` and `gwettop` directly.

### Data backends

`DATA_BACKEND` in `.env` controls where the API reads from:

| Value | Behaviour |
|---|---|
| `auto` (default) | Use PostgreSQL when it is reachable, otherwise fall back to the cleaned CSV files |
| `postgres` | Require PostgreSQL (`fire_events`, `weather_data`) |
| `csv` | Serve from `cleaned_data/` without a database |

The CSV backend loads `cleaned_data/fires_sample.csv` when present (a uniform
random sample of the full dataset; build it with
`python -m api.tools_make_sample --rows 300000`), otherwise the first
`CSV_MAX_ROWS` rows of `cleaned_master_fires.csv`. `/health` reports
`is_subset: true` in that case.

For interactive queries on the full `fire_events` table, create the supporting
indexes once with `python -m api.tools_create_indexes`.

### Logging

Every request is logged with method, path, status, latency and a request id
(also returned in the `X-Request-ID` header) to `logs/api.log`, which rotates
at 5 MB. Uvicorn server logs go to the same file.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run the API on a tiny in-memory dataset through the CSV backend and
exercise the SQL repository against SQLite, so no database is required.

---

## 🗺️ Frontend (Leaflet map)

The API serves `frontend/` at `/`. The page shows fire detections for a date
range and area, coloured by predicted severity (or by probability, observed
FRP, or satellite), with:

- Filters for date range, satellite source, confidence, FRP, day/night and the
  current map view.
- Summary cards, a detections-per-day chart with the high-FRP share, and a
  per-satellite breakdown from `/api/v1/fires/stats`.
- Popups with the detection's attributes, predicted probability and whether
  weather was matched.
- A **Predict severity** panel: click the map to pick a location, adjust the
  detection values and get a live prediction from `/api/v1/predict`.

Basemaps (Esri dark/satellite, OpenStreetMap) need no API key. Set
`SERVE_FRONTEND=0` to run the API without the UI, and `CORS_ORIGINS` if the
frontend is hosted elsewhere.

---

## 🐳 Docker

The repository ships a single image that runs both the API and the pipeline,
plus a Compose stack with PostGIS.

```bash
cp .env.example .env          # set DB_PASSWORD at least
docker compose up -d          # PostGIS + API on http://localhost:8000
docker compose run --rm pipeline   # extract -> transform -> load -> train (needs raw_data/)
```

| Service | Role | Started by |
|---|---|---|
| `db` | PostgreSQL 16 + PostGIS 3.4, data in the `pgdata` volume | `docker compose up` |
| `api` | FastAPI + frontend; waits for the database, mounts `cleaned_data/`, `modeling/artifacts/`, `logs/` | `docker compose up` |
| `pipeline` | One-off `python main.py` run against the stack | `docker compose run --rm pipeline` |
| `scheduler` | Reruns the pipeline every `PIPELINE_INTERVAL_HOURS` | `docker compose --profile scheduled up -d` |

Until the pipeline has loaded the tables, the API serves from the cleaned CSV
files if they exist in `cleaned_data/` (see *Data backends*). Host ports are
set with `API_PORT` and `DB_PORT_HOST` in `.env`.

---

## ⏱️ Scheduled runs

`scheduler.py` reruns `main.py` on an interval without extra dependencies and
logs to `logs/scheduler.log`. Because every stage is idempotent, repeated runs
only redo work whose inputs changed.

```bash
PIPELINE_INTERVAL_HOURS=24 python scheduler.py      # or the Compose "scheduled" profile
```

| Variable | Purpose |
|---|---|
| `PIPELINE_INTERVAL_HOURS` | Hours between runs (default 24, fractions allowed) |
| `PIPELINE_RUN_ON_START` | Run immediately on start-up (default 1) |
| `PIPELINE_MAX_RUNS` | Stop after this many runs (unlimited by default) |

For a system scheduler, call the pipeline directly instead:

- cron: `0 3 * * * cd /opt/geodata && .venv/bin/python main.py >> logs/cron.log 2>&1`
- Windows Task Scheduler: action `python.exe main.py`, "Start in" set to the repository folder.

---

## ✅ Verify the Load

```sql
SELECT COUNT(*) FROM fire_events;
SELECT COUNT(*) FROM weather_data;

-- Quick spatial sanity check
SELECT ST_AsText(geom) FROM fire_events LIMIT 5;
```

---

## 🧯 Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `psycopg2` connection error | Check `.env` values and that PostgreSQL is running |
| `type "geometry" does not exist` | Run `CREATE EXTENSION postgis;` in the target database |
| Package install fails | Use Python 3.10 – 3.12 and a fresh virtual environment |
| A stage is skipped unexpectedly | Its output already exists; delete the file, table, or saved model to rerun it |
| Validation is not running again | Delete `cleaned_data/.fire_validated.flag` |
| Unclear failure | Check `logs/pipeline.log` |
| Model finds no matching weather | Ensure weather rows include `date`, `latitude`, `longitude`, and rerun extraction, transformation, and loading |
| API starts with `backend: csv` although PostgreSQL exists | Check `.env` credentials; with `DATA_BACKEND=auto` the API falls back to CSV when the connection fails (see `logs/api.log`) |
| API queries are slow on the full table | Run `python -m api.tools_create_indexes` once |
| Predictions report `"matched": false` | No NASA POWER record for that date within `MAX_WEATHER_DISTANCE_KM`; extend weather coverage (see roadmap) |

---

## 🗺️ Roadmap

**Completed**

- [x] Project structure, centralized configuration (`config.py`, `.env`), and dependency management
- [x] ETL pipeline for NASA FIRMS (MODIS and VIIRS) and NASA POWER data
- [x] PostgreSQL/PostGIS schema (`fire_events`, `weather_data`) and bulk loading
- [x] High-FRP `RandomForestClassifier` with temporal split, saved to `modeling/artifacts/high_frp_classifier.joblib`
- [x] Pipeline orchestrator (`main.py`) running all stages with one command
- [x] Idempotent execution with skip logic for existing files, tables, and the saved model
- [x] Data-quality checks (`validate_fire_data`) with a flag file to skip repeated validation
- [x] Structured logging to `logs/pipeline.log`
- [x] Version control cleanup and publishing of code and trained model to GitHub

**Next steps** (built on the current pipeline infrastructure)

- [x] Build a lightweight FastAPI service that connects to the database and serves predictions from the trained model
- [x] Extend logging to the API layer
- [x] Build a frontend with interactive maps (Leaflet.js) showing fire locations and predicted severity
- [x] Containerize the project with Docker
- [x] Schedule automated runs (`scheduler.py` / Compose `scheduled` profile, cron or Task Scheduler)
- [ ] Build a Power BI dashboard on top of the stored data

**Future ideas**

- [ ] RAG system on top of the stored data and project knowledge
- [ ] Extend NASA POWER weather coverage to match the full time range of the fire data, increasing the number of training samples and improving precision and recall
- [ ] Orchestrate runs with Airflow for retries, backfills and a run history

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

**Data attribution:** Fire detection data is obtained from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (NASA LANCE). Weather data comes from the NASA POWER API. Please refer to each provider's terms of use and citation requirements when reusing the data.
