# 🌍 GeoData AI: Forest Fires & Climate Intelligence Pipeline

An automated ETL (Extract, Transform, Load) pipeline that ingests, cleans, and stores satellite-based active fire detections (NASA FIRMS) together with climate and vegetation indicators (NASA POWER, Google Earth Engine) in a PostgreSQL/PostGIS spatial database, and trains a GeoAI model that classifies wildfire severity (high FRP) to support wildfire analysis and early-warning systems.

---

## ✨ Features

- Merges multiple NASA FIRMS archives (MODIS and VIIRS) into a single standardized fire dataset containing millions of records.
- Pulls daily meteorological parameters (humidity, precipitation, soil moisture) from the NASA POWER API.
- Supports vegetation and temperature indicators (NDVI, LST) from Google Earth Engine.
- Cleans, validates, and standardizes data into analysis-ready CSV files.
- Efficiently bulk-loads results into PostgreSQL/PostGIS (`fire_events` and `weather_data` tables) for spatial queries.
- Trains a `RandomForestClassifier` to classify high-severity fires, using a proper temporal split to avoid data leakage.
- Saves the trained pipeline as a reusable artifact (`high_frp_classifier.joblib`).
- Keeps all credentials in environment variables, with centralized configuration.

---

## 🛠️ Technology Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 – 3.12 |
| ETL / Processing | Pandas, Requests |
| Database & Spatial | PostgreSQL, PostGIS, SQLAlchemy, Psycopg2 |
| Geospatial / Remote Sensing | Earth Engine API, geemap |
| Machine Learning | scikit-learn (RandomForestClassifier), joblib |

**Data sources**

- **[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)** (NASA LANCE): MODIS and VIIRS active fire archives
- **NASA POWER API**: daily humidity, precipitation, and soil moisture
- **Google Earth Engine**: MODIS NDVI and LST

---

## 🔄 Pipeline Overview

```text
 ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
 │   Extract    │ ──▶ │    Transform    │ ──▶ │       Load       │ ──▶ │      Model       │
 │ NASA FIRMS   │     │ Clean, merge,   │     │ PostgreSQL +     │     │ RandomForest     │
 │ NASA POWER   │     │ standardize     │     │ PostGIS          │     │ high-FRP         │
 │ Earth Engine │     │ → cleaned_data/ │     │ fire_events,     │     │ classifier       │
 │              │     │                 │     │ weather_data     │     │ → .joblib        │
 └──────────────┘     └─────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## 📂 Project Structure

```text
GEODATA
├─ .env                  # Environment variables (DB credentials) – never commit
├─ .env.example          # Template for environment configuration
├─ config.py             # Centralized configuration loader
├─ requirements.txt
├─ raw_data/             # Original datasets downloaded from NASA
│  ├─ fire_archive_J1V-C2_820252.csv
│  ├─ fire_archive_M-C61_820251.csv
│  ├─ fire_archive_SV-C2_820253.csv
│  └─ nasa_power_weather.csv
├─ cleaned_data/         # Cleaned, structured output
│  ├─ cleaned_master_fires.csv
│  └─ cleaned_weather.csv
├─ integration/          # Extraction scripts (API fetchers)
│  └─ NASA_POWER.py
├─ transformation/       # Cleaning and standardization
│  └─ transform.py
├─ loadation/            # Database loading
│  └─ load.py
├─ modeling/             # Model training
│  ├─ model.py
│  └─ artifacts/
│     └─ high_frp_classifier.joblib
└─ README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 – 3.12
- PostgreSQL 14+ with the PostGIS extension installed
- (Optional) A Google Earth Engine account for NDVI/LST data

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

### 4. (Optional) Authenticate Google Earth Engine

```bash
earthengine authenticate
```

### 5. Run the pipeline

```bash
# Step 1: Fetch weather data from NASA POWER
python integration/NASA_POWER.py

# Step 2: Clean and standardize all datasets
python transformation/transform.py

# Step 3: Load into PostgreSQL/PostGIS
python loadation/load.py

# Step 4: Train the high-FRP classification model
python modeling/model.py
```

---

## 🤖 GeoAI Model

The model is a `RandomForestClassifier` that classifies **high FRP (Fire Radiative Power) among already detected fire events**. It does not predict fire occurrence, because `fire_events` contains detections rather than labelled no-fire observations.

**Approach**

- **Temporal split**: training and test sets are separated by time (not randomly), so the model is evaluated on future data relative to what it was trained on.
- **Weather matching**: each fire is matched to the nearest same-day weather record within 100 km.
- **Output**: the trained pipeline is saved to `modeling/artifacts/high_frp_classifier.joblib`.

**Data requirements**

Weather rows must contain `date`, `latitude`, and `longitude` as well as one or more of `RH2M`, `PRECTOTCORR`, and `GWETTOP`. The NASA POWER extraction script preserves these join keys; rerun extraction, transformation, and loading after updating an existing weather table.

**Configuration**

The default high-FRP threshold is 10 MW. Configure the following with environment variables if needed:

| Variable | Purpose |
|---|---|
| `HIGH_FRP_THRESHOLD` | FRP value (MW) above which a fire is labelled high severity |
| `MAX_WEATHER_DISTANCE_KM` | Maximum distance for matching a fire to a weather record |
| `MAX_TRAINING_ROWS` | Cap on the number of rows used for training |
| `DB_SCHEMA` | Database schema containing the tables |

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
| Earth Engine auth error | Re-run `earthengine authenticate` |
| Package install fails | Use Python 3.10 – 3.12 and a fresh virtual environment |
| Model finds no matching weather | Ensure weather rows include `date`, `latitude`, `longitude`, and rerun extraction, transformation, and loading |

---

## 🗺️ Roadmap

**Completed**

- [x] Project structure, centralized configuration (`config.py`, `.env`), and dependency management
- [x] ETL pipeline for NASA FIRMS (MODIS and VIIRS) and NASA POWER data
- [x] PostgreSQL/PostGIS schema (`fire_events`, `weather_data`) and bulk loading
- [x] High-FRP `RandomForestClassifier` with temporal split, saved to `modeling/artifacts/high_frp_classifier.joblib`
- [x] Version control cleanup and publishing of code and trained model to GitHub

**Next steps**

- [ ] Extend NASA POWER weather coverage to match the full time range of the fire data, increasing the number of training samples and improving precision and recall
- [ ] Build a lightweight Django web app that connects to the database and serves predictions from the trained model
- [ ] Add interactive maps (Leaflet.js / Folium) showing fire locations and predicted severity in the web app
- [ ] Add Earth Engine NDVI/LST ingestion to the main run flow
- [ ] Add data-quality checks and logging
- [ ] Schedule automated runs (cron / Airflow)

**Future ideas**

- [ ] RAG system on top of the stored data and project knowledge

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

**Data attribution:** Fire detection data is obtained from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (NASA LANCE). Weather data comes from the NASA POWER API. Please refer to each provider's terms of use and citation requirements when reusing the data.
