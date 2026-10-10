# 🌍 GeoData AI: Forest Fires & Climate Intelligence Pipeline

An automated ETL (Extract, Transform, Load) pipeline that ingests, cleans, and stores satellite-based active fire detections (NASA FIRMS) together with climate and vegetation indicators (NASA POWER, Google Earth Engine) in a PostgreSQL/PostGIS spatial database, built to support wildfire analysis and early-warning systems.

---

## ✨ Features

- Merges multiple NASA FIRMS archives (MODIS and VIIRS) into a single standardized fire dataset.
- Pulls daily meteorological parameters (humidity, precipitation, soil moisture) from the NASA POWER API.
- Supports vegetation and temperature indicators (NDVI, LST) from Google Earth Engine.
- Cleans, validates, and standardizes data into analysis-ready CSV files.
- Loads results into PostgreSQL/PostGIS for spatial queries.
- Keeps all credentials in environment variables, with centralized configuration.

---

## 🛠️ Technology Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 – 3.12 |
| ETL / Processing | Pandas, Requests |
| Database & Spatial | PostgreSQL, PostGIS, SQLAlchemy, Psycopg2 |
| Geospatial / Remote Sensing | Earth Engine API, geemap |

**Data sources**

- **NASA FIRMS**: MODIS and VIIRS active fire archives
- **NASA POWER API**: daily humidity, precipitation, and soil moisture
- **Google Earth Engine**: MODIS NDVI and LST

---

## 🔄 Pipeline Overview

```text
 ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
 │   Extract    │ ──▶ │    Transform    │ ──▶ │       Load       │
 │ NASA FIRMS   │     │ Clean, merge,   │     │ PostgreSQL +     │
 │ NASA POWER   │     │ standardize     │     │ PostGIS          │
 │ Earth Engine │     │ → cleaned_data/ │     │                  │
 └──────────────┘     └─────────────────┘     └──────────────────┘
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

The model classifies **high FRP among already detected fire events**; it does not
predict fire occurrence because `fire_events` contains detections rather than
labelled no-fire observations. Weather rows must contain `date`, `latitude`, and
`longitude` as well as one or more of `RH2M`, `PRECTOTCORR`, and `GWETTOP`.
The NASA POWER extraction script preserves these join keys; rerun extraction,
transformation, and loading after updating an existing weather table.

The default high-FRP threshold is 10 MW, and each fire is matched to the nearest
same-day weather record within 100 km. Configure `HIGH_FRP_THRESHOLD`,
`MAX_WEATHER_DISTANCE_KM`, `MAX_TRAINING_ROWS`, and `DB_SCHEMA` with environment
variables if needed. The trained pipeline is saved to
`modeling/artifacts/high_frp_classifier.joblib`.

---

## ✅ Verify the Load

```sql
SELECT COUNT(*) FROM <fires_table>;

-- Quick spatial sanity check
SELECT ST_AsText(geom) FROM <fires_table> LIMIT 5;
```

Replace `<fires_table>` with your actual table name.

---

## 🧯 Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `psycopg2` connection error | Check `.env` values and that PostgreSQL is running |
| `type "geometry" does not exist` | Run `CREATE EXTENSION postgis;` in the target database |
| Earth Engine auth error | Re-run `earthengine authenticate` |
| Package install fails | Use Python 3.10 – 3.12 and a fresh virtual environment |

---

## 🗺️ Roadmap

- [ ] Add Earth Engine NDVI/LST ingestion to the main run flow
- [ ] Add data-quality checks and logging
- [ ] Schedule automated runs (cron / Airflow)
- [ ] Build wildfire risk prediction models on top of the stored data

---

## 📄 License

Prepared for hackathon project development. Add a license of your choice here.#
