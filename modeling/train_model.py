"""Train and save a model that classifies high-FRP fire events.

The label is based on observed FRP, so this predicts event severity for an
already detected fire; it does not predict fire occurrence. Weather observations
must include date, latitude, and longitude to prevent invalid spatial/temporal
joins.

Environment switches (see README):

* ``REQUIRE_WEATHER_MATCH`` (default 1): keep only fires matched to a same-day
  weather record. Set to 0 to train on every detection; unmatched rows get NaN
  weather which the pipeline's imputer fills. Use this when weather coverage is
  much smaller than fire coverage, otherwise the model only learns one region.
* ``TRAINING_SOURCE`` (default db): ``csv`` trains from ``cleaned_data/`` without
  PostgreSQL (``FIRES_SAMPLE_FILE`` is used when present).
* ``MODEL_ARTIFACT_PATH``: where to save the model.
* ``RF_N_ESTIMATORS`` / ``RF_MIN_SAMPLES_LEAF`` / ``RF_MAX_DEPTH`` (200 / 20 / 18):
  forest size; ``RF_MAX_DEPTH=0`` means unlimited.
"""

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import URL, create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CLEANED_DATA_DIR, DB_CONFIG

LOGGER = logging.getLogger("geodata.model")
MODEL_ARTIFACT_PATH = Path(
    os.getenv(
        "MODEL_ARTIFACT_PATH",
        Path(__file__).resolve().parent / "artifacts" / "high_frp_classifier.joblib",
    )
)
REQUIRE_WEATHER_MATCH = os.getenv("REQUIRE_WEATHER_MATCH", "1").lower() not in {"0", "false", "no"}
TRAINING_SOURCE = os.getenv("TRAINING_SOURCE", "db").strip().lower()
# Forest size. Small leaves on hundreds of thousands of rows produce artifacts of
# a gigabyte or more; the defaults below keep the file in the tens of megabytes.
RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "200"))
RF_MIN_SAMPLES_LEAF = int(os.getenv("RF_MIN_SAMPLES_LEAF", "20"))
RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "18")) or None
EARTH_RADIUS_KM = 6371.0088
FIRE_CHUNK_SIZE = int(os.getenv("FIRE_CHUNK_SIZE", "100000"))
MAX_TRAINING_ROWS = int(os.getenv("MAX_TRAINING_ROWS", "500000"))
HIGH_FRP_THRESHOLD = float(os.getenv("HIGH_FRP_THRESHOLD", "10"))
MAX_WEATHER_DISTANCE_KM = float(os.getenv("MAX_WEATHER_DISTANCE_KM", "100"))

WEATHER_ALIASES = {
    "rh2m": "rh2m",
    "relative_humidity": "rh2m",
    "humidity": "rh2m",
    "prectotcorr": "prectotcorr",
    "precipitation": "prectotcorr",
    "precip": "prectotcorr",
    "gwettop": "gwettop",
    "soil_moisture": "gwettop",
}
FIRE_FEATURES = [
    "latitude",
    "longitude",
    "brightness",
    "confidence",
    "scan",
    "track",
    "type",
    "acq_time",
    "day_of_year",
]


def get_database_engine():
    """Create an SQLAlchemy engine using the project's existing DB settings."""
    url = URL.create(
        "postgresql+psycopg2",
        username=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=int(DB_CONFIG["port"]),
        database=DB_CONFIG["dbname"],
    )
    return create_engine(url, pool_pre_ping=True)


def quote_identifier(engine, identifier):
    return engine.dialect.identifier_preparer.quote(identifier)


def table_columns(engine, table_name, schema):
    columns = inspect(engine).get_columns(table_name, schema=schema)
    if not columns:
        raise ValueError(f"Table {schema}.{table_name} was not found or has no columns.")
    return {column["name"].lower(): column["name"] for column in columns}


def canonical_date(values):
    """Parse SQL dates and NASA POWER YYYYMMDD values consistently."""
    values = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact_dates = values.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(values.where(compact_dates), format="%Y%m%d", errors="coerce")
    parsed.loc[~compact_dates] = pd.to_datetime(values.loc[~compact_dates], errors="coerce")
    return parsed.dt.normalize()


def load_weather(engine, schema):
    """Load weather data and build per-day spatial nearest-neighbor indexes."""
    columns = table_columns(engine, "weather_data", schema)
    date_name = next(
        (columns[name] for name in ("date", "acq_date", "time", "date_key") if name in columns),
        None,
    )
    latitude_name = next(
        (columns[name] for name in ("latitude", "lat") if name in columns),
        None,
    )
    longitude_name = next(
        (columns[name] for name in ("longitude", "lon", "lng") if name in columns),
        None,
    )
    feature_names = {
        WEATHER_ALIASES[name]: actual_name
        for name, actual_name in columns.items()
        if name in WEATHER_ALIASES
    }

    missing_keys = [
        name
        for name, actual in (
            ("date", date_name),
            ("latitude", latitude_name),
            ("longitude", longitude_name),
        )
        if actual is None
    ]
    if missing_keys:
        raise ValueError(
            "weather_data is missing spatial/temporal join columns: "
            f"{', '.join(missing_keys)}. Store a date, latitude, and longitude "
            "with each NASA POWER observation before training. The current ETL "
            "drops the date index and does not store the requested coordinates."
        )
    if not feature_names:
        raise ValueError(
            "weather_data has no recognized climate columns. Expected one or more "
            "of RH2M, PRECTOTCORR, or GWETTOP."
        )

    selection = [
        (date_name, "weather_date"),
        (latitude_name, "weather_latitude"),
        (longitude_name, "weather_longitude"),
        *[(name, canonical) for canonical, name in feature_names.items()],
    ]
    selected_sql = ", ".join(
        f"{quote_identifier(engine, actual)} AS {quote_identifier(engine, canonical)}"
        for actual, canonical in selection
    )
    query = text(
        f"SELECT {selected_sql} FROM "
        f"{quote_identifier(engine, schema)}.{quote_identifier(engine, 'weather_data')}"
    )
    weather = pd.read_sql_query(query, engine)
    return build_weather_indexes(weather, list(feature_names.keys()))


def load_weather_csv(path):
    """CSV counterpart of ``load_weather`` for training without a database."""
    raw = pd.read_csv(path)
    columns = {str(name).strip().lower(): name for name in raw.columns}
    date_name = next((columns[n] for n in ("date", "acq_date", "time", "date_key") if n in columns), None)
    latitude_name = next((columns[n] for n in ("latitude", "lat") if n in columns), None)
    longitude_name = next((columns[n] for n in ("longitude", "lon", "lng") if n in columns), None)
    if None in (date_name, latitude_name, longitude_name):
        raise ValueError(f"{path} needs date, latitude and longitude columns.")
    weather = pd.DataFrame(
        {
            "weather_date": raw[date_name],
            "weather_latitude": raw[latitude_name],
            "weather_longitude": raw[longitude_name],
        }
    )
    features = []
    for lower, original in columns.items():
        canonical = WEATHER_ALIASES.get(lower)
        if canonical and canonical not in weather.columns:
            weather[canonical] = raw[original]
            features.append(canonical)
    if not features:
        raise ValueError(f"{path} has no recognized climate columns (RH2M, PRECTOTCORR, GWETTOP).")
    return build_weather_indexes(weather, features)


def build_weather_indexes(weather, feature_names):
    """Validate a canonical weather frame and build per-day BallTree indexes."""
    weather = weather.copy()
    weather["weather_date"] = canonical_date(weather["weather_date"])

    numeric_columns = ["weather_latitude", "weather_longitude", *feature_names]
    for column in numeric_columns:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    weather[numeric_columns] = weather[numeric_columns].replace(-999, np.nan)
    feature_names = [
        feature for feature in feature_names if weather[feature].notna().any()
    ]
    if not feature_names:
        raise ValueError("weather_data has no valid numeric climate observations.")
    valid_coordinates = (
        weather["weather_latitude"].between(-90, 90)
        & weather["weather_longitude"].between(-180, 180)
    )
    weather = weather.dropna(
        subset=["weather_date", "weather_latitude", "weather_longitude"]
    )
    weather = weather.loc[valid_coordinates].reset_index(drop=True)
    if weather.empty:
        raise ValueError("weather_data has no valid, dated weather locations.")

    # BallTree uses radians and haversine distance. Each event is matched only
    # to a weather observation from the same date.
    indexes = {}
    for date, daily_weather in weather.groupby("weather_date", sort=False):
        coordinates = np.radians(
            daily_weather[["weather_latitude", "weather_longitude"]].to_numpy()
        )
        indexes[date] = (
            BallTree(coordinates, metric="haversine"),
            daily_weather[list(feature_names)].to_numpy(dtype=float),
        )
    LOGGER.info(
        "Loaded %s weather observations across %s dates with features: %s",
        len(weather),
        len(indexes),
        ", ".join(feature_names),
    )
    return indexes, list(feature_names)


def load_fire_chunks(engine, schema):
    """Stream only the fire columns required for labels and predictors."""
    columns = table_columns(engine, "fire_events", schema)
    required = ("latitude", "longitude", "acq_date", "frp")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(
            f"fire_events is missing required columns: {', '.join(missing)}."
        )

    selected = list(required) + [
        name
        for name in ("brightness", "confidence", "scan", "track", "type", "acq_time")
        if name in columns
    ]
    selected_sql = ", ".join(
        f"{quote_identifier(engine, columns[name])} AS {quote_identifier(engine, name)}"
        for name in selected
    )
    query = text(
        f"SELECT {selected_sql} FROM "
        f"{quote_identifier(engine, schema)}.{quote_identifier(engine, 'fire_events')}"
    )
    return pd.read_sql_query(query, engine, chunksize=FIRE_CHUNK_SIZE)


def csv_fire_columns():
    """Column map for the CSV training source (mirrors ``table_columns``)."""
    header = pd.read_csv(csv_fire_path(), nrows=0).columns
    return {str(name).strip().lower(): name for name in header}


def csv_fire_path():
    sample = Path(os.getenv("FIRES_SAMPLE_FILE", Path(CLEANED_DATA_DIR) / "fires_sample.csv"))
    full = Path(os.getenv("FIRES_CSV_FILE", Path(CLEANED_DATA_DIR) / "cleaned_master_fires.csv"))
    path = sample if sample.is_file() else full
    if not path.is_file():
        raise FileNotFoundError(f"No cleaned fire CSV found: {sample} or {full}")
    return path


def load_fire_chunks_csv():
    columns = csv_fire_columns()
    required = ("latitude", "longitude", "acq_date", "frp")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Fire CSV is missing required columns: {', '.join(missing)}.")
    selected = [columns[n] for n in required] + [
        columns[n] for n in ("brightness", "confidence", "scan", "track", "type", "acq_time") if n in columns
    ]
    for chunk in pd.read_csv(csv_fire_path(), usecols=selected, chunksize=FIRE_CHUNK_SIZE, low_memory=False):
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        yield chunk


def match_weather(fires, weather_indexes, weather_features):
    """Attach nearest same-day weather values within the configured radius."""
    fires = fires.copy().reset_index(drop=True)
    fires["_date"] = canonical_date(fires["acq_date"])
    fires["latitude"] = pd.to_numeric(fires["latitude"], errors="coerce")
    fires["longitude"] = pd.to_numeric(fires["longitude"], errors="coerce")
    fires["frp"] = pd.to_numeric(fires["frp"], errors="coerce")
    fires = fires.dropna(
        subset=["_date", "latitude", "longitude", "frp"]
    ).reset_index(drop=True)
    fires = fires.loc[
        fires["latitude"].between(-90, 90)
        & fires["longitude"].between(-180, 180)
    ].reset_index(drop=True)
    if fires.empty:
        return fires

    weather_values = np.full((len(fires), len(weather_features)), np.nan)
    distance_km = np.full(len(fires), np.inf)
    max_distance_radians = MAX_WEATHER_DISTANCE_KM / EARTH_RADIUS_KM

    for date, positions in fires.groupby("_date", sort=False).groups.items():
        if date not in weather_indexes:
            continue
        tree, daily_values = weather_indexes[date]
        coordinates = np.radians(
            fires.loc[positions, ["latitude", "longitude"]].to_numpy()
        )
        distances, nearest = tree.query(coordinates, k=1)
        distances = distances[:, 0]
        nearest = nearest[:, 0]
        valid = distances <= max_distance_radians
        row_positions = np.asarray(list(positions), dtype=np.intp)[valid]
        weather_values[row_positions] = daily_values[nearest[valid]]
        distance_km[row_positions] = distances[valid] * EARTH_RADIUS_KM

    fires[weather_features] = weather_values
    fires["_weather_distance_km"] = distance_km
    if REQUIRE_WEATHER_MATCH:
        return fires.loc[np.isfinite(distance_km)].copy()
    return fires


def make_training_sample(engine, schema, weather_indexes, weather_features):
    """Stream fires, merge weather, and keep a uniform bounded random sample."""
    rng = np.random.default_rng(42)
    sample = pd.DataFrame()
    matched_rows = 0
    total_rows = 0
    if engine is None:
        fire_columns = csv_fire_columns()
        chunks = load_fire_chunks_csv()
    else:
        fire_columns = table_columns(engine, "fire_events", schema)
        chunks = load_fire_chunks(engine, schema)
    feature_columns = [
        column
        for column in FIRE_FEATURES
        if column == "day_of_year" or column in fire_columns
    ]
    for chunk in chunks:
        total_rows += len(chunk)
        merged = match_weather(chunk, weather_indexes, weather_features)
        matched_rows += int(np.isfinite(merged["_weather_distance_km"]).sum()) if len(merged) else 0
        if merged.empty:
            continue

        merged["day_of_year"] = merged["_date"].dt.dayofyear.astype(float)
        for column in feature_columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        feature_columns.extend(
            feature for feature in weather_features if feature not in feature_columns
        )

        selected = merged[feature_columns + ["_date", "frp", "_weather_distance_km"]].copy()
        selected = selected.replace([np.inf, -np.inf], np.nan)
        selected = selected.dropna(subset=["_date", "frp"])
        if selected.empty:
            continue
        selected["target"] = (selected["frp"] >= HIGH_FRP_THRESHOLD).astype("int8")
        selected = selected.drop(columns="frp")
        selected["_priority"] = rng.random(len(selected))
        sample = pd.concat([sample, selected], ignore_index=True)
        if len(sample) > MAX_TRAINING_ROWS:
            sample = sample.nsmallest(MAX_TRAINING_ROWS, "_priority")

    if total_rows == 0:
        raise ValueError("fire_events contains no rows.")
    match_rate = matched_rows / total_rows
    LOGGER.info(
        "Matched %s of %s fire rows to same-day weather within %.1f km (%.1f%%).",
        matched_rows,
        total_rows,
        MAX_WEATHER_DISTANCE_KM,
        match_rate * 100,
    )
    if sample.empty:
        raise ValueError(
            "No usable fire/weather rows were matched. Check weather dates and "
            "coordinates, then increase MAX_WEATHER_DISTANCE_KM if appropriate."
        )
    LOGGER.info(
        "Using %s rows for training (%s with matched weather); high-FRP threshold is %.2f MW.",
        len(sample),
        int(np.isfinite(sample["_weather_distance_km"]).sum()),
        HIGH_FRP_THRESHOLD,
    )
    return sample.drop(columns="_priority"), feature_columns


def training_extent(data):
    """Where and when the training rows come from; saved with the model so the
    API can warn when a prediction falls outside what the model has seen."""
    matched = np.isfinite(data["_weather_distance_km"])
    return {
        "rows": int(len(data)),
        "rows_with_weather": int(matched.sum()),
        "require_weather_match": REQUIRE_WEATHER_MATCH,
        "latitude_min": float(data["latitude"].min()),
        "latitude_max": float(data["latitude"].max()),
        "longitude_min": float(data["longitude"].min()),
        "longitude_max": float(data["longitude"].max()),
        "start_date": str(pd.Timestamp(data["_date"].min()).date()),
        "end_date": str(pd.Timestamp(data["_date"].max()).date()),
        "high_frp_share": float(data["target"].mean()),
    }


def temporal_split(data):
    """Split chronologically so later events are held out from training."""
    dates = np.sort(data["_date"].dropna().unique())
    if len(dates) < 2:
        raise ValueError("At least two distinct event dates are needed for a temporal split.")
    split_at = max(1, int(len(dates) * 0.8))
    if split_at >= len(dates):
        split_at = len(dates) - 1
    cutoff = dates[split_at]
    train = data.loc[data["_date"] < cutoff]
    test = data.loc[data["_date"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("The chronological train/test split produced an empty partition.")
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise ValueError(
            "Both classes must be present in the chronological train and test sets. "
            "Use more dates or a broader data period; do not randomly split these "
            "events because that can leak temporal patterns."
        )
    return train, test, cutoff


def train_and_evaluate(data, feature_columns):
    """Train, report metrics and feature importances, then save the pipeline."""
    train, test, cutoff = temporal_split(data)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                    max_depth=RF_MAX_DEPTH,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train[feature_columns], train["target"])
    predictions = model.predict(test[feature_columns])
    probabilities = model.predict_proba(test[feature_columns])[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(test["target"], predictions)),
        "precision": float(precision_score(test["target"], predictions, zero_division=0)),
        "recall": float(recall_score(test["target"], predictions, zero_division=0)),
        "f1": float(f1_score(test["target"], predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test["target"], probabilities)),
        "test_rows": int(len(test)),
        "train_rows": int(len(train)),
        "test_period_start": str(pd.Timestamp(cutoff).date()),
    }

    LOGGER.info("Chronological test period begins %s.", pd.Timestamp(cutoff).date())
    LOGGER.info(
        "Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
        accuracy_score(test["target"], predictions),
        precision_score(test["target"], predictions, zero_division=0),
        recall_score(test["target"], predictions, zero_division=0),
        f1_score(test["target"], predictions, zero_division=0),
        roc_auc_score(test["target"], probabilities),
    )
    LOGGER.info(
        "Classification report:\n%s",
        classification_report(
            test["target"],
            predictions,
            labels=[0, 1],
            target_names=["normal_frp", "high_frp"],
            zero_division=0,
        ),
    )
    LOGGER.info("Confusion matrix [normal_frp, high_frp]:\n%s", confusion_matrix(
        test["target"], predictions, labels=[0, 1]
    ))
    importances = model.named_steps["classifier"].feature_importances_
    importance_report = sorted(
        zip(feature_columns, importances), key=lambda item: item[1], reverse=True
    )
    LOGGER.info("Feature importance:")
    for feature, importance in importance_report:
        LOGGER.info("  %s: %.4f", feature, importance)

    output_path = MODEL_ARTIFACT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target": f"frp >= {HIGH_FRP_THRESHOLD} MW",
            "weather_features": [
                feature for feature in feature_columns if feature in WEATHER_ALIASES.values()
            ],
            "feature_importances": {feature: float(value) for feature, value in importance_report},
            "metrics": metrics,
            "training_extent": training_extent(data),
            "high_frp_threshold_mw": HIGH_FRP_THRESHOLD,
            "max_weather_distance_km": MAX_WEATHER_DISTANCE_KM,
            "hyperparameters": {
                "n_estimators": RF_N_ESTIMATORS,
                "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
                "max_depth": RF_MAX_DEPTH,
            },
        },
        output_path,
        compress=3,
    )
    LOGGER.info("Saved trained model to %s", output_path)


def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if MODEL_ARTIFACT_PATH.is_file():
        joblib.load(MODEL_ARTIFACT_PATH)
        LOGGER.info("Loaded existing model; ready at %s.", MODEL_ARTIFACT_PATH)
        return 0

    if FIRE_CHUNK_SIZE <= 0 or MAX_TRAINING_ROWS <= 0:
        LOGGER.error("FIRE_CHUNK_SIZE and MAX_TRAINING_ROWS must be positive integers.")
        return 2
    if HIGH_FRP_THRESHOLD < 0 or MAX_WEATHER_DISTANCE_KM <= 0:
        LOGGER.error("FRP threshold must be non-negative and weather distance must be positive.")
        return 2

    schema = os.getenv("DB_SCHEMA", "public")
    engine = None
    try:
        if TRAINING_SOURCE == "csv":
            LOGGER.info("Training from the cleaned CSV files (TRAINING_SOURCE=csv).")
            weather_csv = Path(os.getenv("WEATHER_CSV_FILE", Path(CLEANED_DATA_DIR) / "cleaned_weather.csv"))
            weather_indexes, weather_features = load_weather_csv(weather_csv)
        else:
            engine = get_database_engine()
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            weather_indexes, weather_features = load_weather(engine, schema)
        sample, feature_columns = make_training_sample(
            engine, schema, weather_indexes, weather_features
        )
        if sample["target"].nunique() < 2:
            raise ValueError(
                "The matched training sample contains only one target class. "
                "Check the FRP threshold or use a broader dataset."
            )
        train_and_evaluate(sample, feature_columns)
    except SQLAlchemyError:
        LOGGER.exception(
            "Database operation failed. Check PostgreSQL availability, credentials, "
            "DB_SCHEMA, and table names."
        )
        return 1
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        LOGGER.error("%s", error)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())