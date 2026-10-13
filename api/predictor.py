"""Serve predictions from the trained high-FRP classifier.

Feature construction follows ``modeling/train_model.py``: fire attributes from
FIRMS, ``day_of_year`` from the acquisition date, and same-day nearest weather
(humidity, precipitation, soil moisture) within the configured radius. Missing
values are handled by the median imputer inside the saved pipeline.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.weather_index import WEATHER_FEATURES, WeatherIndex
from modeling.train_model import canonical_date

LOGGER = logging.getLogger("geodata.api.predictor")

PREDICTION_COLUMNS = [
    "high_frp_probability",
    "is_high_frp",
    "predicted_label",
    "weather_matched",
    "weather_distance_km",
]


class SeverityPredictor:
    def __init__(self, model_path: Path, high_frp_threshold_mw: float, decision_threshold: float):
        self.model_path = Path(model_path)
        self.high_frp_threshold_mw = float(high_frp_threshold_mw)
        self.decision_threshold = float(decision_threshold)
        self.model = None
        self.feature_columns: list[str] = []
        self.weather_features: list[str] = []
        self.target_description: str | None = None
        self.feature_importances: dict | None = None
        self.metrics: dict | None = None
        self.training_extent: dict | None = None

    # ------------------------------------------------------------ lifecycle
    def load(self) -> "SeverityPredictor":
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.feature_columns = list(artifact["feature_columns"])
        self.weather_features = [
            name for name in artifact.get("weather_features", WEATHER_FEATURES) if name in self.feature_columns
        ]
        self.target_description = artifact.get("target")
        self.feature_importances = artifact.get("feature_importances") or self._importances_from_model()
        self.metrics = artifact.get("metrics")
        self.training_extent = artifact.get("training_extent")
        LOGGER.info(
            "Loaded model %s with %s features (%s).",
            self.model_path.name, len(self.feature_columns), ", ".join(self.feature_columns),
        )
        return self

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _importances_from_model(self) -> dict | None:
        """Older artifacts did not store importances; read them off the estimator."""
        final = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        values = getattr(final, "feature_importances_", None)
        if values is None:
            return None
        return {name: float(value) for name, value in zip(self.feature_columns, values)}

    def in_training_extent(self, latitude: float, longitude: float, day) -> bool | None:
        """None when the artifact carries no extent (model trained before it was recorded)."""
        extent = self.training_extent
        if not extent:
            return None
        try:
            inside = (
                extent["latitude_min"] <= float(latitude) <= extent["latitude_max"]
                and extent["longitude_min"] <= float(longitude) <= extent["longitude_max"]
            )
            if day is not None and extent.get("start_date") and extent.get("end_date"):
                inside = inside and extent["start_date"] <= str(pd.Timestamp(day).date()) <= extent["end_date"]
            return bool(inside)
        except (KeyError, TypeError, ValueError):
            return None

    def info(self) -> dict:
        estimator = None
        n_estimators = None
        if self.loaded:
            final = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
            estimator = type(final).__name__
            n_estimators = getattr(final, "n_estimators", None)
        return {
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "estimator": estimator,
            "n_estimators": n_estimators,
            "feature_columns": self.feature_columns,
            "weather_features": self.weather_features,
            "target": self.target_description,
            "high_frp_threshold_mw": self.high_frp_threshold_mw,
            "decision_threshold": self.decision_threshold,
            "feature_importances": self.feature_importances if self.loaded else None,
            "metrics": self.metrics if self.loaded else None,
            "training_extent": self.training_extent if self.loaded else None,
        }

    # ----------------------------------------------------------- inference
    def build_features(self, fires: pd.DataFrame, weather_index: WeatherIndex | None) -> pd.DataFrame:
        """Assemble the model's feature matrix plus weather-match diagnostics."""
        features = pd.DataFrame(index=fires.index)
        days = canonical_date(fires["acq_date"]) if "acq_date" in fires.columns else pd.Series(pd.NaT, index=fires.index)

        for column in self.feature_columns:
            if column == "day_of_year":
                features[column] = days.dt.dayofyear.astype(float)
            elif column in self.weather_features:
                continue
            elif column in fires.columns:
                features[column] = pd.to_numeric(fires[column], errors="coerce").astype(float)
            else:
                features[column] = np.nan

        # Weather: caller-supplied values win; otherwise match same-day nearest station.
        supplied = pd.DataFrame(index=fires.index)
        for feature in self.weather_features:
            supplied[feature] = (
                pd.to_numeric(fires[feature], errors="coerce").astype(float)
                if feature in fires.columns else np.nan
            )
        if weather_index is not None and len(weather_index) and "latitude" in fires.columns:
            matched = weather_index.match(days, fires["latitude"], fires["longitude"])
        else:
            matched = pd.DataFrame(index=fires.index)
            matched["weather_distance_km"] = np.nan
            matched["weather_matched"] = False

        for feature in self.weather_features:
            from_station = matched[feature] if feature in matched.columns else pd.Series(np.nan, index=fires.index)
            features[feature] = supplied[feature].where(supplied[feature].notna(), from_station)

        features["weather_matched"] = matched["weather_matched"].astype(bool)
        features["weather_distance_km"] = matched["weather_distance_km"]
        features["weather_supplied"] = supplied.notna().any(axis=1) if self.weather_features else False
        return features

    def predict_frame(
        self,
        fires: pd.DataFrame,
        weather_index: WeatherIndex | None,
        decision_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Return one prediction row per input row, aligned on the input index."""
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")
        threshold = self.decision_threshold if decision_threshold is None else float(decision_threshold)
        if fires.empty:
            return pd.DataFrame(columns=PREDICTION_COLUMNS + self.weather_features)

        features = self.build_features(fires, weather_index)
        matrix = features[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        probabilities = self.model.predict_proba(matrix)[:, 1]

        result = pd.DataFrame(index=fires.index)
        result["high_frp_probability"] = probabilities.round(4)
        result["is_high_frp"] = probabilities >= threshold
        result["predicted_label"] = np.where(result["is_high_frp"], "high_frp", "normal_frp")
        result["weather_matched"] = features["weather_matched"]
        result["weather_supplied"] = features["weather_supplied"]
        result["weather_distance_km"] = features["weather_distance_km"].round(2)
        for feature in self.weather_features:
            result[feature] = features[feature]
        return result
