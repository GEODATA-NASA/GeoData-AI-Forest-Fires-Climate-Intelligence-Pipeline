"""Severity prediction endpoints backed by the trained high-FRP classifier."""

import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.deps import PredictorDep, SettingsDep, WeatherIndexDep
from api.schemas import BatchPredictionRequest, BatchPredictionResponse, FireObservation, Prediction, WeatherUsed
from api.serialization import frame_to_records

LOGGER = logging.getLogger("geodata.api.predict")
router = APIRouter(prefix="/api/v1/predict", tags=["predictions"])


def _predict(observations: list[FireObservation], predictor, weather_index, settings, decision_threshold, echo_input):
    frame = pd.DataFrame([observation.model_dump() for observation in observations])
    frame["acq_date"] = pd.to_datetime(frame["acq_date"])
    results = predictor.predict_frame(frame, weather_index, decision_threshold)
    threshold = predictor.decision_threshold if decision_threshold is None else decision_threshold
    predictions = []
    for observation, record in zip(observations, frame_to_records(results)):
        predictions.append(
            Prediction(
                high_frp_probability=record["high_frp_probability"],
                is_high_frp=record["is_high_frp"],
                predicted_label=record["predicted_label"],
                decision_threshold=threshold,
                high_frp_threshold_mw=settings.high_frp_threshold_mw,
                in_training_extent=predictor.in_training_extent(observation.latitude, observation.longitude, observation.acq_date),
                weather=WeatherUsed(
                    matched=record.get("weather_matched", False),
                    supplied=record.get("weather_supplied", False),
                    distance_km=record.get("weather_distance_km"),
                    rh2m=record.get("rh2m"),
                    prectotcorr=record.get("prectotcorr"),
                    gwettop=record.get("gwettop"),
                ),
                input=observation.model_dump(mode="json") if echo_input else None,
            )
        )
    return predictions


@router.post("", response_model=Prediction, summary="Predict severity for one fire detection")
def predict_one(
    observation: FireObservation,
    predictor: PredictorDep,
    weather_index: WeatherIndexDep,
    settings: SettingsDep,
    decision_threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> Prediction:
    prediction = _predict([observation], predictor, weather_index, settings, decision_threshold, echo_input=True)[0]
    LOGGER.info(
        "Prediction at (%.4f, %.4f) on %s: p(high_frp)=%.3f label=%s weather_matched=%s",
        observation.latitude, observation.longitude, observation.acq_date,
        prediction.high_frp_probability, prediction.predicted_label, prediction.weather.matched,
    )
    return prediction


@router.post("/batch", response_model=BatchPredictionResponse, summary="Predict severity for many detections")
def predict_batch(
    payload: BatchPredictionRequest,
    predictor: PredictorDep,
    weather_index: WeatherIndexDep,
    settings: SettingsDep,
    decision_threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
    echo_input: Annotated[bool, Query(description="Include each input observation in its prediction")] = False,
) -> BatchPredictionResponse:
    if len(payload.observations) > settings.max_batch_predictions:
        raise HTTPException(
            status_code=413,
            detail=f"At most {settings.max_batch_predictions} observations per batch.",
        )
    predictions = _predict(payload.observations, predictor, weather_index, settings, decision_threshold, echo_input)
    LOGGER.info("Batch prediction for %s observations.", len(predictions))
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)
