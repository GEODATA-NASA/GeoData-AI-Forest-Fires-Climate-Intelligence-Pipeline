"""Service health and model metadata."""

import logging

from fastapi import APIRouter, Request

from api.schemas import DateRange, HealthResponse, ModelInfo
from api.version import __version__

LOGGER = logging.getLogger("geodata.api.health")
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service, database and model status")
def health(request: Request) -> HealthResponse:
    state = request.app.state
    repository = getattr(state, "repository", None)
    predictor = getattr(state, "predictor", None)
    weather_index = getattr(state, "weather_index", None)

    data_status: dict = {"ok": False, "backend": None}
    if repository is not None:
        try:
            data_status = repository.health()
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Data backend health check failed: %s", error)
            data_status = {"ok": False, "backend": repository.name, "error": str(error)}

    start, end = (None, None)
    if data_status.get("ok"):
        cached = getattr(state, "fire_date_range", None)
        if cached is None:
            try:
                cached = repository.date_range()
                state.fire_date_range = cached
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("Could not compute fire date range: %s", error)
                cached = (None, None)
        start, end = cached

    model_status = predictor.info() if predictor is not None else {"loaded": False}
    status = "ok" if data_status.get("ok") and model_status.get("loaded") else "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        backend=data_status.get("backend") or "none",
        data=data_status,
        model={"loaded": model_status.get("loaded", False), "estimator": model_status.get("estimator")},
        weather=weather_index.summary() if weather_index is not None else {"observations": 0},
        fire_date_range=DateRange(start_date=start, end_date=end),
    )


@router.get("/api/v1/model", response_model=ModelInfo, summary="Trained model metadata")
def model_info(request: Request) -> ModelInfo:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        settings = request.app.state.settings
        return ModelInfo(
            loaded=False, model_path=str(settings.model_path), estimator=None, n_estimators=None,
            feature_columns=[], weather_features=[], target=None,
            high_frp_threshold_mw=settings.high_frp_threshold_mw,
            decision_threshold=settings.decision_threshold,
        )
    return ModelInfo(**predictor.info())
