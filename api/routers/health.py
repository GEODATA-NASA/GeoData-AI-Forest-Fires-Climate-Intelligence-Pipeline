"""Service health, model metadata and cache refresh."""

import logging

from fastapi import APIRouter, Request

from api.deps import RuntimeDep
from api.schemas import DateRange, HealthResponse, ModelInfo, RefreshResponse
from api.version import __version__

LOGGER = logging.getLogger("geodata.api.health")
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service, database and model status")
def health(request: Request) -> HealthResponse:
    state = request.app.state
    runtime = getattr(state, "runtime", None)
    startup_errors = getattr(state, "startup_errors", {}) or {}
    settings = state.settings

    data_status: dict = {"ok": False, "backend": None, "error": startup_errors.get("data", "starting")}
    start = end = None
    weather_summary: dict = {"observations": 0}
    model_status: dict = {"loaded": False, "estimator": None, "error": startup_errors.get("model")}

    if runtime is not None:
        if runtime.repository is not None:
            try:
                data_status = runtime.repository.health()
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("Data backend health check failed: %s", error)
                data_status = {"ok": False, "backend": runtime.repository.name, "error": str(error)}
            if data_status.get("ok"):
                start, end = runtime.date_range()
                if runtime.date_range_error:
                    data_status["date_range_error"] = runtime.date_range_error
            index = runtime.weather_index()
            weather_summary = index.summary() if index is not None else {"observations": 0, "error": runtime.weather_error}
        if runtime.predictor is not None:
            info = runtime.predictor.info()
            model_status = {"loaded": info["loaded"], "estimator": info["estimator"]}

    status = "ok" if data_status.get("ok") and model_status.get("loaded") else "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        backend=data_status.get("backend") or "none",
        data=data_status,
        model=model_status,
        weather=weather_summary,
        fire_date_range=DateRange(start_date=start, end_date=end),
        cache_ttl_seconds=settings.cache_ttl_seconds,
    )


@router.get("/api/v1/model", response_model=ModelInfo, summary="Trained model metadata")
def model_info(request: Request) -> ModelInfo:
    runtime = getattr(request.app.state, "runtime", None)
    predictor = runtime.predictor if runtime is not None else None
    if predictor is None:
        settings = request.app.state.settings
        return ModelInfo(
            loaded=False, model_path=str(settings.model_path), estimator=None, n_estimators=None,
            feature_columns=[], weather_features=[], target=None,
            high_frp_threshold_mw=settings.high_frp_threshold_mw,
            decision_threshold=settings.decision_threshold,
        )
    info = predictor.info()
    if info.get("feature_importances"):
        info["feature_importances"] = dict(sorted(info["feature_importances"].items(), key=lambda item: -item[1]))
    return ModelInfo(**info)


@router.post("/api/v1/refresh", response_model=RefreshResponse, summary="Rebuild caches after new data was loaded")
def refresh(runtime: RuntimeDep) -> RefreshResponse:
    """Call this after a pipeline run so new dates and weather are served immediately.

    Caches also expire on their own after ``API_CACHE_TTL_SECONDS``.
    """
    LOGGER.info("Cache refresh requested.")
    return RefreshResponse(**runtime.refresh())
