"""FastAPI application factory.

Run locally with::

    uvicorn api.main:app --reload

The app loads the trained classifier, connects to PostgreSQL (or falls back to
the cleaned CSV files), builds the same-day weather index, and serves the
Leaflet frontend from ``frontend/`` at ``/``.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.logging_config import RequestLoggingMiddleware, configure_logging  # noqa: E402
from api.predictor import SeverityPredictor  # noqa: E402
from api.repositories import create_repository  # noqa: E402
from api.routers import fires, health, predict, weather  # noqa: E402
from api.settings import Settings, load_settings  # noqa: E402
from api.version import __version__  # noqa: E402
from api.weather_index import WeatherIndex  # noqa: E402

DESCRIPTION = """
Serves NASA FIRMS fire detections and NASA POWER weather from the GeoData pipeline
database, plus **high-FRP severity predictions** from the trained RandomForest model.

* `GET /api/v1/fires` - filtered, paginated detections (optionally with predictions)
* `GET /api/v1/fires/geojson` - the same as GeoJSON for maps
* `GET /api/v1/fires/stats` - daily / per-satellite aggregates
* `POST /api/v1/predict` - score one detection (weather is matched automatically)
* `POST /api/v1/predict/batch` - score many detections
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    logger = configure_logging(settings.log_file, settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting GeoData API v%s (backend=%s).", __version__, settings.data_backend)
        app.state.settings = settings
        app.state.repository = None
        app.state.predictor = None
        app.state.weather_index = None
        app.state.fire_date_range = None

        app.state.repository = create_repository(settings)

        try:
            app.state.predictor = SeverityPredictor(
                settings.model_path, settings.high_frp_threshold_mw, settings.decision_threshold
            ).load()
        except Exception:  # noqa: BLE001
            logger.exception("Model could not be loaded; prediction endpoints will return 503.")

        try:
            weather_index = WeatherIndex(app.state.repository.weather(), settings.max_weather_distance_km)
            app.state.weather_index = weather_index
            logger.info("Weather index ready: %s", weather_index.summary())
        except Exception:  # noqa: BLE001
            logger.exception("Weather data could not be indexed; predictions will use imputed weather.")

        yield

        logger.info("Shutting down GeoData API.")
        if app.state.repository is not None:
            app.state.repository.close()

    app = FastAPI(
        title="GeoData AI - Forest Fires & Climate Intelligence API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled error on %s %s [rid=%s]", request.method, request.url.path, request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )

    app.include_router(health.router)
    app.include_router(fires.router)
    app.include_router(weather.router)
    app.include_router(predict.router)

    if settings.serve_frontend and settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def root():
            return {"name": app.title, "version": __version__, "docs": "/docs", "health": "/health"}

    return app


app = create_app()
