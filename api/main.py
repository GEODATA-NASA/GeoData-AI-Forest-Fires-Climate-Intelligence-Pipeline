"""FastAPI application factory.

Run locally with::

    uvicorn api.main:app --reload

The app loads the trained classifier, connects to PostgreSQL (or falls back to
the cleaned CSV files), builds the same-day weather index, and serves the
Leaflet frontend from ``frontend/`` at ``/``. A missing backend or model does
not stop the service: the affected endpoints answer 503 and ``/health`` says why.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.logging_config import RequestLoggingMiddleware, configure_logging  # noqa: E402
from api.predictor import SeverityPredictor  # noqa: E402
from api.repositories import create_repository  # noqa: E402
from api.assistant.service import AssistantService  # noqa: E402
from api.routers import assistant, fires, health, predict, weather  # noqa: E402
from api.runtime import RuntimeData  # noqa: E402
from api.settings import Settings, load_settings  # noqa: E402
from api.version import __version__  # noqa: E402

DESCRIPTION = """
Serves NASA FIRMS fire detections and NASA POWER weather from the GeoData pipeline
database, plus **high-FRP severity predictions** from the trained RandomForest model.

* `GET /api/v1/fires` - filtered, paginated detections (optionally with predictions)
* `GET /api/v1/fires/geojson` - the same as GeoJSON for maps
* `GET /api/v1/fires/stats` - daily / per-satellite aggregates
* `POST /api/v1/predict` - score one detection (weather is matched automatically)
* `POST /api/v1/predict/batch` - score many detections
* `POST /api/v1/refresh` - rebuild caches after the pipeline loaded new data
* `POST /api/v1/ask` - ask a question in plain language; Claude answers using the data tools
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    logger = configure_logging(settings.log_file, settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting GeoData API v%s (backend=%s).", __version__, settings.data_backend)
        app.state.settings = settings
        app.state.startup_errors = {}

        repository = None
        try:
            repository = create_repository(settings)
        except Exception as error:  # noqa: BLE001 - keep serving /health and the frontend
            app.state.startup_errors["data"] = str(error)
            logger.exception("No data backend is available; data endpoints will return 503.")

        predictor = None
        try:
            predictor = SeverityPredictor(
                settings.model_path, settings.high_frp_threshold_mw, settings.decision_threshold
            ).load()
        except Exception as error:  # noqa: BLE001
            app.state.startup_errors["model"] = str(error)
            logger.exception("Model could not be loaded; prediction endpoints will return 503.")

        runtime = RuntimeData(settings, repository, predictor)
        app.state.runtime = runtime
        if repository is not None:
            runtime.weather_index()
            runtime.date_range()

        if getattr(app.state, "assistant", None) is None:  # tests may inject one
            try:
                app.state.assistant = AssistantService(
                    runtime,
                    model=settings.assistant_model,
                    effort=settings.assistant_effort,
                    max_rounds=settings.assistant_max_rounds,
                    enabled=settings.assistant_enabled,
                    use_fallbacks=settings.assistant_fallbacks,
                )
                logger.info("Assistant: %s", app.state.assistant.status())
            except Exception:  # noqa: BLE001 - the rest of the API must still start
                logger.exception("The assistant could not be initialised; /api/v1/ask will return 503.")
                app.state.assistant = None

        yield

        logger.info("Shutting down GeoData API.")
        runtime.close()

    app = FastAPI(
        title="GeoData AI - Forest Fires & Climate Intelligence API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.runtime = None
    app.state.assistant = None

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms"],
    )

    app.include_router(health.router)
    app.include_router(fires.router)
    app.include_router(weather.router)
    app.include_router(predict.router)
    app.include_router(assistant.router)

    if settings.serve_frontend and settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        def root():
            return {"name": app.title, "version": __version__, "docs": "/docs", "health": "/health"}

    return app


app = create_app()
