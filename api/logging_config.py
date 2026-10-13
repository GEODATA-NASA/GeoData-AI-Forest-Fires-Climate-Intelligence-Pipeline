"""Logging for the API layer.

Extends the pipeline's structured logging: API events, uvicorn server logs and
one line per HTTP request go to ``logs/api.log`` (rotating) and to the console.
"""

import logging
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

API_LOGGER_NAME = "geodata.api"
REQUEST_LOGGER = logging.getLogger(f"{API_LOGGER_NAME}.request")
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_FILE_HANDLER_NAME = "geodata-api-file"
_CONSOLE_HANDLER_NAME = "geodata-api-console"


def configure_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """Attach a rotating file handler to the API and uvicorn loggers (idempotent)."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.set_name(_FILE_HANDLER_NAME)

    for name in (API_LOGGER_NAME, "uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not any(handler.get_name() == _FILE_HANDLER_NAME for handler in logger.handlers):
            logger.addHandler(file_handler)

    # Keep API logs out of logs/pipeline.log (configured on the root logger by
    # config.py) but still show them on the console.
    api_logger = logging.getLogger(API_LOGGER_NAME)
    api_logger.propagate = False
    if not any(handler.get_name() == _CONSOLE_HANDLER_NAME for handler in api_logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.set_name(_CONSOLE_HANDLER_NAME)
        api_logger.addHandler(console)
    return api_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status and latency for every request with a request id."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            REQUEST_LOGGER.exception(
                "%s %s -> 500 in %.1f ms [rid=%s]",
                request.method, request.url.path, elapsed_ms, request_id,
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        query = f"?{request.url.query}" if request.url.query else ""
        REQUEST_LOGGER.info(
            "%s %s%s -> %s in %.1f ms [rid=%s]",
            request.method, request.url.path, query, response.status_code, elapsed_ms, request_id,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
        return response
