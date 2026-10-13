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
from starlette.responses import JSONResponse

API_LOGGER_NAME = "geodata.api"
REQUEST_LOGGER = logging.getLogger(f"{API_LOGGER_NAME}.request")
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_FILE_HANDLER_NAME = "geodata-api-file"
_CONSOLE_HANDLER_NAME = "geodata-api-console"
# "uvicorn.error" propagates to "uvicorn"; "uvicorn.access" does not. Attaching
# to these two therefore captures every server record exactly once.
_LOGGER_NAMES = (API_LOGGER_NAME, "uvicorn", "uvicorn.access")


def _find_handler(logger: logging.Logger, name: str) -> logging.Handler | None:
    return next((h for h in logger.handlers if h.get_name() == name), None)


def configure_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """Attach one rotating file handler to the API and uvicorn loggers.

    Calling it again with the same path is a no-op; a different path replaces
    the handler (and closes the old file) so tests and multiple apps do not leak
    file descriptors.
    """
    log_file = Path(log_file).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT)

    api_logger = logging.getLogger(API_LOGGER_NAME)
    existing = _find_handler(api_logger, _FILE_HANDLER_NAME)
    if existing is not None and Path(getattr(existing, "baseFilename", "")).resolve() == log_file:
        file_handler = existing
    else:
        for name in _LOGGER_NAMES:
            logger = logging.getLogger(name)
            stale = _find_handler(logger, _FILE_HANDLER_NAME)
            if stale is not None:
                logger.removeHandler(stale)
                stale.close()
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.set_name(_FILE_HANDLER_NAME)

    for name in _LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if _find_handler(logger, _FILE_HANDLER_NAME) is None:
            logger.addHandler(file_handler)

    # Keep API logs out of logs/pipeline.log (configured on the root logger by
    # config.py) but still show them on the console.
    api_logger.propagate = False
    if _find_handler(api_logger, _CONSOLE_HANDLER_NAME) is None:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.set_name(_CONSOLE_HANDLER_NAME)
        api_logger.addHandler(console)
    return api_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status and latency for every request with a request id.

    Unhandled errors are logged here once and turned into a JSON 500 that still
    carries the request id headers, so clients can correlate failures.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        query = f"?{request.url.query}" if request.url.query else ""
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - last line of defence, logged with traceback
            elapsed_ms = (time.perf_counter() - started) * 1000
            REQUEST_LOGGER.exception(
                "%s %s%s -> 500 in %.1f ms [rid=%s]", request.method, request.url.path, query, elapsed_ms, request_id,
            )
            response = JSONResponse(status_code=500, content={"detail": "Internal server error.", "request_id": request_id})
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000
            REQUEST_LOGGER.info(
                "%s %s%s -> %s in %.1f ms [rid=%s]",
                request.method, request.url.path, query, response.status_code, elapsed_ms, request_id,
            )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
        return response
