"""Run the pipeline on a fixed interval (a dependency-free alternative to cron).

Each tick runs ``main.py`` in a subprocess so a failing run never takes the
scheduler down. Configure with environment variables:

    PIPELINE_INTERVAL_HOURS   hours between runs (default 24; fractions allowed)
    PIPELINE_RUN_ON_START     1 to run immediately on start-up (default 1)
    PIPELINE_MAX_RUNS         stop after this many runs (default: unlimited)

Run it directly (``python scheduler.py``) or as the ``scheduler`` service in
docker-compose. For system schedulers use cron / Task Scheduler instead and
call ``python main.py`` directly (see README).
"""

import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "scheduler.log", encoding="utf-8"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("geodata.scheduler")

_stop = False


def _request_stop(signum, _frame):
    global _stop
    LOGGER.info("Received signal %s; stopping after the current run.", signum)
    _stop = True


def run_pipeline_once() -> int:
    started = time.monotonic()
    LOGGER.info("Starting pipeline run.")
    result = subprocess.run([sys.executable, str(ROOT_DIR / "main.py")], cwd=ROOT_DIR, check=False)
    elapsed = timedelta(seconds=round(time.monotonic() - started))
    if result.returncode == 0:
        LOGGER.info("Pipeline run finished successfully in %s.", elapsed)
    else:
        LOGGER.error("Pipeline run failed with exit code %s after %s. See logs/pipeline.log.", result.returncode, elapsed)
    return result.returncode


def main() -> int:
    interval_hours = float(os.getenv("PIPELINE_INTERVAL_HOURS", "24"))
    run_on_start = os.getenv("PIPELINE_RUN_ON_START", "1").lower() not in {"0", "false", "no"}
    max_runs = int(os.getenv("PIPELINE_MAX_RUNS", "0")) or None
    if interval_hours <= 0:
        LOGGER.error("PIPELINE_INTERVAL_HOURS must be positive.")
        return 2

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _request_stop)

    LOGGER.info("Scheduler started: every %s h, run on start=%s, max runs=%s.", interval_hours, run_on_start, max_runs or "unlimited")
    runs = 0
    next_run = datetime.now(timezone.utc) if run_on_start else datetime.now(timezone.utc) + timedelta(hours=interval_hours)
    while not _stop:
        now = datetime.now(timezone.utc)
        if now >= next_run:
            run_pipeline_once()
            runs += 1
            if max_runs and runs >= max_runs:
                LOGGER.info("Reached PIPELINE_MAX_RUNS=%s; exiting.", max_runs)
                break
            next_run = datetime.now(timezone.utc) + timedelta(hours=interval_hours)
            LOGGER.info("Next run at %s.", next_run.isoformat(timespec="minutes"))
        time.sleep(min(30, max(1, (next_run - datetime.now(timezone.utc)).total_seconds())))
    LOGGER.info("Scheduler stopped after %s run(s).", runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
