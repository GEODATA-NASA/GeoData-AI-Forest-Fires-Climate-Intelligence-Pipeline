"""Orchestration task functions and the Airflow DAG structure."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from orchestration import tasks

DAG_FILE = Path(__file__).resolve().parents[1] / "orchestration" / "dags" / "geodata_pipeline.py"


def test_check_cleaned_outputs_fails_on_missing_files(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "CLEANED_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fires"):
        tasks.check_cleaned_outputs()
    (tmp_path / "cleaned_master_fires.csv").write_text("latitude\n1\n")
    (tmp_path / "cleaned_weather.csv").write_text("date\n20250101\n")
    summary = tasks.check_cleaned_outputs()
    assert summary["fires"]["exists"] and summary["weather"]["bytes"] > 0


def test_train_model_raises_on_failure(monkeypatch):
    import modeling.train_model as training

    monkeypatch.setattr(training, "main", lambda: 1)
    with pytest.raises(RuntimeError, match="status 1"):
        tasks.train_model(force=False)


def test_refresh_api_is_skipped_without_url(monkeypatch):
    monkeypatch.delenv("GEODATA_API_URL", raising=False)
    assert tasks.refresh_api() == {"refreshed": False, "reason": "GEODATA_API_URL not set"}


def test_refresh_api_never_fails_the_pipeline():
    result = tasks.refresh_api("http://127.0.0.1:9", timeout=2)  # nothing listens on port 9
    assert result["refreshed"] is False


def test_refresh_api_posts_to_the_refresh_endpoint():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["path"] = self.path
            body = json.dumps({"weather": {"observations": 3}, "fire_date_range": {"start_date": "2025-01-01", "end_date": "2025-12-31"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        result = tasks.refresh_api(f"http://127.0.0.1:{server.server_port}")
    finally:
        thread.join(timeout=5)
        server.server_close()
    assert seen["path"] == "/api/v1/refresh"
    assert result["refreshed"] is True and result["fire_date_range"]["end_date"] == "2025-12-31"


def test_dag_structure(monkeypatch):
    import os
    import sys

    if sys.platform == "win32" and not hasattr(os, "register_at_fork"):
        # Airflow only runs on POSIX; this lets the DAG be parsed (not run) on Windows.
        monkeypatch.setattr(os, "register_at_fork", lambda **kwargs: None, raising=False)
    pytest.importorskip("airflow")
    try:
        from airflow.models.dagbag import DagBag
    except ImportError:  # DagBag needs fcntl (POSIX); load the file directly instead
        DagBag = None

    if DagBag is not None:
        bag = DagBag(dag_folder=str(DAG_FILE), include_examples=False)
        assert not bag.import_errors, bag.import_errors
        dag = bag.get_dag("geodata_pipeline")
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location("geodata_pipeline_dag", DAG_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dag = module.pipeline_dag
    assert dag is not None and dag.dag_id == "geodata_pipeline"
    upstream = {task.task_id: sorted(task.upstream_task_ids) for task in dag.tasks}
    assert upstream == {
        "download_weather": [],
        "transform_fires": ["download_weather"],
        "transform_weather": ["download_weather"],
        "check_cleaned_outputs": ["transform_fires", "transform_weather"],
        "load_database": ["check_cleaned_outputs"],
        "train_model": ["load_database"],
        "refresh_api": ["train_model"],
    }
    assert dag.max_active_runs == 1 and dag.catchup is False
