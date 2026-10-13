"""The service must keep answering when the model or the data backend is missing."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings


def _settings(data_dir, tmp_path, **overrides) -> Settings:
    base = dict(
        data_backend="csv",
        fires_csv=data_dir / "fires.csv",
        fires_sample_csv=data_dir / "missing_sample.csv",
        weather_csv=data_dir / "weather.csv",
        log_file=tmp_path / "api.log",
        serve_frontend=False,
    )
    base.update(overrides)
    return Settings(**base)


def test_missing_model_keeps_data_endpoints_alive(data_dir, tmp_path):
    settings = _settings(data_dir, tmp_path, model_path=Path(tmp_path) / "missing.joblib")
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["model"]["loaded"] is False and "missing.joblib" in health["model"]["error"]
        assert health["data"]["ok"] is True

        assert client.get("/api/v1/fires").json()["count"] == 6
        assert client.get("/api/v1/fires/geojson", params={"predict": False}).json()["metadata"]["count"] == 6
        assert client.get("/api/v1/fires/stats").json()["total"] == 6
        assert client.get("/api/v1/fires", params={"predict": True}).status_code == 503
        assert client.get("/api/v1/fires/geojson").status_code == 503
        assert client.post("/api/v1/predict", json={"latitude": 0, "longitude": 0, "acq_date": "2025-01-01"}).status_code == 503
        assert client.get("/api/v1/model").json()["loaded"] is False


def test_missing_data_backend_keeps_service_alive(tmp_path):
    settings = _settings(tmp_path, tmp_path, fires_csv=tmp_path / "nope.csv", weather_csv=tmp_path / "nope_w.csv")
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["backend"] == "none"
        assert "nope.csv" in health["data"]["error"]
        assert health["model"]["loaded"] is True
        assert client.get("/api/v1/fires").status_code == 503
        assert client.get("/api/v1/weather").status_code == 503
        # Predictions still work with imputed weather.
        prediction = client.post("/api/v1/predict", json={"latitude": 0, "longitude": 0, "acq_date": "2025-01-01"}).json()
        assert prediction["weather"]["matched"] is False
        assert client.get("/").status_code == 200  # JSON root when the frontend is disabled


def test_geojson_truncation_is_exact(client):
    assert client.get("/api/v1/fires/geojson", params={"limit": 6}).json()["metadata"]["truncated"] is False
    assert client.get("/api/v1/fires/geojson", params={"limit": 5}).json()["metadata"]["truncated"] is True


def test_refresh_rebuilds_caches(client):
    payload = client.post("/api/v1/refresh").json()
    assert payload["weather"]["observations"] == 3
    assert payload["fire_date_range"] == {"start_date": "2025-01-02", "end_date": "2025-03-16"}
    assert client.get("/health").json()["cache_ttl_seconds"] == 300


def test_unhandled_errors_return_json_with_request_id(client, monkeypatch):
    runtime = client.app.state.runtime
    monkeypatch.setattr(runtime.repository, "fire_stats", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    response = client.get("/api/v1/fires/stats", headers={"X-Request-ID": "rid-42"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error.", "request_id": "rid-42"}
    assert response.headers["X-Request-ID"] == "rid-42"


def test_conflicting_frp_bounds_are_rejected(client):
    assert client.get("/api/v1/fires", params={"min_frp": 10, "max_frp": 5}).status_code == 422
