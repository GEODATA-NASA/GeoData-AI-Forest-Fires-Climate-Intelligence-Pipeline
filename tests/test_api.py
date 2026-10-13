"""End-to-end tests for the FastAPI service using the CSV backend."""


def test_health_reports_backend_model_and_ranges(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["backend"] == "csv"
    assert payload["model"]["loaded"] is True
    assert payload["fire_date_range"] == {"start_date": "2025-01-02", "end_date": "2025-03-16"}
    assert payload["weather"]["features"] == ["rh2m", "prectotcorr", "gwettop"]


def test_model_info_lists_training_features(client):
    payload = client.get("/api/v1/model").json()
    assert payload["loaded"] is True
    assert payload["feature_columns"][:2] == ["latitude", "longitude"]
    assert "day_of_year" in payload["feature_columns"]


def test_list_fires_filters_and_orders(client):
    response = client.get(
        "/api/v1/fires",
        params={"start_date": "2025-01-01", "end_date": "2025-01-31", "include_total": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert [item["acq_date"] for item in payload["items"]] == ["2025-01-05", "2025-01-02", "2025-01-02"]

    response = client.get("/api/v1/fires", params={"source": ["modis"], "min_confidence": 90, "order": "frp_desc"})
    items = response.json()["items"]
    assert [item["frp"] for item in items] == [250.0, 45.0]

    response = client.get("/api/v1/fires", params={"bbox": "-121,44,-119,46"})
    assert response.json()["count"] == 2

    response = client.get("/api/v1/fires", params={"daynight": "N", "limit": 1, "offset": 1})
    payload = response.json()
    assert payload["count"] == 1 and payload["offset"] == 1


def test_list_fires_can_attach_predictions(client):
    payload = client.get("/api/v1/fires", params={"predict": True, "end_date": "2025-01-31"}).json()
    for item in payload["items"]:
        assert 0.0 <= item["high_frp_probability"] <= 1.0
        assert item["predicted_label"] in {"high_frp", "normal_frp"}
        assert item["weather_matched"] is True  # Cairo points, January, station within 100 km


def test_invalid_filters_are_rejected(client):
    assert client.get("/api/v1/fires", params={"bbox": "1,2,3"}).status_code == 422
    assert client.get("/api/v1/fires", params={"bbox": "10,5,-10,6"}).status_code == 422
    assert client.get("/api/v1/fires", params={"start_date": "2025-02-01", "end_date": "2025-01-01"}).status_code == 422
    assert client.get("/api/v1/fires", params={"order": "random"}).status_code == 422
    assert client.get("/api/v1/fires", params={"daynight": "X"}).status_code == 422


def test_geojson_feature_collection(client):
    payload = client.get("/api/v1/fires/geojson", params={"limit": 4}).json()
    assert payload["type"] == "FeatureCollection"
    assert payload["metadata"]["count"] == 4
    assert payload["metadata"]["truncated"] is True
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert set(feature["properties"]) >= {"frp", "acq_date", "high_frp_probability", "is_high_frp"}


def test_stats_aggregate_by_day_and_source(client):
    payload = client.get("/api/v1/fires/stats").json()
    assert payload["total"] == 6
    assert payload["high_frp_count"] == 3  # frp >= 10: 45, 120, 250
    assert payload["high_frp_share"] == 0.5
    assert [row["date"] for row in payload["by_date"]] == ["2025-01-02", "2025-01-05", "2025-03-15", "2025-03-16"]
    by_source = {row["source"]: row["count"] for row in payload["by_source"]}
    assert by_source == {"modis": 4, "viirs_j1": 1, "viirs_suomi": 1}
    assert {row["daynight"]: row["count"] for row in payload["by_daynight"]} == {"D": 3, "N": 3}


def test_date_range_endpoint(client):
    assert client.get("/api/v1/fires/date-range").json() == {"start_date": "2025-01-02", "end_date": "2025-03-16"}


def test_weather_listing_with_date_filter(client):
    payload = client.get("/api/v1/weather", params={"start_date": "2025-01-02"}).json()
    assert payload["count"] == 2
    assert payload["items"][0]["date"] == "2025-01-02"
    assert payload["items"][0]["rh2m"] == 71.2


def test_predict_matches_same_day_weather(client):
    body = {"latitude": 30.05, "longitude": 31.2, "acq_date": "2025-01-05", "acq_time": 1100,
            "brightness": 340, "confidence": 100, "scan": 0.4, "track": 0.4, "type": 0}
    payload = client.post("/api/v1/predict", json=body).json()
    assert 0.0 <= payload["high_frp_probability"] <= 1.0
    assert payload["predicted_label"] in {"high_frp", "normal_frp"}
    assert payload["weather"]["matched"] is True
    assert payload["weather"]["rh2m"] == 48.95
    assert payload["weather"]["distance_km"] < 100
    assert payload["input"]["acq_date"] == "2025-01-05"


def test_predict_without_weather_station_is_imputed(client):
    body = {"latitude": -33.9, "longitude": 23.4, "acq_date": "2025-03-15", "brightness": 318}
    payload = client.post("/api/v1/predict", json=body).json()
    assert payload["weather"]["matched"] is False
    assert payload["weather"]["rh2m"] is None


def test_predict_uses_caller_supplied_weather(client):
    body = {"latitude": 30.0, "longitude": 31.0, "acq_date": "2025-01-02", "rh2m": 15, "prectotcorr": 0, "gwettop": 0.05}
    payload = client.post("/api/v1/predict", json=body).json()
    assert payload["weather"]["supplied"] is True
    assert payload["weather"]["rh2m"] == 15


def test_predict_validation_and_threshold(client):
    assert client.post("/api/v1/predict", json={"latitude": 95, "longitude": 0, "acq_date": "2025-01-01"}).status_code == 422
    assert client.post("/api/v1/predict", json={"latitude": 0, "longitude": 0}).status_code == 422
    body = {"latitude": 30.0, "longitude": 31.0, "acq_date": "2025-01-02"}
    low = client.post("/api/v1/predict", params={"decision_threshold": 0.0}, json=body).json()
    high = client.post("/api/v1/predict", params={"decision_threshold": 1.0}, json=body).json()
    assert low["is_high_frp"] is True and high["is_high_frp"] is False


def test_batch_prediction(client):
    observations = [
        {"latitude": 30.0, "longitude": 31.0, "acq_date": "2025-01-02"},
        {"latitude": 45.0, "longitude": -120.0, "acq_date": "2025-03-15", "brightness": 360, "confidence": 95},
    ]
    payload = client.post("/api/v1/predict/batch", params={"echo_input": True}, json={"observations": observations}).json()
    assert payload["count"] == 2
    assert payload["predictions"][0]["weather"]["matched"] is True
    assert payload["predictions"][1]["weather"]["matched"] is False
    assert payload["predictions"][1]["input"]["latitude"] == 45.0
    assert client.post("/api/v1/predict/batch", json={"observations": []}).status_code == 422


def test_requests_carry_request_id_and_timing_headers(client):
    response = client.get("/health", headers={"X-Request-ID": "test-123"})
    assert response.headers["X-Request-ID"] == "test-123"
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "GeoData AI" in response.text
    assert client.get("/app.js").status_code == 200
