"""Read-only tools the assistant can call, and their executor.

Every tool reads through the same repository, model and weather index the
REST endpoints use, so answers match what the map shows. Tool inputs come from
the model and are validated and clamped here before they reach a query.
"""

import json
import logging
from collections import defaultdict
from datetime import date

import pandas as pd

from api.filters import ORDER_OPTIONS, FireFilters
from api.serialization import frame_to_records

LOGGER = logging.getLogger("geodata.api.assistant.tools")

SOURCES = ["modis", "viirs_j1", "viirs_suomi"]
MAX_LIST_ROWS = 50

_DATE = {"type": "string", "format": "date", "description": "YYYY-MM-DD"}
_BBOX = {
    "type": "array",
    "items": {"type": "number"},
    "description": "Optional bounding box [min_lon, min_lat, max_lon, max_lat] in decimal degrees. "
                   "For a named country or region, use an approximate rectangle around it.",
}
_SOURCES = {
    "type": "array",
    "items": {"type": "string", "enum": SOURCES},
    "description": "Optional satellite sources; omit for all.",
}

TOOLS = [
    {
        "name": "get_data_coverage",
        "description": "What data is loaded: the date range of fire detections, whether the API serves the full table or a random "
                       "sample (and the sample fraction), weather coverage, and the backend. Call this first when the question "
                       "depends on which dates or how much data exist.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_fire_statistics",
        "description": "Aggregate statistics for fire detections matching filters: total count, mean and max FRP, the count and "
                       "share of severe detections (FRP at or above the threshold), and breakdowns by satellite, day/night and "
                       "date (daily for ranges up to 62 days, monthly beyond). Use this for any count, trend or comparison.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": _DATE,
                "end_date": _DATE,
                "bbox": _BBOX,
                "sources": _SOURCES,
                "min_confidence": {"type": "integer", "description": "Optional minimum confidence, 0-100."},
                "daynight": {"type": "string", "enum": ["D", "N"], "description": "Optional: D for day, N for night."},
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_fires",
        "description": f"Individual detections matching filters, at most {MAX_LIST_ROWS}, ordered as requested, optionally with the "
                       "model's predicted probability of being severe. Use it to name the strongest fires, give examples, or "
                       "inspect a place.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": _DATE,
                "end_date": _DATE,
                "bbox": _BBOX,
                "sources": _SOURCES,
                "min_frp": {"type": "number", "description": "Optional minimum FRP in MW."},
                "order": {"type": "string", "enum": list(ORDER_OPTIONS), "description": "Sort order; default frp_desc."},
                "limit": {"type": "integer", "description": f"Rows to return, 1-{MAX_LIST_ROWS}; default 10."},
                "include_predictions": {"type": "boolean", "description": "Attach the model's severity probability."},
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "predict_fire_severity",
        "description": "Score a hypothetical or real detection with the trained model. Returns the probability that its FRP is at "
                       "or above the severity threshold, the weather matched for that place and day, and whether the input lies "
                       "inside the data the model was trained on.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "acq_date": _DATE,
                "acq_time": {"type": "integer", "description": "HHMM UTC."},
                "brightness": {"type": "number", "description": "Brightness temperature in K."},
                "confidence": {"type": "integer", "description": "0-100."},
                "scan": {"type": "number", "description": "Pixel size along scan, km (VIIRS about 0.38, MODIS about 1.0)."},
                "track": {"type": "number", "description": "Pixel size along track, km."},
                "type": {"type": "integer", "description": "0 vegetation, 1 volcano, 2 static source, 3 offshore."},
            },
            "required": ["latitude", "longitude", "acq_date"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_model_info",
        "description": "The severity model's definition, evaluation metrics on its chronological hold-out, feature importances, "
                       "hyperparameters and the area and period it was trained on.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "search_project_docs",
        "description": "Search the project's documentation and data dictionary (pipeline stages, column meanings, how weather is "
                       "matched, configuration, API endpoints, deployment). Use for questions about how the system works.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """A problem with the tool input that the model can fix and retry."""


def _parse_date(value, name: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as error:
        raise ToolError(f"{name} must be a date in YYYY-MM-DD format, got {value!r}.") from error


def _filters(args: dict, limit: int = 1, order: str = "acq_date_desc") -> FireFilters:
    start = _parse_date(args["start_date"], "start_date")
    end = _parse_date(args["end_date"], "end_date")
    if start > end:
        raise ToolError("start_date must be on or before end_date.")
    filters = FireFilters(start_date=start, end_date=end, limit=limit, order=order)
    bbox = args.get("bbox")
    if bbox:
        if len(bbox) != 4:
            raise ToolError("bbox must have four numbers: [min_lon, min_lat, max_lon, max_lat].")
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
        if not (-90 <= min_lat <= max_lat <= 90 and -180 <= min_lon <= max_lon <= 180):
            raise ToolError("bbox is out of range or inverted; use [min_lon, min_lat, max_lon, max_lat].")
        filters.min_lon, filters.min_lat, filters.max_lon, filters.max_lat = min_lon, min_lat, max_lon, max_lat
    sources = [s for s in (args.get("sources") or []) if s in SOURCES]
    filters.sources = sources or None
    if args.get("min_confidence") is not None:
        filters.min_confidence = max(0, min(100, int(args["min_confidence"])))
    if args.get("daynight") in ("D", "N"):
        filters.daynight = args["daynight"]
    if args.get("min_frp") is not None:
        filters.min_frp = max(0.0, float(args["min_frp"]))
    return filters


def _round(value, digits=2):
    return None if value is None else round(float(value), digits)


class ToolExecutor:
    def __init__(self, runtime, knowledge):
        self.runtime = runtime
        self.knowledge = knowledge

    @property
    def _repository(self):
        if self.runtime.repository is None:
            raise ToolError("No data backend is available right now.")
        return self.runtime.repository

    def run(self, name: str, args: dict) -> dict:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"Unknown tool {name!r}.")
        return handler(args or {})

    # ---------------------------------------------------------------- tools
    def _tool_get_data_coverage(self, _args):
        repository = self._repository
        health = repository.health()
        start, end = self.runtime.date_range()
        index = self.runtime.weather_index()
        coverage = {
            "backend": repository.name,
            "fire_date_range": {"start": start.isoformat() if start else None, "end": end.isoformat() if end else None},
            "rows_served": health.get("rows"),
            "is_random_sample": bool(health.get("is_subset")),
            "sample_fraction": health.get("sample_fraction"),
            "weather": index.summary() if index is not None else None,
        }
        if coverage["is_random_sample"]:
            coverage["note"] = (
                "Counts come from a uniform random sample. Shares, averages and rankings are representative; "
                "absolute counts are scaled down by sample_fraction (divide by it to estimate the full count)."
            )
        return coverage

    def _tool_get_fire_statistics(self, args):
        filters = _filters(args)
        threshold = self.runtime.settings.high_frp_threshold_mw
        stats = self._repository.fire_stats(filters, threshold)
        span_days = (filters.end_date - filters.start_date).days + 1
        if span_days > 62:
            months = defaultdict(lambda: {"count": 0, "high_frp_count": 0})
            for row in stats["by_date"]:
                bucket = months[row["date"][:7]]
                bucket["count"] += row["count"]
                bucket["high_frp_count"] += row["high_frp_count"]
            series = {"by_month": [{"month": month, **values} for month, values in sorted(months.items())]}
        else:
            series = {"by_date": [{"date": r["date"], "count": r["count"], "high_frp_count": r["high_frp_count"]} for r in stats["by_date"]]}
        peak = max(stats["by_date"], key=lambda r: r["count"]) if stats["by_date"] else None
        health = self._repository.health()
        result = {
            "filters": filters.describe(),
            "severity_threshold_mw": threshold,
            "total_detections": stats["total"],
            "mean_frp_mw": _round(stats["avg_frp"]),
            "max_frp_mw": _round(stats["max_frp"]),
            "severe_detections": stats["high_frp_count"],
            "severe_share": _round(stats["high_frp_share"], 4),
            "busiest_day": {"date": peak["date"], "count": peak["count"]} if peak else None,
            "by_source": [{"source": s["source"], "count": s["count"], "mean_frp_mw": _round(s["avg_frp"]), "severe": s["high_frp_count"]} for s in stats["by_source"]],
            "by_daynight": stats["by_daynight"],
            **series,
        }
        if health.get("is_subset"):
            fraction = health.get("sample_fraction")
            result["is_random_sample"] = True
            if fraction:
                result["estimated_full_total"] = int(round(stats["total"] / fraction))
        return result

    def _tool_list_fires(self, args):
        limit = max(1, min(MAX_LIST_ROWS, int(args.get("limit") or 10)))
        order = args.get("order") if args.get("order") in ORDER_OPTIONS else "frp_desc"
        filters = _filters(args, limit=limit, order=order)
        frame = self._repository.fires(filters)
        columns = [c for c in ("acq_date", "acq_time", "latitude", "longitude", "frp", "brightness", "confidence", "scan", "track", "daynight", "satellite_source", "type") if c in frame.columns]
        frame = frame[columns]
        if args.get("include_predictions") and not frame.empty:
            predictor = self.runtime.predictor
            if predictor is None or not predictor.loaded:
                raise ToolError("The model is not loaded, so predictions are unavailable. Retry without include_predictions.")
            predictions = predictor.predict_frame(frame, self.runtime.weather_index())
            frame = frame.join(predictions[["high_frp_probability", "weather_matched"]])
        return {"filters": filters.describe(), "count": len(frame), "fires": frame_to_records(frame)}

    def _tool_predict_fire_severity(self, args):
        predictor = self.runtime.predictor
        if predictor is None or not predictor.loaded:
            raise ToolError("The model is not loaded.")
        latitude, longitude = float(args["latitude"]), float(args["longitude"])
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ToolError("latitude must be within -90..90 and longitude within -180..180.")
        day = _parse_date(args["acq_date"], "acq_date")
        row = {"latitude": latitude, "longitude": longitude, "acq_date": pd.Timestamp(day)}
        for key in ("acq_time", "brightness", "confidence", "scan", "track", "type"):
            if args.get(key) is not None:
                row[key] = args[key]
        frame = pd.DataFrame([row])
        result = predictor.predict_frame(frame, self.runtime.weather_index()).iloc[0]
        weather = {f: _round(result.get(f)) for f in predictor.weather_features if pd.notna(result.get(f))}
        return {
            "input": {**row, "acq_date": day.isoformat()},
            "probability_severe": _round(result["high_frp_probability"], 4),
            "predicted_label": result["predicted_label"],
            "decision_threshold": predictor.decision_threshold,
            "severity_threshold_mw": predictor.high_frp_threshold_mw,
            "weather_matched": bool(result["weather_matched"]),
            "weather_distance_km": _round(result["weather_distance_km"]) if pd.notna(result["weather_distance_km"]) else None,
            "weather": weather,
            "in_training_extent": predictor.in_training_extent(latitude, longitude, day),
            "missing_inputs_imputed": [k for k in ("acq_time", "brightness", "confidence", "scan", "track", "type") if k not in row],
        }

    def _tool_get_model_info(self, _args):
        predictor = self.runtime.predictor
        if predictor is None or not predictor.loaded:
            raise ToolError("The model is not loaded.")
        info = predictor.info()
        importances = info.get("feature_importances") or {}
        return {
            "target": info["target"],
            "estimator": info["estimator"],
            "decision_threshold": info["decision_threshold"],
            "metrics_on_holdout": info.get("metrics"),
            "feature_importances": {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda kv: -kv[1])},
            "training_extent": info.get("training_extent"),
            "model_file": info["model_path"].replace("\\", "/").rsplit("/", 1)[-1],
        }

    def _tool_search_project_docs(self, args):
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolError("query must not be empty.")
        hits = self.knowledge.search(query, limit=4)
        return {"query": query, "results": hits or [], "note": None if hits else "No matching documentation."}


def serialize(result: dict) -> str:
    return json.dumps(result, default=str, ensure_ascii=False)
