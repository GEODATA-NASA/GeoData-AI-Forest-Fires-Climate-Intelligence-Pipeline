"""Helpers turning pandas frames into JSON-safe records."""

import math

import numpy as np
import pandas as pd


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of dicts with plain Python values."""
    if frame.empty:
        return []
    prepared = frame.copy()
    for column in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[column]):
            prepared[column] = prepared[column].dt.strftime("%Y-%m-%d")
    records = prepared.astype(object).to_dict("records")
    return [{key: _scalar(value) for key, value in record.items()} for record in records]
