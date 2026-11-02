"""Point-in-time checks for historical prediction inputs.

A date-only snapshot is conservatively treated as available at the end of that
day; intraday availability must be represented by an explicit timestamp.
"""
from __future__ import annotations

import pandas as pd


def available_as_of(
    frame: pd.DataFrame,
    prediction_time: str | pd.Timestamp,
    *,
    available_column: str = "available_at",
    strict: bool = True,
) -> pd.DataFrame:
    """Return only rows available before the forecast was issued.

    Reject missing provenance instead of silently using future or unknown data.
    """
    if available_column not in frame.columns:
        if strict:
            raise ValueError(f"Missing point-in-time provenance column: {available_column}")
        return frame.iloc[0:0].copy()
    cutoff = pd.Timestamp(prediction_time)
    if cutoff.tzinfo is None:
        raise ValueError("prediction_time must include a timezone")
    stamps = pd.to_datetime(frame[available_column], errors="coerce", utc=True)
    if strict and stamps.isna().any():
        raise ValueError("Missing or invalid availability timestamps")
    mask = stamps.notna() & (stamps <= cutoff.tz_convert("UTC"))
    return frame.loc[mask].copy()


def assert_no_future_inputs(
    frame: pd.DataFrame,
    prediction_time: str | pd.Timestamp,
    *,
    available_column: str = "available_at",
) -> None:
    """Fail closed when a forecast uses unavailable historical inputs."""
    selected = available_as_of(frame, prediction_time, available_column=available_column)
    if len(selected) != len(frame):
        raise ValueError("Forecast contains data unavailable at prediction time")
