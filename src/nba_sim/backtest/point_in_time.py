"""Point-in-time checks for historical prediction inputs.

Date-only or timezone-naive values are not evidence of availability and are
rejected. Historical replay requires an explicit timezone-aware timestamp.
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
    # utc=True would silently interpret naive timestamps as UTC. That is not
    # valid source provenance, so validate each value before UTC conversion.
    def parse_stamp(value):
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return pd.NaT
        if pd.isna(stamp) or stamp.tzinfo is None:
            return pd.NaT
        return stamp.tz_convert("UTC")

    stamps = pd.Series(
        (parse_stamp(value) for value in frame[available_column]),
        index=frame.index,
        dtype="datetime64[ns, UTC]",
    )
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


def validate_snapshot_manifest(manifest: pd.DataFrame) -> None:
    """Validate immutable source metadata required for an auditable replay.

    The prediction timestamp is not a substitute for source availability.
    Every row must have an explicit UTC-offset timestamp and stable identity.
    """
    required = ("source", "observed_at", "available_at", "prediction_at", "game_id", "model_version", "snapshot_id")
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"Missing snapshot manifest columns: {', '.join(missing)}")
    for column in ("source", "game_id", "model_version", "snapshot_id"):
        if manifest[column].isna().any() or manifest[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing snapshot identity: {column}")
    timestamps = {}
    for column in ("observed_at", "available_at", "prediction_at"):
        raw = manifest[column]
        if raw.isna().any():
            raise ValueError(f"Missing snapshot timestamp: {column}")
        # Pandas' utc=True accepts naive values by assuming UTC. That would
        # silently fabricate provenance, so reject them explicitly.
        for value in raw:
            stamp = pd.Timestamp(value)
            if pd.isna(stamp) or stamp.tzinfo is None:
                raise ValueError(f"Snapshot timestamp must be timezone-aware: {column}")
        timestamps[column] = pd.to_datetime(raw, utc=True, errors="raise")
    if (timestamps["observed_at"] > timestamps["available_at"]).any():
        raise ValueError("Snapshot cannot be available before observation")
    if (timestamps["available_at"] > timestamps["prediction_at"]).any():
        raise ValueError("Snapshot became available after prediction cutoff")
    # When tipoff is recorded, reject predictions made at or after tipoff.
    # An as-of filter alone cannot detect a postgame snapshot with an
    # internally consistent observation/availability/prediction timeline.
    if "game_start_at" in manifest.columns:
        raw = manifest["game_start_at"]
        if raw.isna().any():
            raise ValueError("Missing snapshot timestamp: game_start_at")
        for value in raw:
            try:
                stamp = pd.Timestamp(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid snapshot timestamp: game_start_at") from exc
            if pd.isna(stamp) or stamp.tzinfo is None:
                raise ValueError("Snapshot timestamp must be timezone-aware: game_start_at")
        starts = pd.to_datetime(raw, utc=True, errors="raise")
        if (timestamps["prediction_at"] >= starts).any():
            raise ValueError("Snapshot prediction must precede game start")
    if manifest.duplicated(subset=["game_id", "source", "snapshot_id"]).any():
        raise ValueError("Duplicate snapshot identity")


def replay_inputs_as_of(manifest: pd.DataFrame, prediction_time: str | pd.Timestamp) -> pd.DataFrame:
    """Return the latest available snapshot per game and source at a cutoff.

    A historical replay must not combine multiple revisions of the same
    source. The snapshot's own prediction timestamp must also be no later
    than the requested replay cutoff.
    """
    validate_snapshot_manifest(manifest)
    eligible = available_as_of(manifest, prediction_time)
    eligible = available_as_of(
        eligible, prediction_time, available_column="prediction_at"
    )
    if eligible.empty:
        return eligible
    # Stable sorting makes the tie-breaker deterministic when two snapshots
    # have the same availability timestamp.
    ordered = eligible.assign(
        _available_utc=pd.to_datetime(eligible["available_at"], utc=True),
        _prediction_utc=pd.to_datetime(eligible["prediction_at"], utc=True),
    ).sort_values(
        ["_available_utc", "_prediction_utc", "snapshot_id"],
        kind="stable",
    )
    latest = ordered.drop_duplicates(["game_id", "source"], keep="last")
    return latest.drop(columns=["_available_utc", "_prediction_utc"]).sort_index()
