import pandas as pd
import pytest

from nba_sim.backtest.point_in_time import available_as_of, assert_no_future_inputs


def test_excludes_future_and_same_day_late_updates():
    rows = pd.DataFrame({
        "value": [1, 2, 3],
        "available_at": [
            "2026-10-21T23:00:00Z",
            "2026-10-22T17:00:00Z",
            "2026-10-22T23:00:00Z",
        ],
    })
    result = available_as_of(rows, "2026-10-22T18:00:00Z")
    assert result["value"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="unavailable"):
        assert_no_future_inputs(rows, "2026-10-22T18:00:00Z")


def test_rejects_missing_provenance():
    with pytest.raises(ValueError, match="provenance"):
        available_as_of(pd.DataFrame({"value": [1]}), "2026-10-22T18:00:00Z")


def test_rejects_missing_timestamps_and_naive_cutoff():
    rows = pd.DataFrame({"available_at": [None]})
    with pytest.raises(ValueError, match="invalid"):
        available_as_of(rows, "2026-10-22T18:00:00Z")
    with pytest.raises(ValueError, match="timezone"):
        available_as_of(rows, "2026-10-22")


def test_non_strict_mode_drops_unknown_rows():
    rows = pd.DataFrame({"available_at": [None, "2026-10-21T10:00:00Z"]})
    assert len(available_as_of(rows, "2026-10-22T18:00:00Z", strict=False)) == 1


def _manifest():
    return pd.DataFrame([{
        "source": "injuries", "game_id": "game-1", "snapshot_id": "snap-1",
        "model_version": "v1", "observed_at": "2026-10-22T10:00:00Z",
        "available_at": "2026-10-22T10:05:00Z",
        "prediction_at": "2026-10-22T11:00:00Z",
    }])


def test_snapshot_manifest_accepts_auditable_pregame_input():
    from nba_sim.backtest.point_in_time import replay_inputs_as_of
    assert len(replay_inputs_as_of(_manifest(), "2026-10-22T11:00:00Z")) == 1


@pytest.mark.parametrize("column", ["observed_at", "available_at", "prediction_at"])
def test_snapshot_manifest_rejects_naive_timestamps(column):
    from nba_sim.backtest.point_in_time import validate_snapshot_manifest
    rows = _manifest()
    rows.loc[0, column] = "2026-10-22T10:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_snapshot_manifest(rows)


def test_snapshot_manifest_rejects_future_or_duplicate_sources():
    from nba_sim.backtest.point_in_time import validate_snapshot_manifest
    rows = _manifest()
    rows.loc[0, "available_at"] = "2026-10-22T12:00:00Z"
    with pytest.raises(ValueError, match="after prediction"):
        validate_snapshot_manifest(rows)
    rows = pd.concat([_manifest(), _manifest()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_snapshot_manifest(rows)
