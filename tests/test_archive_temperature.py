"""Temperature archive schema and time-boundary tests."""

import datetime
from io import BytesIO

import pandas as pd
import pandera.errors
import pytest

from shallweswim.archive.observations import (
    OBSERVATION_COLUMNS,
    TEMPERATURE_UNIT,
    TEMPERATURE_VALUE_COLUMN,
    NormalizedObservations,
    ObservationModel,
    normalize_observations,
    read_observations,
)


def _temperature_frame(times: list[str]) -> pd.DataFrame:
    """Build a client-style frame indexed by timezone-aware UTC instants."""
    return pd.DataFrame(
        {"water_temp": [60.0 + index for index in range(len(times))]},
        index=pd.DatetimeIndex(times, tz="UTC", name="time"),
    )


def _normalize_result(
    frame: pd.DataFrame, *, retrieved_at: datetime.datetime
) -> NormalizedObservations:
    return normalize_observations(
        frame,
        value_column=TEMPERATURE_VALUE_COLUMN,
        unit=TEMPERATURE_UNIT,
        retrieved_at=retrieved_at,
    )


def _normalize_temperature(
    frame: pd.DataFrame, *, retrieved_at: datetime.datetime
) -> pd.DataFrame:
    return _normalize_result(frame, retrieved_at=retrieved_at).frame


def test_temperature_archive_schema_contract() -> None:
    assert TEMPERATURE_VALUE_COLUMN == "water_temp"
    assert TEMPERATURE_UNIT == "F"

    frame = _normalize_temperature(
        _temperature_frame(["2026-01-15 12:00", "2026-01-15 12:06"]),
        retrieved_at=datetime.datetime(2026, 1, 15, 17, 10),
    )

    assert tuple(frame.columns) == OBSERVATION_COLUMNS
    assert str(frame.dtypes["observed_at"]) == "datetime64[ns, UTC]"
    assert str(frame.dtypes["value"]) == "float64"
    assert str(frame.dtypes["unit"]) == "string"
    assert str(frame.dtypes["retrieved_at"]) == "datetime64[ns, UTC]"
    assert frame["unit"].tolist() == ["F", "F"]


def test_resampling_gaps_are_not_archived_as_observations() -> None:
    source = _temperature_frame(
        ["2026-01-15 12:00", "2026-01-15 13:00", "2026-01-15 14:00"]
    )
    source.loc[pd.Timestamp("2026-01-15 13:00", tz="UTC"), "water_temp"] = float("nan")

    frame = _normalize_temperature(
        source,
        retrieved_at=datetime.datetime(2026, 1, 15, 19, 10),
    )

    assert frame["value"].tolist() == [60.0, 62.0]
    assert frame["observed_at"].dt.strftime("%H:%M").tolist() == ["12:00", "14:00"]


def test_all_nan_frame_yields_empty_valid_archive_frame() -> None:
    source = _temperature_frame(["2026-01-15 12:00", "2026-01-15 13:00"])
    source["water_temp"] = float("nan")

    frame = _normalize_temperature(
        source,
        retrieved_at=datetime.datetime(2026, 1, 15, 19, 10),
    )

    assert frame.empty
    assert tuple(frame.columns) == OBSERVATION_COLUMNS
    ObservationModel.validate(frame, lazy=True)


def test_fall_back_folds_archive_as_two_rows() -> None:
    """US/Eastern 2026-11-01 01:30 names two UTC instants an hour apart."""
    result = _normalize_result(
        _temperature_frame(
            [
                "2026-11-01 04:30",
                "2026-11-01 05:30",
                "2026-11-01 06:30",
                "2026-11-01 07:30",
            ]
        ),
        retrieved_at=datetime.datetime(2026, 11, 1, 8, 0),
    )

    assert result.conflicting_dropped == 0
    frame = result.frame
    assert frame["observed_at"].is_unique
    assert frame["observed_at"].is_monotonic_increasing
    # 05:30 and 06:30 UTC are both local 01:30; both survive as distinct rows.
    assert frame["observed_at"].dt.strftime("%H:%M").tolist() == [
        "04:30",
        "05:30",
        "06:30",
        "07:30",
    ]
    assert frame["value"].tolist() == [60.0, 61.0, 62.0, 63.0]


def test_repeated_instant_keeps_first_without_collapsing_folds() -> None:
    result = _normalize_result(
        _temperature_frame(
            [
                "2026-11-01 04:30",
                "2026-11-01 04:30",
                "2026-11-01 05:30",
                "2026-11-01 06:30",
            ]
        ),
        retrieved_at=datetime.datetime(2026, 11, 1, 8, 0),
    )

    # The repeated 04:30 claims 61.0 against the kept 60.0, so it is counted.
    assert result.conflicting_dropped == 1
    assert result.frame["observed_at"].is_unique
    assert result.frame["observed_at"].dt.strftime("%H:%M").tolist() == [
        "04:30",
        "05:30",
        "06:30",
    ]
    # The repeated 04:30 keeps the first reading; both local 01:30 folds survive.
    assert result.frame["value"].tolist() == [60.0, 62.0, 63.0]


def test_identical_repeated_instant_collapses_without_a_conflict() -> None:
    source = pd.DataFrame(
        {"water_temp": [60.0, 60.0, 61.0]},
        index=pd.DatetimeIndex(
            ["2026-01-15 12:00", "2026-01-15 12:00", "2026-01-15 12:06"],
            tz="UTC",
            name="time",
        ),
    )

    result = _normalize_result(
        source, retrieved_at=datetime.datetime(2026, 1, 15, 17, 10)
    )

    assert result.conflicting_dropped == 0
    assert result.frame["value"].tolist() == [60.0, 61.0]


def test_timezone_naive_feed_index_is_rejected() -> None:
    """Clients return UTC; a naive index is a bug, not an interpretable frame."""
    source = pd.DataFrame(
        {"water_temp": [60.0]},
        index=pd.DatetimeIndex(["2026-01-15 12:00"], name="time"),
    )

    with pytest.raises(ValueError, match="Feed timestamps must be timezone-aware"):
        _normalize_temperature(
            source, retrieved_at=datetime.datetime(2026, 1, 15, 17, 10)
        )


def test_reader_normalizes_and_validates_parquet() -> None:
    expected = _normalize_temperature(
        _temperature_frame(["2026-01-15 12:00"]),
        retrieved_at=datetime.datetime(2026, 1, 15, 17, 10),
    )
    parquet = BytesIO()
    expected.to_parquet(parquet, index=False)
    parquet.seek(0)

    actual = read_observations(parquet, expected_unit=TEMPERATURE_UNIT)

    pd.testing.assert_frame_equal(actual, expected)


def test_reader_rejects_noncanonical_unit() -> None:
    frame = _normalize_temperature(
        _temperature_frame(["2026-01-15 12:00"]),
        retrieved_at=datetime.datetime(2026, 1, 15, 17, 10),
    )
    frame["unit"] = pd.Series(["C"], dtype="string")
    parquet = BytesIO()
    frame.to_parquet(parquet, index=False)
    parquet.seek(0)

    with pytest.raises(ValueError, match="unit must be F"):
        read_observations(parquet, expected_unit=TEMPERATURE_UNIT)


def test_reader_rejects_unreviewed_columns() -> None:
    frame = _normalize_temperature(
        _temperature_frame(["2026-01-15 12:00"]),
        retrieved_at=datetime.datetime(2026, 1, 15, 17, 10),
    )
    frame["quality"] = pd.Series(["provisional"], dtype="string")

    with pytest.raises(ValueError, match="unexpected columns: quality"):
        read_observations(
            BytesIO(_to_parquet_bytes(frame)), expected_unit=TEMPERATURE_UNIT
        )


def test_pandera_model_rejects_timezone_naive_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(["2026-01-15 12:00"]),
            "value": pd.Series([60.0], dtype="float64"),
            "unit": pd.Series(["F"], dtype="string"),
            "retrieved_at": pd.to_datetime(["2026-01-15 17:10"]),
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        ObservationModel.validate(frame, lazy=True)


def _to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    parquet = BytesIO()
    frame.to_parquet(parquet, index=False)
    return parquet.getvalue()
