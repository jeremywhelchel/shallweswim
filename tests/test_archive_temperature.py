"""Temperature archive schema and time-boundary tests."""

import datetime
from io import BytesIO

import pandas as pd
import pandera.errors
import pytest
import pytz

from shallweswim.archive.observations import (
    OBSERVATION_COLUMNS,
    TEMPERATURE_UNIT,
    TEMPERATURE_VALUE_COLUMN,
    ObservationModel,
    normalize_observations,
    read_observations,
)

EASTERN = pytz.timezone("US/Eastern")


def _temperature_frame(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"water_temp": [60.0 + index for index in range(len(times))]},
        index=pd.DatetimeIndex(times, name="time"),
    )


def _normalize_temperature(
    frame: pd.DataFrame, *, retrieved_at: datetime.datetime
) -> pd.DataFrame:
    return normalize_observations(
        frame,
        value_column=TEMPERATURE_VALUE_COLUMN,
        unit=TEMPERATURE_UNIT,
        timezone=EASTERN,
        retrieved_at=retrieved_at,
    )


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
    source.loc[pd.Timestamp("2026-01-15 13:00"), "water_temp"] = float("nan")

    frame = _normalize_temperature(
        source,
        retrieved_at=datetime.datetime(2026, 1, 15, 19, 10),
    )

    assert frame["value"].tolist() == [60.0, 62.0]
    assert frame["observed_at"].dt.strftime("%H:%M").tolist() == ["17:00", "19:00"]


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


def test_fall_back_fold_is_inferred_from_ordered_repeated_hour() -> None:
    frame = _normalize_temperature(
        _temperature_frame(
            [
                "2026-11-01 00:30",
                "2026-11-01 01:00",
                "2026-11-01 01:30",
                "2026-11-01 01:00",
                "2026-11-01 01:30",
                "2026-11-01 02:00",
            ]
        ),
        retrieved_at=datetime.datetime(2026, 11, 1, 8, 0),
    )

    assert frame["observed_at"].is_unique
    assert frame["observed_at"].is_monotonic_increasing
    assert frame["observed_at"].dt.strftime("%H:%M").tolist() == [
        "04:30",
        "05:00",
        "05:30",
        "06:00",
        "06:30",
        "07:00",
    ]


def test_unresolvable_fall_back_fold_raises() -> None:
    with pytest.raises(ValueError, match="Cannot infer dst time"):
        _normalize_temperature(
            _temperature_frame(["2026-11-01 01:30"]),
            retrieved_at=datetime.datetime(2026, 11, 1, 8, 0),
        )


def test_nonexistent_spring_forward_time_raises() -> None:
    with pytest.raises(ValueError, match="nonexistent time"):
        _normalize_temperature(
            _temperature_frame(["2026-03-08 02:30"]),
            retrieved_at=datetime.datetime(2026, 3, 8, 8, 0),
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
