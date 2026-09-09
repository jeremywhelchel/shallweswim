"""Normalized scalar-observation schema, conversion, and read boundary."""

import datetime
from pathlib import Path
from typing import Annotated, BinaryIO

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pa_typing

OBSERVATION_COLUMNS = (
    "observed_at",
    "value",
    "unit",
    "retrieved_at",
)

# Phase 1's named domain binding to the shared scalar-observation mechanism.
TEMPERATURE_VALUE_COLUMN = "water_temp"
TEMPERATURE_UNIT = "F"

# Additive nullable fields belong here and in ObservationModel when the archive
# contract grows. The reader fills fields absent from older Parquet objects before
# Pandera validation.
_ADDITIVE_NULLABLE_COLUMNS: dict[str, str] = {}


class ObservationModel(pa.DataFrameModel):
    """The normalized scalar-observation archive row contract."""

    observed_at: pa_typing.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = (
        pa.Field(nullable=False)
    )
    value: pa_typing.Series[float] = pa.Field(nullable=False)
    unit: pa_typing.Series[pd.StringDtype] = pa.Field(nullable=False)
    retrieved_at: pa_typing.Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]] = (
        pa.Field(nullable=False)
    )

    class Config:  # type: ignore[misc]
        """Require the exact, ordered archive schema without coercion."""

        strict = True
        ordered = True
        coerce = False


def normalize_observations(
    frame: pd.DataFrame,
    *,
    value_column: str,
    unit: str,
    timezone: datetime.tzinfo,
    retrieved_at: datetime.datetime,
) -> pd.DataFrame:
    """Convert a feed's scalar values and naive local times to archive rows.

    Ambiguous fall-back times are inferred only when the ordered observations
    contain enough information to distinguish both folds. Ambiguous or
    nonexistent wall times that cannot be resolved raise instead of being
    guessed or shifted.
    """
    if value_column not in frame.columns:
        raise ValueError(f"Feed frame must contain {value_column}")
    if not unit:
        raise ValueError("Observation unit must not be empty")

    # Resampling gaps are not observations. Drop them without losing valid rows.
    observations = frame.loc[frame[value_column].notna()]

    observed_at = pd.DatetimeIndex(observations.index)
    if observed_at.tz is not None:
        raise ValueError("Feed timestamps must be timezone naive")

    observed_at_utc = (
        observed_at.tz_localize(
            timezone,
            ambiguous="infer",
            nonexistent="raise",
        )
        .tz_convert("UTC")
        .as_unit("ns")
    )

    retrieved = pd.Timestamp(retrieved_at)
    if retrieved.tzinfo is None:
        retrieved = retrieved.tz_localize("UTC")
    else:
        retrieved = retrieved.tz_convert("UTC")
    retrieved = retrieved.as_unit("ns")

    result = pd.DataFrame(
        {
            "observed_at": observed_at_utc,
            "value": observations[value_column].to_numpy(dtype="float64"),
            "unit": pd.Series(unit, index=range(len(observations)), dtype="string"),
            "retrieved_at": pd.Series(
                retrieved, index=range(len(observations)), dtype="datetime64[ns, UTC]"
            ),
        }
    )
    return normalize_archive_frame(result, expected_unit=unit)


def normalize_archive_frame(frame: pd.DataFrame, *, expected_unit: str) -> pd.DataFrame:
    """Normalize an archive frame across additive schemas and validate it."""
    normalized = frame.copy()
    for column, dtype in _ADDITIVE_NULLABLE_COLUMNS.items():
        if column not in normalized.columns:
            normalized[column] = pd.Series(pd.NA, index=normalized.index, dtype=dtype)

    missing = set(OBSERVATION_COLUMNS) - set(normalized.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Observation archive is missing columns: {missing_names}")

    known_columns = set(OBSERVATION_COLUMNS) | set(_ADDITIVE_NULLABLE_COLUMNS)
    unexpected = set(normalized.columns) - known_columns
    if unexpected:
        unexpected_names = ", ".join(sorted(unexpected))
        raise ValueError(
            f"Observation archive has unexpected columns: {unexpected_names}"
        )

    normalized = normalized.loc[:, [*OBSERVATION_COLUMNS, *_ADDITIVE_NULLABLE_COLUMNS]]
    normalized["unit"] = normalized["unit"].astype("string")
    normalized["value"] = normalized["value"].astype("float64")
    for timestamp_column in ("observed_at", "retrieved_at"):
        dtype = normalized[timestamp_column].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            normalized[timestamp_column] = normalized[timestamp_column].dt.as_unit("ns")

    validated = ObservationModel.validate(normalized, lazy=True)
    if not validated["unit"].eq(expected_unit).all():
        raise ValueError(f"Observation archive unit must be {expected_unit}")
    return validated


def read_observations(
    path_or_buffer: str | Path | BinaryIO, *, expected_unit: str
) -> pd.DataFrame:
    """Read, normalize, and validate one scalar-observation Parquet object."""
    return normalize_archive_frame(
        pd.read_parquet(path_or_buffer), expected_unit=expected_unit
    )
