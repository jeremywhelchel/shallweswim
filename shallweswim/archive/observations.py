"""Normalized scalar-observation schema, conversion, and read boundary."""

import dataclasses
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

# Named domain bindings to the shared scalar-observation mechanism. The
# measurement is both the archive path prefix segment and the citation_key
# segment that identifies what a source measures.
TEMPERATURE_MEASUREMENT = "temperature"
TEMPERATURE_VALUE_COLUMN = "water_temp"
TEMPERATURE_UNIT = "F"

CURRENTS_MEASUREMENT = "currents"
CURRENTS_VALUE_COLUMN = "velocity"
CURRENTS_UNIT = "kt"

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


@dataclasses.dataclass(frozen=True)
class NormalizedObservations:
    """Archive rows plus the rows dropped to produce them.

    `ambiguous_dropped` counts unresolvable fall-back rows. `conflicting_dropped`
    counts rows that repeated an instant already kept but claimed a different
    value; identical repeats collapse silently and are not counted.
    """

    frame: pd.DataFrame
    ambiguous_dropped: int
    conflicting_dropped: int


def normalize_observations(
    frame: pd.DataFrame,
    *,
    value_column: str,
    unit: str,
    timezone: datetime.tzinfo,
    retrieved_at: datetime.datetime,
) -> NormalizedObservations:
    """Convert a feed's scalar values and naive local times to archive rows.

    Ambiguous fall-back times are inferred only when the ordered observations
    contain enough information to distinguish both folds. When inference is
    impossible the ambiguous rows are dropped and counted for the caller, which
    knows the source identity; a fold is never guessed. Nonexistent
    spring-forward wall times still raise instead of being shifted.

    A repeated UTC instant keeps its first row. A repeat that claims a different
    value is also dropped, but counted separately so the discarded claim is
    visible rather than silent.
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

    try:
        localized = observed_at.tz_localize(
            timezone,
            ambiguous="infer",
            nonexistent="raise",
        )
    except ValueError:
        # pandas reports every failed localization as a plain ValueError, both
        # "no repeated times" and "there are N dst switches". Retry with the
        # ambiguous rows marked so the resolvable ones survive; nonexistent
        # times raise again from this second attempt.
        localized = observed_at.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="raise",
        )

    resolved = localized.notna()
    ambiguous_dropped = int((~resolved).sum())
    if ambiguous_dropped:
        observations = observations.loc[resolved]
        localized = localized[resolved]

    observed_at_utc = localized.tz_convert("UTC").as_unit("ns")

    # Native-cadence provider frames may repeat a reading. Two fall-back folds
    # are distinct UTC instants, so collapsing on the instant keeps both.
    repeated = observed_at_utc.duplicated(keep="first")
    conflicting_dropped = 0
    if repeated.any():
        values = observations[value_column].to_numpy(dtype="float64")
        kept = pd.Series(values[~repeated], index=observed_at_utc[~repeated])
        conflicting_dropped = int(
            (repeated & (values != kept.reindex(observed_at_utc).to_numpy())).sum()
        )
        observations = observations.loc[~repeated]
        observed_at_utc = observed_at_utc[~repeated]

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
    return NormalizedObservations(
        frame=normalize_archive_frame(result, expected_unit=unit),
        ambiguous_dropped=ambiguous_dropped,
        conflicting_dropped=conflicting_dropped,
    )


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
