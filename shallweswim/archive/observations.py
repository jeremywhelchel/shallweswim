"""Normalized scalar-observation schema, conversion, and read boundary.

Capture receives the UTC-indexed frame a client returned, so conversion here is
a timezone change rather than an interpretation of local wall times.
"""

import dataclasses
import datetime
from pathlib import Path
from typing import Annotated, BinaryIO
from urllib.parse import quote

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

# The provider products the archive records, one name per way a provider
# publishes a reading. Capture writes the product of the fetch that returned a
# row, and `archive/merge.py` ranks them per provider. They are named here, with
# the column they fill, because the merge writer and the feeds that state them
# both read them and the archive layer imports neither `core/` nor `clients/`.
COOPS_HOURLY_PRODUCT = "coops:h"
COOPS_SIX_MINUTE_PRODUCT = "coops:6-min"
NDBC_FILES_PRODUCT = "ndbc:files"
NDBC_REALTIME_PRODUCT = "ndbc:realtime"
NWIS_PRODUCT = "nwis"
CSPF_PRODUCT = "cspf"
IRISH_LIGHTS_PRODUCT = "irish-lights"

# The column those products are written to.
PRODUCT_COLUMN = "product"


def validate_source_identity(source_identity: str, measurement: str) -> tuple[str, str]:
    """Return the provider and station of a source identity for this measurement.

    Args:
        source_identity: A feed's `citation_key`, shaped
            `<provider>:<measurement>:<station>`, where the station may itself
            carry a colon-separated parameter suffix.
        measurement: The measurement the caller is archiving or reading.

    Returns:
        The provider and station segments of the identity.

    Raises:
        ValueError: If the identity does not name this measurement, or either
            of its other segments is empty.
    """
    provider, source_measurement, station = source_identity.split(":", 2)
    if source_measurement != measurement or not provider or not station:
        raise ValueError(f"Expected a {measurement} source identity")
    return provider, station


def partition_key(source_identity: str, measurement: str, year: int) -> str:
    """Return the archive object key holding one source's UTC year.

    Capture writes and hydration reads the same keys, so both derive them here.

    Raises:
        ValueError: If the source identity does not name this measurement.
    """
    provider, station = validate_source_identity(source_identity, measurement)
    # Percent encoding is reversible, including USGS's station:parameter suffix.
    return (
        f"archive/{measurement}/{quote(provider, safe='')}"
        f"/{quote(station, safe='')}/{year}.parquet"
    )


# Additive nullable fields belong here and in ObservationModel when the archive
# contract grows. The reader fills fields absent from older Parquet objects before
# Pandera validation, so a row written before a field existed reads back as null.
_ADDITIVE_NULLABLE_COLUMNS: dict[str, str] = {PRODUCT_COLUMN: "string"}


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
    # Null for every row archived before the column existed; those rows rank
    # below every named product in a merge.
    product: pa_typing.Series[pd.StringDtype] = pa.Field(nullable=True)

    class Config:  # type: ignore[misc]
        """Require the exact, ordered archive schema without coercion."""

        strict = True
        ordered = True
        coerce = False


@dataclasses.dataclass(frozen=True)
class NormalizedObservations:
    """Archive rows plus the rows dropped to produce them.

    `conflicting_dropped` counts rows that repeated an instant already kept but
    claimed a different value; identical repeats collapse silently and are not
    counted.
    """

    frame: pd.DataFrame
    conflicting_dropped: int


def normalize_observations(
    frame: pd.DataFrame,
    *,
    value_column: str,
    unit: str,
    retrieved_at: datetime.datetime,
    product: str | None = None,
) -> NormalizedObservations:
    """Convert a client's scalar values and UTC instants to archive rows.

    The frame must carry a timezone-aware index, which every client returns, so
    the observation time is an exact instant and both folds of a daylight-saving
    fall-back hour stay distinct.

    A repeated UTC instant keeps its first row. A repeat that claims a different
    value is also dropped, but counted separately so the discarded claim is
    visible rather than silent.

    Args:
        frame: The client frame, indexed by timezone-aware UTC instants.
        value_column: The frame column holding the scalar value.
        unit: The canonical unit every row carries.
        retrieved_at: When this fetch happened.
        product: The provider product this fetch returned, written on every
            row; None leaves the column null, which ranks below every product.

    Raises:
        ValueError: If the value column is missing, the unit is empty, or the
            frame index is timezone naive.
    """
    if value_column not in frame.columns:
        raise ValueError(f"Feed frame must contain {value_column}")
    if not unit:
        raise ValueError("Observation unit must not be empty")

    # Resampling gaps are not observations. Drop them without losing valid rows.
    observations = frame.loc[frame[value_column].notna()]

    observed_at = pd.DatetimeIndex(observations.index)
    if observed_at.tz is None:
        raise ValueError("Feed timestamps must be timezone-aware")

    observed_at_utc = observed_at.tz_convert("UTC").as_unit("ns")

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
            PRODUCT_COLUMN: pd.Series(
                product if product is not None else pd.NA,
                index=range(len(observations)),
                dtype="string",
            ),
        }
    )
    return NormalizedObservations(
        frame=normalize_archive_frame(result, expected_unit=unit),
        conflicting_dropped=conflicting_dropped,
    )


def normalize_archive_frame(frame: pd.DataFrame, *, expected_unit: str) -> pd.DataFrame:
    """Normalize an archive frame across additive schemas and validate it."""
    normalized = frame.copy()
    for column, dtype in _ADDITIVE_NULLABLE_COLUMNS.items():
        if column not in normalized.columns:
            normalized[column] = pd.Series(pd.NA, index=normalized.index, dtype=dtype)
        else:
            normalized[column] = normalized[column].astype(dtype)

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
