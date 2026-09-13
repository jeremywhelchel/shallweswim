"""Read archived observations back into the shape a client returned them in.

This is a read-only path: it calls the store's read operation and never the
conditional write the merge writer uses, so hydration cannot modify the archive.
"""

import asyncio
import datetime
from io import BytesIO

import pandas as pd

from shallweswim.archive.observations import partition_key, read_observations
from shallweswim.archive.store import ObjectStore


async def hydrate_year(
    store: ObjectStore,
    *,
    source_identity: str,
    measurement: str,
    value_column: str,
    unit: str,
    year: int,
    timezone: datetime.tzinfo,
) -> pd.DataFrame | None:
    """Read one archived station-local year as a client-shaped frame.

    Archive partitions are UTC years, but a historical year frame is the
    station-local year a provider fetch would return. The local year's final
    hours therefore live in the next UTC partition, so both are read and the
    result is trimmed to local instants of this year.

    Args:
        store: Object store to read the partitions from; only `read` is called.
        source_identity: The feed's `citation_key`.
        measurement: The archived measurement, such as `temperature`.
        value_column: The feed's value column, which names the frame's column.
        unit: The unit the archived rows must carry.
        year: The station-local year to read.
        timezone: The station timezone that bounds the local year.

    Returns:
        A frame indexed by timezone-aware UTC instants named `time`, carrying
        the feed's value column for instants inside local `year` and sorted by
        instant, or None when the archive holds no partition for `year`. The
        next year's partition may be absent, which simply contributes no rows.

    Raises:
        ValueError: If the source identity does not name this measurement, or a
            stored partition does not satisfy the archive schema.
    """
    stored = await store.read(partition_key(source_identity, measurement, year))
    if stored is None:
        return None
    following = await store.read(partition_key(source_identity, measurement, year + 1))
    partitions = [stored.data] if following is None else [stored.data, following.data]
    # Parquet decoding and validation are CPU work on the caller's event loop.
    return await asyncio.to_thread(
        _local_year_frame, partitions, value_column, unit, year, timezone
    )


def _local_year_frame(
    partitions: list[bytes],
    value_column: str,
    unit: str,
    year: int,
    timezone: datetime.tzinfo,
) -> pd.DataFrame:
    """Convert stored partitions into one client-shaped local-year frame."""
    rows = pd.concat(
        [read_observations(BytesIO(data), expected_unit=unit) for data in partitions]
    ).sort_values("observed_at", kind="stable")
    index = pd.DatetimeIndex(rows["observed_at"], name="time")
    frame = pd.DataFrame(
        {value_column: rows["value"].to_numpy(dtype="float64")}, index=index
    )
    # Compare naive local wall times so both folds of a fall-back hour, and the
    # local year's final hours from the next UTC partition, are placed by the
    # local calendar rather than by their UTC instant.
    local = index.tz_convert(timezone).tz_localize(None)
    inside = (local >= pd.Timestamp(year=year, month=1, day=1)) & (
        local < pd.Timestamp(year=year + 1, month=1, day=1)
    )
    return frame[inside]
