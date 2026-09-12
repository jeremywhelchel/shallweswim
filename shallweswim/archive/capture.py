"""Temperature capture for the transitional in-process feed updater."""

import asyncio
import datetime
from functools import cache
from urllib.parse import quote

import pandas as pd

from shallweswim.archive.merge import merge_observations
from shallweswim.archive.observations import (
    TEMPERATURE_UNIT,
    TEMPERATURE_VALUE_COLUMN,
    normalize_observations,
)
from shallweswim.archive.store import GcsObjectStore


@cache
def _store_for(bucket: str) -> GcsObjectStore:
    """Reuse the GCS client and connection pool for each bucket in this process."""
    return GcsObjectStore(bucket)


def _partitions(
    frame: pd.DataFrame,
    source_identity: str,
    timezone: datetime.tzinfo,
    retrieved_at: datetime.datetime,
) -> list[tuple[str, pd.DataFrame]]:
    """Normalize and partition by UTC year, preserving source identity."""
    provider, measurement, station = source_identity.split(":", 2)
    if measurement != "temperature" or not provider or not station:
        raise ValueError("Expected a temperature source identity")
    # Percent encoding is reversible, including USGS's station:parameter suffix.
    prefix = f"archive/temperature/{quote(provider, safe='')}/{quote(station, safe='')}"
    rows = normalize_observations(
        frame,
        value_column=TEMPERATURE_VALUE_COLUMN,
        unit=TEMPERATURE_UNIT,
        timezone=timezone,
        retrieved_at=retrieved_at,
    )
    return [
        (f"{prefix}/{year}.parquet", partition)
        for year, partition in rows.groupby(rows["observed_at"].dt.year)
    ]


async def capture_temperature(
    bucket: str,
    *,
    frame: pd.DataFrame,
    source_identity: str,
    timezone: datetime.tzinfo,
    retrieved_at: datetime.datetime,
) -> None:
    """Merge each UTC year; the feed caller isolates preparation failures."""
    partitions = await asyncio.to_thread(
        _partitions, frame, source_identity, timezone, retrieved_at
    )
    if not partitions:
        return
    store = await asyncio.to_thread(_store_for, bucket)
    for key, incoming in partitions:
        try:
            await merge_observations(
                store,
                key=key,
                source_identity=source_identity,
                incoming=incoming,
                expected_unit=TEMPERATURE_UNIT,
            )
        except Exception:
            # The merge writer already emitted the failed event. Continue so a
            # failed year does not prevent capture of other years in this fetch.
            continue
