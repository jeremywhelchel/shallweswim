"""Scalar-observation capture for the transitional in-process feed updater."""

import asyncio
import datetime
import logging
from functools import cache
from urllib.parse import quote

import pandas as pd

from shallweswim.archive.merge import merge_observations
from shallweswim.archive.observations import normalize_observations
from shallweswim.archive.store import GcsObjectStore


@cache
def _store_for(bucket: str) -> GcsObjectStore:
    """Reuse the GCS client and connection pool for each bucket in this process."""
    return GcsObjectStore(bucket)


def _partitions(
    frame: pd.DataFrame,
    source_identity: str,
    measurement: str,
    value_column: str,
    unit: str,
    timezone: datetime.tzinfo,
    retrieved_at: datetime.datetime,
) -> list[tuple[str, pd.DataFrame]]:
    """Normalize and partition by UTC year, preserving source identity."""
    provider, source_measurement, station = source_identity.split(":", 2)
    if source_measurement != measurement or not provider or not station:
        raise ValueError(f"Expected a {measurement} source identity")
    # Percent encoding is reversible, including USGS's station:parameter suffix.
    prefix = (
        f"archive/{measurement}/{quote(provider, safe='')}/{quote(station, safe='')}"
    )
    normalized = normalize_observations(
        frame,
        value_column=value_column,
        unit=unit,
        timezone=timezone,
        retrieved_at=retrieved_at,
    )
    if normalized.ambiguous_dropped:
        logging.warning(
            "Archive normalization dropped %d unresolvable ambiguous row(s) for %s",
            normalized.ambiguous_dropped,
            source_identity,
            extra={
                "component": "archive",
                "operation": "normalize",
                "source_identity": source_identity,
                "outcome": "ambiguous_dropped",
                "record_count": normalized.ambiguous_dropped,
            },
        )
    if normalized.conflicting_dropped:
        logging.warning(
            "Archive normalization dropped %d conflicting repeated row(s) for %s",
            normalized.conflicting_dropped,
            source_identity,
            extra={
                "component": "archive",
                "operation": "normalize",
                "source_identity": source_identity,
                "outcome": "conflict_dropped",
                "record_count": normalized.conflicting_dropped,
            },
        )
    rows = normalized.frame
    return [
        (f"{prefix}/{year}.parquet", partition)
        for year, partition in rows.groupby(rows["observed_at"].dt.year)
    ]


async def capture_observations(
    bucket: str,
    *,
    frame: pd.DataFrame,
    source_identity: str,
    measurement: str,
    value_column: str,
    unit: str,
    timezone: datetime.tzinfo,
    retrieved_at: datetime.datetime,
) -> None:
    """Merge each UTC year; the feed caller isolates preparation failures."""
    partitions = await asyncio.to_thread(
        _partitions,
        frame,
        source_identity,
        measurement,
        value_column,
        unit,
        timezone,
        retrieved_at,
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
                expected_unit=unit,
            )
        except Exception:
            # The merge writer already emitted the failed event. Continue so a
            # failed year does not prevent capture of other years in this fetch.
            continue
