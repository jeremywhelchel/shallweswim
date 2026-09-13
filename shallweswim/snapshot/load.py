"""Load the current snapshot generation into frames and plots."""

import asyncio
import dataclasses
from typing import Literal

import pandas as pd

from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.snapshot.model import SCHEMA_VERSION, CurrentPointer, Manifest
from shallweswim.snapshot.serialize import parquet_to_frame
from shallweswim.snapshot.store import SnapshotStore, object_key


@dataclasses.dataclass(frozen=True)
class LoadedSnapshot:
    """The current generation's manifest with its frames and plots restored."""

    pointer: CurrentPointer
    manifest: Manifest
    frames: dict[str, dict[FeedName, pd.DataFrame]]
    plots: dict[str, dict[PlotName, bytes]]


async def _read_object(
    store: SnapshotStore, key: str, size_bytes: int, suffix: Literal["parquet", "svg"]
) -> bytes:
    """Read one referenced object and check it is the object the manifest names.

    Raises:
        ValueError: If the object is absent or its size or content differs
            from what the manifest recorded.
    """
    data = await store.read_object(key)
    if data is None:
        raise ValueError(f"Snapshot object missing: {key}")
    if len(data) != size_bytes:
        raise ValueError(
            f"Snapshot object {key} has {len(data)} bytes, manifest says {size_bytes}"
        )
    if object_key(data, suffix) != key:
        raise ValueError(f"Snapshot object content does not match its key: {key}")
    return data


async def load_current(store: SnapshotStore) -> LoadedSnapshot | None:
    """Load the generation the current pointer names, or None if none exists.

    Raises:
        ValueError: If the manifest is missing, has a schema version this code
            does not read, or references an object that is missing or differs
            from what it recorded.
    """
    current = await store.read_current()
    if current is None:
        return None
    pointer, _version = current
    manifest = await store.read_manifest(pointer.manifest_key)
    if manifest is None:
        raise ValueError(f"Snapshot manifest missing: {pointer.manifest_key}")
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot schema version {manifest.schema_version} is not {SCHEMA_VERSION}"
        )

    frames: dict[str, dict[FeedName, pd.DataFrame]] = {}
    plots: dict[str, dict[PlotName, bytes]] = {}
    for code, location in manifest.locations.items():
        frames[code] = {}
        for feed_name, feed in location.feeds.items():
            data = await _read_object(store, feed.key, feed.size_bytes, "parquet")
            frames[code][feed_name] = await asyncio.to_thread(
                parquet_to_frame, data, feed_name
            )
        plots[code] = {}
        for plot_name, plot in location.plots.items():
            plots[code][plot_name] = await _read_object(
                store, plot.key, plot.size_bytes, "svg"
            )
    return LoadedSnapshot(
        pointer=pointer, manifest=manifest, frames=frames, plots=plots
    )
