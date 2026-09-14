"""Load the current snapshot generation into frames and plots.

A load is incremental. Objects are content-addressed, so a key already held by
a previously loaded generation names byte-identical content: the decoded frame
or plot bytes are reused and the object is never read again. Only the keys a
load has not seen are fetched, with bounded concurrency, and every fetched
object is checked against the size and key the manifest recorded.
"""

import asyncio
import dataclasses
from typing import Literal

import pandas as pd

from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.snapshot.model import (
    SCHEMA_VERSION,
    CurrentPointer,
    FeedObject,
    Manifest,
    PlotObject,
)
from shallweswim.snapshot.serialize import parquet_to_frame
from shallweswim.snapshot.store import SnapshotStore, object_key

# Objects read at once. The web loads on an elected request, so the load shares
# the instance's CPU and connection pool with serving.
MAX_CONCURRENT_READS = 8


@dataclasses.dataclass(frozen=True)
class LoadedSnapshot:
    """The current generation's manifest with its frames and plots restored."""

    pointer: CurrentPointer
    manifest: Manifest
    frames: dict[str, dict[FeedName, pd.DataFrame]]
    plots: dict[str, dict[PlotName, bytes]]
    objects_read: int


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


async def _load_frame(
    store: SnapshotStore,
    semaphore: asyncio.Semaphore,
    feed: FeedObject,
    feed_name: FeedName,
) -> pd.DataFrame:
    """Read one Parquet object and validate it as the named feed's frame."""
    async with semaphore:
        data = await _read_object(store, feed.key, feed.size_bytes, "parquet")
        return await asyncio.to_thread(parquet_to_frame, data, feed_name)


async def _load_plot(
    store: SnapshotStore, semaphore: asyncio.Semaphore, plot: PlotObject
) -> bytes:
    """Read one SVG object."""
    async with semaphore:
        return await _read_object(store, plot.key, plot.size_bytes, "svg")


def _held_frames(
    previous: LoadedSnapshot | None,
) -> dict[str, pd.DataFrame]:
    """Map each object key the previous generation holds to its decoded frame."""
    if previous is None:
        return {}
    return {
        feed.key: previous.frames[code][feed_name]
        for code, location in previous.manifest.locations.items()
        for feed_name, feed in location.feeds.items()
    }


def _held_plots(previous: LoadedSnapshot | None) -> dict[str, bytes]:
    """Map each object key the previous generation holds to its plot bytes."""
    if previous is None:
        return {}
    return {
        plot.key: previous.plots[code][plot_name]
        for code, location in previous.manifest.locations.items()
        for plot_name, plot in location.plots.items()
    }


async def load_current(
    store: SnapshotStore, previous: LoadedSnapshot | None = None
) -> LoadedSnapshot | None:
    """Load the generation the current pointer names.

    Args:
        store: The snapshot store to read from.
        previous: A generation this process already holds, whose objects are
            reused by key rather than read again.

    Returns:
        None if no generation is published; `previous` itself, unchanged, when
        the current pointer still names the generation `previous` holds, which
        callers detect by identity; otherwise the newly loaded generation.
        `objects_read` counts the objects this load read, so it is meaningful
        only on a newly loaded generation.

    Raises:
        ValueError: If the manifest is missing, has a schema version this code
            does not read, or references an object that is missing or differs
            from what it recorded.
    """
    current = await store.read_current()
    if current is None:
        return None
    pointer, _version = current
    if previous is not None and pointer.generation_id == previous.pointer.generation_id:
        return previous
    manifest = await store.read_manifest(pointer.manifest_key)
    if manifest is None:
        raise ValueError(f"Snapshot manifest missing: {pointer.manifest_key}")
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot schema version {manifest.schema_version} is not {SCHEMA_VERSION}"
        )

    held_frames = _held_frames(previous)
    held_plots = _held_plots(previous)
    # Each new key is read once even when several manifest entries name it.
    new_feeds: dict[str, tuple[FeedObject, FeedName]] = {}
    new_plots: dict[str, PlotObject] = {}
    for location in manifest.locations.values():
        for feed_name, feed in location.feeds.items():
            if feed.key not in held_frames:
                new_feeds.setdefault(feed.key, (feed, feed_name))
        for plot in location.plots.values():
            if plot.key not in held_plots:
                new_plots.setdefault(plot.key, plot)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_READS)
    feed_keys = list(new_feeds)
    read_frames = dict(
        zip(
            feed_keys,
            await asyncio.gather(
                *(_load_frame(store, semaphore, *new_feeds[key]) for key in feed_keys)
            ),
            strict=True,
        )
    )
    plot_keys = list(new_plots)
    read_plots = dict(
        zip(
            plot_keys,
            await asyncio.gather(
                *(_load_plot(store, semaphore, new_plots[key]) for key in plot_keys)
            ),
            strict=True,
        )
    )

    frames: dict[str, dict[FeedName, pd.DataFrame]] = {}
    plots: dict[str, dict[PlotName, bytes]] = {}
    for code, location in manifest.locations.items():
        frames[code] = {
            feed_name: (
                held_frames[feed.key]
                if feed.key in held_frames
                else read_frames[feed.key]
            )
            for feed_name, feed in location.feeds.items()
        }
        plots[code] = {
            plot_name: (
                held_plots[plot.key] if plot.key in held_plots else read_plots[plot.key]
            )
            for plot_name, plot in location.plots.items()
        }
    return LoadedSnapshot(
        pointer=pointer,
        manifest=manifest,
        frames=frames,
        plots=plots,
        objects_read=len(feed_keys) + len(plot_keys),
    )
