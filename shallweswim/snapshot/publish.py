"""Publish one snapshot generation.

Objects are written first, the manifest second, and the current pointer last,
conditionally, so readers see either the complete previous generation or the
complete new one. A generation whose objects and metadata all match the
current one is not published at all.
"""

import asyncio
import dataclasses
import datetime
import logging
import time

from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.snapshot.model import (
    SCHEMA_VERSION,
    CurrentPointer,
    FeedObject,
    LocationManifest,
    Manifest,
    PlotObject,
    Snapshot,
    generation_id,
)
from shallweswim.snapshot.serialize import frame_to_parquet
from shallweswim.snapshot.store import PromotionConflictError, SnapshotStore, object_key


@dataclasses.dataclass(frozen=True)
class PublishResult:
    """Outcome of one publish attempt.

    `objects_written` and `objects_reused` partition the generation's distinct
    object keys; `bytes_written` sums the objects and manifest actually stored.
    `reason` explains a failed outcome.
    """

    outcome: str
    generation_id: str
    objects_written: int
    objects_reused: int
    bytes_written: int
    duration_ms: int
    reason: str | None = None


def _event_fields(result: PublishResult, run_id: str) -> dict[str, object]:
    return {
        "component": "snapshot",
        "operation": "publish",
        "outcome": result.outcome,
        "duration_ms": result.duration_ms,
        "record_count": result.objects_written,
        "generation_id": result.generation_id,
        "run_id": run_id,
    }


def _duration_ms(started_at: float) -> int:
    return max(0, round((time.monotonic() - started_at) * 1000))


async def _serialize(
    snapshot: Snapshot,
) -> tuple[dict[str, LocationManifest], dict[str, bytes]]:
    """Serialize every frame and hash every object, keyed for the manifest."""
    locations: dict[str, LocationManifest] = {}
    objects: dict[str, bytes] = {}
    for code, location in snapshot.locations.items():
        feed_objects: dict[FeedName, FeedObject] = {}
        for feed_name, feed in location.feeds.items():
            data = await asyncio.to_thread(frame_to_parquet, feed.frame)
            key = object_key(data, "parquet")
            objects[key] = data
            feed_objects[feed_name] = FeedObject(
                key=key, size_bytes=len(data), **feed.metadata.model_dump()
            )
        plot_objects: dict[PlotName, PlotObject] = {}
        for plot_name, plot in location.plots.items():
            key = object_key(plot.data, "svg")
            objects[key] = plot.data
            plot_objects[plot_name] = PlotObject(
                key=key,
                size_bytes=len(plot.data),
                feed=plot.feed,
                feed_fetch_timestamp=plot.feed_fetch_timestamp,
            )
        locations[code] = LocationManifest(feeds=feed_objects, plots=plot_objects)
    return locations, objects


async def publish(
    store: SnapshotStore,
    snapshot: Snapshot,
    *,
    run_id: str,
    now: datetime.datetime,
) -> PublishResult:
    """Publish `snapshot` as a new generation unless it equals the current one.

    Args:
        store: The snapshot store to publish into.
        snapshot: The serving state of every location to publish.
        run_id: The publishing job's run identifier, part of the generation id.
        now: The timezone-aware publication instant.

    Returns:
        The attempt's outcome: `success`, `unchanged` when nothing was written,
        or `failed` when a newer publisher promoted first. Exactly one
        structured `snapshot.publish` event is logged per attempt.

    Raises:
        Exception: Any store or serialization failure, after logging the failed
            event; the caller decides whether the run continues.
    """
    started_at = time.monotonic()
    new_generation_id = generation_id(now, run_id)
    try:
        current = await store.read_current()
        current_version: str | None = None
        current_manifest: Manifest | None = None
        previous_generation_id: str | None = None
        if current is not None:
            pointer, current_version = current
            previous_generation_id = pointer.generation_id
            current_manifest = await store.read_manifest(pointer.manifest_key)

        locations, objects = await _serialize(snapshot)
        manifest = Manifest(
            schema_version=SCHEMA_VERSION,
            generation_id=new_generation_id,
            published_at=now,
            previous_generation_id=previous_generation_id,
            locations=locations,
        )

        if (
            current_manifest is not None
            and current_manifest.schema_version == manifest.schema_version
            and current_manifest.locations == manifest.locations
        ):
            result = PublishResult(
                "unchanged",
                new_generation_id,
                0,
                len(objects),
                0,
                _duration_ms(started_at),
            )
            logging.info(
                "Snapshot unchanged from generation %s",
                previous_generation_id,
                extra=_event_fields(result, run_id),
            )
            return result

        objects_written = 0
        bytes_written = 0
        for key, data in objects.items():
            if await store.write_object(key, data):
                objects_written += 1
                bytes_written += len(data)
        key, manifest_data = await store.write_manifest(manifest)
        bytes_written += len(manifest_data)
        objects_reused = len(objects) - objects_written

        try:
            await store.promote(
                CurrentPointer(manifest_key=key, generation_id=new_generation_id),
                current_version,
            )
        except PromotionConflictError as error:
            result = PublishResult(
                "failed",
                new_generation_id,
                objects_written,
                objects_reused,
                bytes_written,
                _duration_ms(started_at),
                reason=str(error),
            )
            logging.error(
                "Snapshot publish failed for generation %s: %s",
                new_generation_id,
                error,
                extra=_event_fields(result, run_id),
            )
            return result

        result = PublishResult(
            "success",
            new_generation_id,
            objects_written,
            objects_reused,
            bytes_written,
            _duration_ms(started_at),
        )
        logging.info(
            "Snapshot published generation %s",
            new_generation_id,
            extra=_event_fields(result, run_id),
        )
        return result
    except Exception as error:
        result = PublishResult(
            "failed", new_generation_id, 0, 0, 0, _duration_ms(started_at), str(error)
        )
        logging.error(
            "Snapshot publish failed for generation %s: %s",
            new_generation_id,
            error,
            extra=_event_fields(result, run_id),
        )
        raise
