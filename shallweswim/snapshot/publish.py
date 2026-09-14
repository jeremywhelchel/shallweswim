"""Publish one snapshot generation.

Objects are written first, the manifest second, and the current pointer last,
conditionally, so readers see either the complete previous generation or the
complete new one. A generation whose objects and metadata all match the
current one is not published at all.

A feed that failed this run keeps its last published entry: assembly copies it
from the base generation, the generation the publisher observed when it
started, and updates only the failure fields. The manifest then says both when
the served frame was fetched and that its source is failing now, which the
per-feed freshness event reports.
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
    LocationSnapshot,
    Manifest,
    PlotObject,
    Snapshot,
    generation_id,
)
from shallweswim.snapshot.serialize import frame_to_parquet
from shallweswim.snapshot.store import PromotionConflictError, SnapshotStore, object_key

# Bounded outcomes of the per-feed freshness event: the run fetched the feed,
# the manifest carries its last published entry forward, or the feed is
# configured with nothing to serve.
FRESHNESS_SUCCESS = "success"
FRESHNESS_CARRIED = "carried"
FRESHNESS_ABSENT = "absent"


@dataclasses.dataclass(frozen=True)
class PublishResult:
    """Outcome of one publish attempt.

    `objects_written` and `objects_reused` partition the distinct object keys
    this run serialized; carried-forward objects are referenced by key without
    being read or rewritten and are in neither count. `bytes_written` sums the
    objects and manifest actually stored. `reason` explains a failed outcome.
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


def _carry_forward(
    location: LocationManifest,
    snapshot: LocationSnapshot,
    base: LocationManifest | None,
) -> LocationManifest:
    """Fill one location's gaps from the base generation.

    A failed feed keeps the entry the base generation published for the same
    source, with this run's failure count added to the base entry's and this
    run's error and scheduled retry replacing it. A plot this run did not
    produce keeps the base generation's, whose `feed_fetch_timestamp` states
    which feed state it shows, but only while the feed it was drawn from is
    still in the assembled manifest; a plot never outlives its feed. Nothing is
    carried for a feed the base generation lacks or published from a different
    source, and a feed or location that is no longer configured is never
    resurrected, because it is not in the snapshot at all.

    Args:
        location: The feeds and plots this run produced for the location.
        snapshot: The location's built serving state, holding its failures.
        base: The same location in the base generation, if it has one.

    Returns:
        The location's assembled manifest entry.
    """
    if base is None:
        return location
    feed_objects = dict(location.feeds)
    for feed_name, failure in snapshot.failures.items():
        base_feed = base.feeds.get(feed_name)
        if base_feed is None or base_feed.source_identity != failure.source_identity:
            continue
        feed_objects[feed_name] = base_feed.model_copy(
            update={
                "consecutive_failures": (
                    base_feed.consecutive_failures + failure.consecutive_failures
                ),
                "last_error": failure.last_error,
                "next_fetch_after": failure.next_fetch_after,
            }
        )
    plot_objects = dict(location.plots)
    for plot_name, base_plot in base.plots.items():
        if plot_name in plot_objects or base_plot.feed not in feed_objects:
            continue
        plot_objects[plot_name] = base_plot
    return LocationManifest(feeds=feed_objects, plots=plot_objects)


def _log_freshness(
    manifest: Manifest, snapshot: Snapshot, now: datetime.datetime
) -> None:
    """Log one freshness event per location and configured feed.

    The event is emitted after assembly whether or not the generation is
    promoted, so a feed stuck on carried-forward data shows a growing age even
    while nothing else about the snapshot changes.

    Args:
        manifest: The assembled manifest, whose entries name the served frames.
        snapshot: The built serving state, distinguishing fetched from carried.
        now: The publication instant the ages are measured against.
    """
    for code, location in manifest.locations.items():
        built = snapshot.locations[code]
        for feed_name, feed_object in location.feeds.items():
            fetched = feed_name in built.feeds
            outcome = FRESHNESS_SUCCESS if fetched else FRESHNESS_CARRIED
            age_seconds = int((now - feed_object.fetch_timestamp).total_seconds())
            logging.log(
                logging.INFO if fetched else logging.WARNING,
                f"[{code}] {feed_name} freshness {outcome} (age {age_seconds}s)",
                extra={
                    "component": "snapshot",
                    "operation": "freshness",
                    "location": code,
                    "feed": feed_name,
                    "outcome": outcome,
                    "age_seconds": age_seconds,
                },
            )
        for feed_name in built.failures:
            if feed_name in location.feeds:
                continue
            logging.warning(
                f"[{code}] {feed_name} freshness {FRESHNESS_ABSENT}",
                extra={
                    "component": "snapshot",
                    "operation": "freshness",
                    "location": code,
                    "feed": feed_name,
                    "outcome": FRESHNESS_ABSENT,
                },
            )


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
        `skipped` when assembly produced a manifest referencing no object at
        all, or `failed` when a newer publisher promoted first. One structured
        `snapshot.publish` event is logged per attempt that reaches the store,
        and one `snapshot.freshness` event per location and configured feed
        once the manifest is assembled.

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
            locations={
                code: _carry_forward(
                    location,
                    snapshot.locations[code],
                    None
                    if current_manifest is None
                    else current_manifest.locations.get(code),
                )
                for code, location in locations.items()
            },
        )
        _log_freshness(manifest, snapshot, now)

        if not any(
            location.feeds or location.plots for location in manifest.locations.values()
        ):
            result = PublishResult(
                "skipped",
                new_generation_id,
                0,
                0,
                0,
                _duration_ms(started_at),
                reason="no location holds data",
            )
            logging.warning("No location holds data; nothing to publish")
            return result

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
