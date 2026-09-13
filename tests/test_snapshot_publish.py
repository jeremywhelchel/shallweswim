"""Publication and load of snapshot generations on the memory store."""

import datetime
import logging
from unittest.mock import patch

import pandas as pd
import pytest

from shallweswim.archive.store import MemoryObjectStore
from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.snapshot.load import load_current
from shallweswim.snapshot.model import SCHEMA_VERSION, CurrentPointer
from shallweswim.snapshot.publish import PublishResult, publish
from shallweswim.snapshot.store import CURRENT_KEY, SnapshotStore, manifest_key
from tests.snapshot_fixtures import (
    FETCHED_AT,
    SAMPLE_OBJECT_COUNT,
    sample_snapshot,
)

NOW = datetime.datetime(2026, 9, 13, 15, 0, 0, tzinfo=datetime.UTC)
LATER = NOW + datetime.timedelta(hours=1)


def _publish_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "component", None) == "snapshot"
    ]


def _assert_event(
    record: logging.LogRecord, result: PublishResult, run_id: str, level: int
) -> None:
    assert record.levelno == level
    assert record.operation == "publish"
    assert record.outcome == result.outcome
    assert record.duration_ms == result.duration_ms
    assert record.record_count == result.objects_written
    assert record.generation_id == result.generation_id
    assert record.run_id == run_id


@pytest.mark.asyncio
async def test_first_publish_writes_every_object_and_promotes(caplog) -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)

    with caplog.at_level(logging.INFO):
        result = await publish(store, sample_snapshot(), run_id="run-1", now=NOW)

    assert result.outcome == "success"
    assert result.generation_id == "20260913T150000Z-run-1"
    assert result.objects_written == SAMPLE_OBJECT_COUNT
    assert result.objects_reused == 0
    assert result.reason is None
    keys = sorted(objects._objects)
    assert keys.count(CURRENT_KEY) == 1
    assert [key for key in keys if key.startswith("published/manifests/")] == [
        manifest_key(result.generation_id)
    ]
    assert len([key for key in keys if key.startswith("published/objects/")]) == (
        SAMPLE_OBJECT_COUNT
    )
    manifest_bytes = objects._objects[manifest_key(result.generation_id)]
    assert result.bytes_written == len(manifest_bytes) + sum(
        len(objects._objects[key])
        for key in keys
        if key.startswith("published/objects/")
    )

    current = await store.read_current()
    assert current is not None
    assert current[0] == CurrentPointer(
        manifest_key=manifest_key(result.generation_id),
        generation_id=result.generation_id,
    )
    manifest = await store.read_manifest(current[0].manifest_key)
    assert manifest is not None
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.published_at == NOW
    assert manifest.previous_generation_id is None
    location = manifest.locations["nyc"]
    assert set(location.feeds) == set(FeedName)
    assert set(location.plots) == set(PlotName)
    assert location.plots[PlotName.LIVE_TEMPS].feed is FeedName.LIVE_TEMPS
    assert location.plots[PlotName.LIVE_TEMPS].feed_fetch_timestamp == FETCHED_AT
    for feed in location.feeds.values():
        assert feed.size_bytes == len(objects._objects[feed.key])

    events = _publish_events(caplog)
    assert len(events) == 1
    _assert_event(events[0], result, "run-1", logging.INFO)


@pytest.mark.asyncio
async def test_identical_publish_is_unchanged_and_writes_nothing(caplog) -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    first = await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    before = dict(objects._objects)

    with caplog.at_level(logging.INFO):
        result = await publish(store, sample_snapshot(), run_id="run-2", now=LATER)

    assert result.outcome == "unchanged"
    assert result.generation_id == "20260913T160000Z-run-2"
    assert result.objects_written == 0
    assert result.objects_reused == SAMPLE_OBJECT_COUNT
    assert result.bytes_written == 0
    assert objects._objects == before
    current = await store.read_current()
    assert current is not None
    assert current[0].generation_id == first.generation_id
    events = _publish_events(caplog)
    assert len(events) == 1
    _assert_event(events[0], result, "run-2", logging.INFO)


@pytest.mark.asyncio
async def test_changed_frame_writes_one_object_and_reuses_the_rest() -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    first = await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    before = dict(objects._objects)

    result = await publish(
        store, sample_snapshot(live_offset=0.5), run_id="run-2", now=LATER
    )

    assert result.outcome == "success"
    assert result.objects_written == 1
    assert result.objects_reused == SAMPLE_OBJECT_COUNT - 1
    added = set(objects._objects) - set(before)
    assert len(added) == 2
    assert manifest_key(result.generation_id) in added
    assert all(
        before[key] == objects._objects[key] for key in before if key != CURRENT_KEY
    )
    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    assert manifest.previous_generation_id == first.generation_id
    previous = await store.read_manifest(manifest_key(first.generation_id))
    assert previous is not None
    changed = {
        name
        for name, feed in manifest.locations["nyc"].feeds.items()
        if feed.key != previous.locations["nyc"].feeds[name].key
    }
    assert changed == {FeedName.LIVE_TEMPS}
    assert manifest.locations["nyc"].plots == previous.locations["nyc"].plots
    current = await store.read_current()
    assert current is not None
    assert current[0].generation_id == result.generation_id


@pytest.mark.asyncio
async def test_metadata_only_change_publishes_a_manifest_without_objects() -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    before = dict(objects._objects)

    refreshed = sample_snapshot(
        live_fetch_timestamp=FETCHED_AT + datetime.timedelta(minutes=10)
    )
    result = await publish(store, refreshed, run_id="run-2", now=LATER)

    assert result.outcome == "success"
    assert result.objects_written == 0
    assert result.objects_reused == SAMPLE_OBJECT_COUNT
    assert set(objects._objects) - set(before) == {manifest_key(result.generation_id)}
    assert result.bytes_written == len(
        objects._objects[manifest_key(result.generation_id)]
    )


@pytest.mark.asyncio
async def test_newer_publisher_wins_the_promotion(caplog) -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    await publish(store, sample_snapshot(), run_id="run-0", now=NOW)
    original_read_current = SnapshotStore.read_current
    other_generation: list[str] = []

    async def read_then_lose_the_race(self: SnapshotStore):  # type: ignore[no-untyped-def]
        # The late publisher observes the pointer, then another publisher
        # (which reads through the same patched method, so it runs only once)
        # promotes a newer generation before the late one promotes.
        observed = await original_read_current(self)
        if not other_generation:
            other_generation.append("racing")
            other = await publish(
                SnapshotStore(objects),
                sample_snapshot(live_offset=2.0),
                run_id="run-other",
                now=LATER,
            )
            other_generation[0] = other.generation_id
        return observed

    with (
        patch.object(SnapshotStore, "read_current", read_then_lose_the_race),
        caplog.at_level(logging.INFO),
    ):
        result = await publish(
            store, sample_snapshot(live_offset=1.0), run_id="run-late", now=LATER
        )

    assert result.outcome == "failed"
    assert result.reason is not None
    assert "current pointer changed" in result.reason
    assert result.objects_written == 1
    current = await store.read_current()
    assert current is not None
    assert current[0].generation_id == other_generation[0]
    assert await store.read_manifest(manifest_key(result.generation_id)) is not None
    events = [
        record
        for record in _publish_events(caplog)
        if record.generation_id == result.generation_id
    ]
    assert len(events) == 1
    _assert_event(events[0], result, "run-late", logging.ERROR)


@pytest.mark.asyncio
async def test_unexpected_store_failure_logs_failed_and_raises(caplog) -> None:
    store = SnapshotStore(MemoryObjectStore())

    async def broken(self: SnapshotStore, key: str, data: bytes) -> bool:
        raise RuntimeError("bucket offline")

    with (
        patch.object(SnapshotStore, "write_object", broken),
        caplog.at_level(logging.INFO),
        pytest.raises(RuntimeError, match="bucket offline"),
    ):
        await publish(store, sample_snapshot(), run_id="run-1", now=NOW)

    events = _publish_events(caplog)
    assert len(events) == 1
    assert events[0].levelno == logging.ERROR
    assert events[0].outcome == "failed"
    assert events[0].run_id == "run-1"
    assert await store.read_current() is None


@pytest.mark.asyncio
async def test_generation_ids_differ_within_one_second() -> None:
    store = SnapshotStore(MemoryObjectStore())

    first = await publish(store, sample_snapshot(), run_id="run-a", now=NOW)
    second = await publish(
        store, sample_snapshot(live_offset=0.5), run_id="run-b", now=NOW
    )

    assert first.generation_id == "20260913T150000Z-run-a"
    assert second.generation_id == "20260913T150000Z-run-b"
    manifest = await store.read_manifest(manifest_key(second.generation_id))
    assert manifest is not None
    assert manifest.previous_generation_id == first.generation_id


@pytest.mark.asyncio
async def test_missing_current_manifest_still_publishes() -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    first = await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    del objects._objects[manifest_key(first.generation_id)]

    result = await publish(store, sample_snapshot(), run_id="run-2", now=LATER)

    assert result.outcome == "success"
    assert result.objects_written == 0
    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    assert manifest.previous_generation_id == first.generation_id


@pytest.mark.asyncio
async def test_load_current_restores_frames_and_plots() -> None:
    store = SnapshotStore(MemoryObjectStore())
    assert await load_current(store) is None
    snapshot = sample_snapshot()
    result = await publish(store, snapshot, run_id="run-1", now=NOW)

    loaded = await load_current(store)

    assert loaded is not None
    assert loaded.pointer.generation_id == result.generation_id
    assert loaded.manifest.generation_id == result.generation_id
    location = snapshot.locations["nyc"]
    assert set(loaded.frames["nyc"]) == set(location.feeds)
    for feed_name, feed in location.feeds.items():
        pd.testing.assert_frame_equal(
            feed.frame, loaded.frames["nyc"][feed_name], check_freq=False
        )
    assert loaded.plots["nyc"] == {
        name: plot.data for name, plot in location.plots.items()
    }


@pytest.mark.asyncio
async def test_load_current_rejects_objects_that_differ_from_the_manifest() -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    loaded = await load_current(store)
    assert loaded is not None
    plot_key = loaded.manifest.locations["nyc"].plots[PlotName.LIVE_TEMPS].key

    objects._objects[plot_key] = b"<svg>tampered</svg>"
    with pytest.raises(ValueError, match="bytes, manifest says"):
        await load_current(store)

    objects._objects[plot_key] = b"<svg>livX</svg>"
    with pytest.raises(ValueError, match="does not match its key"):
        await load_current(store)

    del objects._objects[plot_key]
    with pytest.raises(ValueError, match="missing"):
        await load_current(store)
