"""Publication and load of snapshot generations on the memory store."""

import dataclasses
import datetime
import logging
from unittest.mock import patch

import pandas as pd
import pytest

from shallweswim.archive.store import MemoryObjectStore
from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.snapshot.load import load_current
from shallweswim.snapshot.model import SCHEMA_VERSION, CurrentPointer, Snapshot
from shallweswim.snapshot.publish import PublishResult, publish
from shallweswim.snapshot.store import CURRENT_KEY, SnapshotStore, manifest_key
from tests.snapshot_fixtures import (
    FETCHED_AT,
    RETRY_AT,
    SAMPLE_OBJECT_COUNT,
    feed_failure,
    feed_hold,
    sample_snapshot,
)

NOW = datetime.datetime(2026, 9, 13, 15, 0, 0, tzinfo=datetime.UTC)
LATER = NOW + datetime.timedelta(hours=1)


def _publish_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "component", None) == "snapshot"
        and record.operation == "publish"
    ]


def _freshness_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "component", None) == "snapshot"
        and record.operation == "freshness"
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


# =============================================================================
# Carry-forward of failed feeds and the per-feed freshness event
# =============================================================================


def _freshness(caplog: pytest.LogCaptureFixture) -> dict[str, logging.LogRecord]:
    """The run's freshness events keyed by feed name."""
    return {record.feed: record for record in _freshness_events(caplog)}


@pytest.mark.asyncio
async def test_failed_feed_carries_the_base_entry_forward(caplog) -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    first = await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    base = await store.read_manifest(manifest_key(first.generation_id))
    assert base is not None
    before = dict(objects._objects)

    failing = sample_snapshot(
        failures={
            FeedName.LIVE_TEMPS: feed_failure(
                FeedName.LIVE_TEMPS, consecutive_failures=2, last_error="boom"
            )
        }
    )
    with caplog.at_level(logging.INFO):
        result = await publish(store, failing, run_id="run-2", now=LATER)

    # A carried-forward entry changes the manifest without writing any object.
    assert result.outcome == "success"
    assert result.objects_written == 0
    assert result.objects_reused == SAMPLE_OBJECT_COUNT - 2
    assert set(objects._objects) - set(before) == {manifest_key(result.generation_id)}
    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    base_feed = base.locations["nyc"].feeds[FeedName.LIVE_TEMPS]
    carried = manifest.locations["nyc"].feeds[FeedName.LIVE_TEMPS]
    assert carried == base_feed.model_copy(
        update={
            "consecutive_failures": 2,
            "last_error": "boom",
            "next_fetch_after": RETRY_AT,
        }
    )
    assert carried.key == base_feed.key
    assert carried.size_bytes == base_feed.size_bytes
    assert carried.fetch_timestamp == FETCHED_AT
    assert carried.record_count == base_feed.record_count
    # The plot drawn from the failed feed comes along unchanged.
    assert manifest.locations["nyc"].plots == base.locations["nyc"].plots

    loaded = await load_current(store)
    assert loaded is not None
    pd.testing.assert_frame_equal(
        loaded.frames["nyc"][FeedName.LIVE_TEMPS],
        sample_snapshot().locations["nyc"].feeds[FeedName.LIVE_TEMPS].frame,
        check_freq=False,
    )

    events = _freshness(caplog)
    assert set(events) == {name.value for name in FeedName}
    live = events[FeedName.LIVE_TEMPS]
    # The carried frame is months older than the feed's ten-minute interval
    # plus the health buffer, so the verdict is stale.
    assert live.outcome == "stale"
    assert live.levelno == logging.WARNING
    assert live.age_seconds == int((LATER - FETCHED_AT).total_seconds())
    assert live.location == "nyc"
    assert f"(age {live.age_seconds}s)" in live.getMessage()
    tides = events[FeedName.TIDES]
    assert tides.outcome == "success"
    assert tides.levelno == logging.INFO
    assert tides.age_seconds == int((LATER - FETCHED_AT).total_seconds())


@pytest.mark.asyncio
async def test_repeated_failures_accumulate_on_the_carried_entry() -> None:
    store = SnapshotStore(MemoryObjectStore())
    await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    await publish(
        store,
        sample_snapshot(
            failures={
                FeedName.LIVE_TEMPS: feed_failure(
                    FeedName.LIVE_TEMPS, consecutive_failures=2
                )
            }
        ),
        run_id="run-2",
        now=LATER,
    )

    result = await publish(
        store,
        sample_snapshot(
            failures={
                FeedName.LIVE_TEMPS: feed_failure(
                    FeedName.LIVE_TEMPS,
                    consecutive_failures=3,
                    last_error="still failing",
                    next_fetch_after=None,
                )
            }
        ),
        run_id="run-3",
        now=LATER + datetime.timedelta(hours=1),
    )

    assert result.outcome == "success"
    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    carried = manifest.locations["nyc"].feeds[FeedName.LIVE_TEMPS]
    assert carried.consecutive_failures == 5
    assert carried.last_error == "still failing"
    assert carried.next_fetch_after is None
    assert carried.fetch_timestamp == FETCHED_AT


@pytest.mark.asyncio
async def test_nothing_is_carried_when_the_base_lacks_the_feed(caplog) -> None:
    store = SnapshotStore(MemoryObjectStore())
    failure = {FeedName.LIVE_TEMPS: feed_failure(FeedName.LIVE_TEMPS)}
    first = await publish(
        store, sample_snapshot(failures=failure), run_id="run-1", now=NOW
    )

    with caplog.at_level(logging.INFO):
        result = await publish(
            store, sample_snapshot(failures=failure), run_id="run-2", now=LATER
        )

    assert first.outcome == "success"
    assert result.outcome == "unchanged"
    manifest = await store.read_manifest(manifest_key(first.generation_id))
    assert manifest is not None
    assert FeedName.LIVE_TEMPS not in manifest.locations["nyc"].feeds
    assert PlotName.LIVE_TEMPS not in manifest.locations["nyc"].plots
    live = _freshness(caplog)[FeedName.LIVE_TEMPS]
    assert live.outcome == "absent"
    assert live.levelno == logging.WARNING
    assert not hasattr(live, "age_seconds")


@pytest.mark.asyncio
async def test_nothing_is_carried_when_the_source_identity_differs() -> None:
    store = SnapshotStore(MemoryObjectStore())
    await publish(store, sample_snapshot(), run_id="run-1", now=NOW)

    result = await publish(
        store,
        sample_snapshot(
            failures={
                FeedName.LIVE_TEMPS: feed_failure(
                    FeedName.LIVE_TEMPS, source_identity="coops:live_temps:9999999"
                )
            }
        ),
        run_id="run-2",
        now=LATER,
    )

    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    assert FeedName.LIVE_TEMPS not in manifest.locations["nyc"].feeds


@pytest.mark.asyncio
async def test_feeds_and_locations_no_longer_configured_are_dropped() -> None:
    store = SnapshotStore(MemoryObjectStore())
    base_location = sample_snapshot().locations["nyc"]
    await publish(
        store,
        Snapshot(locations={"nyc": base_location, "obs": base_location}),
        run_id="run-1",
        now=NOW,
    )

    # The location keeps only two feeds, and the other location is gone.
    reduced = dataclasses.replace(
        base_location,
        feeds={
            name: feed
            for name, feed in base_location.feeds.items()
            if name in {FeedName.TIDES, FeedName.CURRENTS}
        },
        plots={},
    )
    result = await publish(
        store, Snapshot(locations={"nyc": reduced}), run_id="run-2", now=LATER
    )

    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    assert set(manifest.locations) == {"nyc"}
    assert set(manifest.locations["nyc"].feeds) == {FeedName.TIDES, FeedName.CURRENTS}
    # Every base plot was drawn from a feed that is no longer published, and a
    # plot never outlives its feed.
    assert manifest.locations["nyc"].plots == {}


@pytest.mark.asyncio
async def test_a_plot_this_run_did_not_produce_is_copied_from_the_base() -> None:
    store = SnapshotStore(MemoryObjectStore())
    first = await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    base = await store.read_manifest(manifest_key(first.generation_id))
    assert base is not None

    # Every feed refreshed, but one plot did not complete in time.
    refetched = FETCHED_AT + datetime.timedelta(minutes=10)
    location = sample_snapshot(live_fetch_timestamp=refetched).locations["nyc"]
    incomplete = dataclasses.replace(
        location,
        plots={
            name: plot
            for name, plot in location.plots.items()
            if name is not PlotName.LIVE_TEMPS
        },
    )
    result = await publish(
        store, Snapshot(locations={"nyc": incomplete}), run_id="run-2", now=LATER
    )

    assert result.outcome == "success"
    assert result.objects_written == 0
    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    # The feed the plot was drawn from is still published, so the plot is too.
    assert set(manifest.locations["nyc"].plots) == set(PlotName)
    carried = manifest.locations["nyc"].plots[PlotName.LIVE_TEMPS]
    assert carried == base.locations["nyc"].plots[PlotName.LIVE_TEMPS]
    # The copied plot states the feed state it shows, which is now behind the
    # feed's own refreshed fetch timestamp.
    assert carried.feed_fetch_timestamp == FETCHED_AT
    assert manifest.locations["nyc"].feeds[FeedName.LIVE_TEMPS].fetch_timestamp == (
        refetched
    )


@pytest.mark.asyncio
async def test_a_plot_is_carried_only_while_its_feed_is_published() -> None:
    store = SnapshotStore(MemoryObjectStore())
    first = await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    base = await store.read_manifest(manifest_key(first.generation_id))
    assert base is not None

    # live_temps fails and is carried forward, so its plot comes with it;
    # historic_temps is no longer configured, so both of its plots are dropped.
    failing = sample_snapshot(
        failures={FeedName.LIVE_TEMPS: feed_failure(FeedName.LIVE_TEMPS)}
    )
    location = failing.locations["nyc"]
    deconfigured = dataclasses.replace(
        location,
        feeds={
            name: feed
            for name, feed in location.feeds.items()
            if name is not FeedName.HISTORIC_TEMPS
        },
        plots={},
    )
    result = await publish(
        store, Snapshot(locations={"nyc": deconfigured}), run_id="run-2", now=LATER
    )

    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    assert set(manifest.locations["nyc"].feeds) == {
        FeedName.LIVE_TEMPS,
        FeedName.TIDES,
        FeedName.CURRENTS,
    }
    assert set(manifest.locations["nyc"].plots) == {PlotName.LIVE_TEMPS}
    assert (
        manifest.locations["nyc"].plots[PlotName.LIVE_TEMPS]
        == base.locations["nyc"].plots[PlotName.LIVE_TEMPS]
    )


@pytest.mark.asyncio
async def test_a_location_whose_feeds_all_failed_without_a_base_publishes_nothing(
    caplog,
) -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    failures = {name: feed_failure(name) for name in FeedName}

    with caplog.at_level(logging.INFO):
        result = await publish(
            store, sample_snapshot(failures=failures), run_id="run-1", now=NOW
        )

    assert result.outcome == "skipped"
    assert result.objects_written == 0
    assert objects._objects == {}
    assert _publish_events(caplog) == []
    events = _freshness(caplog)
    assert {record.outcome for record in events.values()} == {"absent"}
    assert set(events) == {name.value for name in FeedName}


# =============================================================================
# Feeds the run held because they were not due
# =============================================================================


@pytest.mark.asyncio
async def test_held_feed_keeps_its_base_entry_and_plots_exactly(caplog) -> None:
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    first = await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    base = await store.read_manifest(manifest_key(first.generation_id))
    assert base is not None

    # live_temps was not due this run; every other feed refetched.
    held = sample_snapshot(holds={FeedName.LIVE_TEMPS: feed_hold(FeedName.LIVE_TEMPS)})
    with caplog.at_level(logging.INFO):
        result = await publish(store, held, run_id="run-2", now=LATER)

    # Nothing about the held feed changed, so the generation is identical.
    assert result.outcome == "unchanged"
    manifest = await store.read_manifest(manifest_key(first.generation_id))
    assert manifest is not None
    assert (
        manifest.locations["nyc"].feeds[FeedName.LIVE_TEMPS]
        == (base.locations["nyc"].feeds[FeedName.LIVE_TEMPS])
    )
    assert manifest.locations["nyc"].plots == base.locations["nyc"].plots

    events = _freshness(caplog)
    live = events[FeedName.LIVE_TEMPS]
    assert live.outcome == "held"
    assert live.levelno == logging.INFO
    assert live.age_seconds == int((LATER - FETCHED_AT).total_seconds())
    assert events[FeedName.TIDES].outcome == "success"


@pytest.mark.asyncio
async def test_a_run_that_holds_every_feed_publishes_nothing(caplog) -> None:
    """The `unchanged` outcome becomes reachable once no feed is due."""
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    await publish(store, sample_snapshot(), run_id="run-1", now=NOW)
    before = dict(objects._objects)

    holds = {name: feed_hold(name) for name in FeedName}
    with caplog.at_level(logging.INFO):
        result = await publish(
            store, sample_snapshot(holds=holds), run_id="run-2", now=LATER
        )

    assert result.outcome == "unchanged"
    assert result.objects_written == 0
    assert objects._objects == before
    events = _freshness(caplog)
    assert {record.outcome for record in events.values()} == {"held"}
    assert all(record.levelno == logging.INFO for record in events.values())


@pytest.mark.asyncio
async def test_a_held_feed_the_base_never_published_is_absent(caplog) -> None:
    store = SnapshotStore(MemoryObjectStore())
    hold = {FeedName.LIVE_TEMPS: feed_hold(FeedName.LIVE_TEMPS)}
    await publish(store, sample_snapshot(holds=hold), run_id="run-1", now=NOW)

    with caplog.at_level(logging.INFO):
        result = await publish(
            store, sample_snapshot(holds=hold), run_id="run-2", now=LATER
        )

    # Nothing can be carried for it, so the second run has nothing new to say.
    assert result.outcome == "unchanged"
    assert await store.read_manifest(manifest_key(result.generation_id)) is None
    live = _freshness(caplog)[FeedName.LIVE_TEMPS]
    assert live.outcome == "absent"
    assert live.levelno == logging.WARNING
    assert not hasattr(live, "age_seconds")


@pytest.mark.asyncio
async def test_a_held_feed_whose_source_identity_differs_is_not_carried() -> None:
    store = SnapshotStore(MemoryObjectStore())
    await publish(store, sample_snapshot(), run_id="run-1", now=NOW)

    result = await publish(
        store,
        sample_snapshot(
            holds={
                FeedName.LIVE_TEMPS: feed_hold(
                    FeedName.LIVE_TEMPS, source_identity="coops:live_temps:9999999"
                )
            }
        ),
        run_id="run-2",
        now=LATER,
    )

    manifest = await store.read_manifest(manifest_key(result.generation_id))
    assert manifest is not None
    assert FeedName.LIVE_TEMPS not in manifest.locations["nyc"].feeds
    assert PlotName.LIVE_TEMPS not in manifest.locations["nyc"].plots


@pytest.mark.asyncio
async def test_carried_feed_within_its_interval_is_carried_not_stale(caplog) -> None:
    """A failed fetch soon after the last good one is carried, not yet stale."""
    objects = MemoryObjectStore()
    store = SnapshotStore(objects)
    await publish(store, sample_snapshot(), run_id="run-1", now=NOW)

    failing = sample_snapshot(
        failures={
            FeedName.LIVE_TEMPS: feed_failure(
                FeedName.LIVE_TEMPS, consecutive_failures=1, last_error="boom"
            )
        }
    )
    # Ten minutes after the frame was fetched: inside the live feed's
    # 600-second interval plus the fifteen-minute health buffer.
    soon = FETCHED_AT + datetime.timedelta(minutes=10)
    with caplog.at_level(logging.INFO):
        await publish(store, failing, run_id="run-2", now=soon)

    live = _freshness(caplog)[FeedName.LIVE_TEMPS]
    assert live.outcome == "carried"
    assert live.levelno == logging.WARNING
    assert live.age_seconds == 600
