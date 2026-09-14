"""Generation collection on the memory store, at a controlled instant.

Every test drives `collect_generations` with an explicit `now` and ages objects
by writing the memory store's recorded creation times, so the retention and
safety windows are exercised without waiting for real time to pass. Each test
pins one of the contract's safety rules.
"""

import datetime
import logging

import pytest

from shallweswim.archive.store import MemoryObjectStore
from shallweswim.core.feeds import FeedName
from shallweswim.snapshot.gc import (
    OBJECT_SAFETY_AGE,
    RETAINED_GENERATION_AGE,
    collect_generations,
)
from shallweswim.snapshot.model import (
    SCHEMA_VERSION,
    CurrentPointer,
    FeedObject,
    LocationManifest,
    Manifest,
)
from shallweswim.snapshot.store import (
    CURRENT_KEY,
    OBJECTS_PREFIX,
    SnapshotStore,
    manifest_key,
)

NOW = datetime.datetime(2026, 9, 14, 12, 0, 0, tzinfo=datetime.UTC)

# Comfortably outside each window, and comfortably inside it.
LONG_AGO = NOW - RETAINED_GENERATION_AGE - datetime.timedelta(hours=16)
RECENTLY = NOW - datetime.timedelta(minutes=30)
SWEEPABLE = NOW - OBJECT_SAFETY_AGE - datetime.timedelta(minutes=30)


def _object_key(name: str) -> str:
    return f"{OBJECTS_PREFIX}/sha256-{name}.parquet"


def _feed_object(key: str) -> FeedObject:
    """A manifest entry naming one object; only its key matters here."""
    return FeedObject(
        key=key,
        size_bytes=1,
        source_identity="coops:live_temps:8518750",
        fetch_timestamp=NOW,
        next_fetch_after=None,
        expiration_seconds=None,
        record_count=1,
        consecutive_failures=0,
        last_error=None,
        timezone="US/Eastern",
        historical=None,
    )


def _manifest(
    generation_id: str, published_at: datetime.datetime, keys: list[str]
) -> Manifest:
    """One generation referencing `keys`, one per feed name."""
    return Manifest(
        schema_version=SCHEMA_VERSION,
        generation_id=generation_id,
        published_at=published_at,
        previous_generation_id=None,
        locations={
            "nyc": LocationManifest(
                feeds={
                    name: _feed_object(key)
                    for name, key in zip(FeedName, keys, strict=False)
                },
                plots={},
            )
        },
    )


async def _write_object(
    store: SnapshotStore,
    memory: MemoryObjectStore,
    key: str,
    *,
    created_at: datetime.datetime,
) -> None:
    """Write one object and age it to `created_at`."""
    await store.write_object(key, key.encode())
    memory._created[key] = created_at


async def _write_generation(
    store: SnapshotStore,
    generation_id: str,
    published_at: datetime.datetime,
    keys: list[str],
    *,
    promote: bool = False,
) -> str:
    """Write one generation's manifest, optionally promoting it to current."""
    key, _data = await store.write_manifest(
        _manifest(generation_id, published_at, keys)
    )
    if promote:
        current = await store.read_current()
        await store.promote(
            CurrentPointer(manifest_key=key, generation_id=generation_id),
            None if current is None else current[1],
        )
    return key


def _keys(memory: MemoryObjectStore, prefix: str) -> list[str]:
    return sorted(key for key in memory._objects if key.startswith(prefix))


def _gc_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "component", None) == "snapshot" and record.operation == "gc"
    ]


@pytest.mark.asyncio
async def test_retained_manifests_keep_every_object_they_reference() -> None:
    """Including an object the current generation reuses from an older one."""
    memory = MemoryObjectStore()
    store = SnapshotStore(memory)
    reused, dropped, fresh, recent = (
        _object_key("reused"),
        _object_key("dropped"),
        _object_key("fresh"),
        _object_key("recent"),
    )
    for key in (reused, dropped, fresh, recent):
        # Old enough that only reachability can save them.
        await _write_object(store, memory, key, created_at=SWEEPABLE)
    # The expired generation wrote both `reused` and `dropped`.
    expired = await _write_generation(store, "gen-expired", LONG_AGO, [reused, dropped])
    # A generation inside the retention window that is not current.
    retained = await _write_generation(
        store, "gen-retained", NOW - datetime.timedelta(hours=6), [recent]
    )
    current = await _write_generation(
        store, "gen-current", NOW, [reused, fresh], promote=True
    )

    result = await collect_generations(store, now=NOW)

    assert result.outcome == "success"
    assert result.manifests_examined == 3
    assert result.manifests_deleted == 1
    assert result.objects_deleted == 1
    assert _keys(memory, OBJECTS_PREFIX) == sorted([reused, fresh, recent])
    assert expired not in memory._objects
    assert {retained, current, CURRENT_KEY} <= set(memory._objects)


@pytest.mark.asyncio
async def test_unreferenced_object_survives_only_the_safety_window() -> None:
    """The same object is kept while young and deleted once it ages out."""
    memory = MemoryObjectStore()
    store = SnapshotStore(memory)
    orphan = _object_key("orphan")
    await _write_object(store, memory, orphan, created_at=RECENTLY)
    await _write_generation(store, "gen-current", NOW, [], promote=True)

    inside = await collect_generations(store, now=NOW)
    assert inside.objects_deleted == 0
    assert orphan in memory._objects

    # The same store one hour on: nothing else changed but the object's age.
    later = RECENTLY + OBJECT_SAFETY_AGE + datetime.timedelta(minutes=1)
    outside = await collect_generations(store, now=later)

    assert outside.objects_examined == 1
    assert outside.objects_deleted == 1
    assert orphan not in memory._objects


@pytest.mark.asyncio
async def test_a_publisher_that_has_not_promoted_yet_loses_nothing() -> None:
    """Its manifest is inside the retention window and its objects inside the safety one."""
    memory = MemoryObjectStore()
    store = SnapshotStore(memory)
    published = _object_key("current")
    in_flight = _object_key("in-flight")
    await _write_object(store, memory, published, created_at=SWEEPABLE)
    await _write_object(
        store, memory, in_flight, created_at=NOW - datetime.timedelta(seconds=30)
    )
    await _write_generation(store, "gen-current", NOW, [published], promote=True)
    # Written seconds ago by a publisher that has not promoted it yet.
    unpromoted = await _write_generation(
        store, "gen-in-flight", NOW - datetime.timedelta(seconds=30), [in_flight]
    )

    result = await collect_generations(store, now=NOW)

    assert result.manifests_deleted == 0
    assert result.objects_deleted == 0
    assert unpromoted in memory._objects
    assert _keys(memory, OBJECTS_PREFIX) == sorted([published, in_flight])


@pytest.mark.asyncio
async def test_a_manifest_older_than_the_window_that_is_still_current_is_kept() -> None:
    """A job that has not published for days still has a serving generation."""
    memory = MemoryObjectStore()
    store = SnapshotStore(memory)
    served = _object_key("served")
    await _write_object(store, memory, served, created_at=LONG_AGO)
    stale_current = await _write_generation(
        store, "gen-stale", LONG_AGO, [served], promote=True
    )

    result = await collect_generations(store, now=NOW)

    assert result.manifests_deleted == 0
    assert result.objects_deleted == 0
    assert stale_current in memory._objects
    assert CURRENT_KEY in memory._objects
    assert served in memory._objects


@pytest.mark.asyncio
async def test_a_sweep_without_a_current_pointer_retains_only_the_window() -> None:
    """No pointer means nothing extra is retained, and nothing extra is deleted."""
    memory = MemoryObjectStore()
    store = SnapshotStore(memory)
    orphan = _object_key("orphan")
    await _write_object(store, memory, orphan, created_at=SWEEPABLE)
    expired = await _write_generation(store, "gen-expired", LONG_AGO, [orphan])

    result = await collect_generations(store, now=NOW)

    assert result.outcome == "success"
    assert result.manifests_deleted == 1
    assert result.objects_deleted == 1
    assert expired not in memory._objects
    assert memory._objects == {}


@pytest.mark.asyncio
async def test_an_unparsable_manifest_is_retained_and_stops_object_deletion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Its references are unknown, so no object can be shown to be unreachable."""
    memory = MemoryObjectStore()
    store = SnapshotStore(memory)
    orphan = _object_key("orphan")
    await _write_object(store, memory, orphan, created_at=SWEEPABLE)
    await _write_generation(store, "gen-current", NOW, [], promote=True)
    broken = manifest_key("gen-broken")
    await memory.compare_and_swap(broken, expected_version=None, data=b"{not json")

    with caplog.at_level(logging.INFO):
        result = await collect_generations(store, now=NOW)

    assert result.outcome == "success"
    assert result.manifests_deleted == 0
    assert result.objects_examined == 1
    assert result.objects_deleted == 0
    assert broken in memory._objects
    assert orphan in memory._objects
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert broken in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_one_success_event_carries_the_bounded_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory = MemoryObjectStore()
    store = SnapshotStore(memory)
    orphan = _object_key("orphan")
    await _write_object(store, memory, orphan, created_at=SWEEPABLE)
    await _write_generation(store, "gen-expired", LONG_AGO, [orphan])

    with caplog.at_level(logging.INFO):
        result = await collect_generations(store, now=NOW)

    (event,) = _gc_events(caplog)
    assert event.levelno == logging.INFO
    assert event.outcome == "success"
    assert event.duration_ms == result.duration_ms
    assert event.record_count == result.objects_deleted == 1
    assert "1 of 1 manifests" in event.getMessage()
    assert "1 of 1 examined objects" in event.getMessage()


@pytest.mark.asyncio
async def test_a_store_failure_logs_one_failed_event_and_does_not_raise(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = MemoryObjectStore()
    store = SnapshotStore(memory)
    await _write_generation(store, "gen-current", NOW, [], promote=True)

    async def failing_list(prefix: str) -> list[object]:
        raise RuntimeError("bucket listing refused")

    monkeypatch.setattr(memory, "list", failing_list)

    with caplog.at_level(logging.INFO):
        result = await collect_generations(store, now=NOW)

    assert result.outcome == "failed"
    assert result.reason == "bucket listing refused"
    assert result.objects_deleted == 0
    (event,) = _gc_events(caplog)
    assert event.levelno == logging.ERROR
    assert event.outcome == "failed"
    assert event.record_count == 0
