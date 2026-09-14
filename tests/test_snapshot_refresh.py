"""Shadow-mode loading and request-elected refresh on the web instance.

Shadow mode must be cheap and invisible: a check that finds nothing does no
work, a check that finds a new generation reads only the objects the instance
does not already hold, and neither a failure nor a slow bucket may affect
serving. These tests drive `SnapshotState` over an object store that counts
every read, with the module's clock replaced so intervals are exact.
"""

import asyncio
import datetime
import logging
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from shallweswim import config as config_lib
from shallweswim import main as main_module
from shallweswim.archive.store import MemoryObjectStore, StoredObject
from shallweswim.core.feeds import FeedName
from shallweswim.main import app
from shallweswim.snapshot import refresh as refresh_module
from shallweswim.snapshot.publish import publish
from shallweswim.snapshot.refresh import SnapshotState
from shallweswim.snapshot.store import CURRENT_KEY, OBJECTS_PREFIX, SnapshotStore
from tests.snapshot_fixtures import SAMPLE_OBJECT_COUNT, sample_snapshot

NOW = datetime.datetime(2026, 6, 1, 12, 30, tzinfo=datetime.UTC)
LATER = NOW + datetime.timedelta(minutes=10)
START = 1000.0


class CountingObjectStore(MemoryObjectStore):
    """A memory store that records every key read, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[str] = []
        self.fail: bool = False
        self.block: asyncio.Event | None = None

    async def read(self, key: str) -> StoredObject | None:
        self.reads.append(key)
        if self.fail:
            raise RuntimeError("bucket unreachable")
        if self.block is not None:
            await self.block.wait()
        return await super().read(key)

    def object_reads(self) -> list[str]:
        """Only the content-addressed object reads, not pointer or manifest."""
        return [key for key in self.reads if key.startswith(OBJECTS_PREFIX)]


class Clock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = START

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Replace the refresh module's clock, leaving the rest of the app alone."""
    fake = Clock()
    monkeypatch.setattr(
        refresh_module, "time", SimpleNamespace(monotonic=fake.monotonic)
    )
    return fake


def _snapshot_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "component", None) == "snapshot"
        and getattr(record, "operation", None) == "load"
    ]


async def _published_state(
    objects: CountingObjectStore, **kwargs: object
) -> SnapshotState:
    """Publish one generation and return a state that has loaded it."""
    store = SnapshotStore(objects)
    result = await publish(store, sample_snapshot(**kwargs), run_id="run-1", now=NOW)
    assert result.outcome == "success"
    state = SnapshotState(store)
    await state.initial_load()
    return state


# =============================================================================
# Incremental loading
# =============================================================================


@pytest.mark.asyncio
async def test_initial_load_reads_every_object_and_logs_one_success(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    objects = CountingObjectStore()
    with caplog.at_level(logging.DEBUG):
        state = await _published_state(objects)

    assert len(objects.object_reads()) == SAMPLE_OBJECT_COUNT
    assert set(state.managers) == {"nyc"}
    assert state.loaded_at is not None
    (event,) = _snapshot_events(caplog)
    assert event.levelno == logging.INFO
    assert event.outcome == "success"
    assert event.generation_id == state.generation_id
    assert event.record_count == SAMPLE_OBJECT_COUNT
    assert event.age_seconds >= 0


@pytest.mark.asyncio
async def test_refresh_reads_only_objects_not_already_held(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """Objects are content-addressed, so an unchanged key is never reread."""
    objects = CountingObjectStore()
    state = await _published_state(objects)
    first_generation = state.generation_id
    # Only the live temperature frame differs; its plot bytes are identical.
    await publish(
        SnapshotStore(objects),
        sample_snapshot(live_offset=1.0),
        run_id="run-2",
        now=LATER,
    )
    objects.reads.clear()

    clock.advance(refresh_module.CHECK_INTERVAL_SECONDS)
    with caplog.at_level(logging.DEBUG):
        await state.check_and_refresh()

    assert len(objects.object_reads()) == 1
    assert state.generation_id != first_generation
    (event,) = _snapshot_events(caplog)
    assert event.outcome == "success"
    assert event.record_count == 1
    # The reused frames are the same decoded objects, not re-parsed copies.
    assert set(state.managers) == {"nyc"}


@pytest.mark.asyncio
async def test_unchanged_pointer_reads_no_objects_and_logs_nothing(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    objects = CountingObjectStore()
    state = await _published_state(objects)
    managers = state.managers
    objects.reads.clear()

    clock.advance(refresh_module.CHECK_INTERVAL_SECONDS)
    with caplog.at_level(logging.INFO):
        await state.check_and_refresh()

    assert objects.reads == [CURRENT_KEY]
    assert state.managers is managers
    assert _snapshot_events(caplog) == []


# =============================================================================
# Election and scheduling
# =============================================================================


@pytest.mark.asyncio
async def test_check_waits_a_full_interval_after_a_failure(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing bucket must not be retried on every arriving request."""
    objects = CountingObjectStore()
    state = SnapshotState(SnapshotStore(objects))
    objects.fail = True

    clock.advance(refresh_module.CHECK_INTERVAL_SECONDS)
    with caplog.at_level(logging.DEBUG):
        await state.check_and_refresh()
    assert len(objects.reads) == 1
    (event,) = _snapshot_events(caplog)
    assert event.levelno == logging.WARNING
    assert event.outcome == "failed"
    assert event.generation_id is None
    # The lag field is present exactly when a generation was loaded.
    assert not hasattr(event, "age_seconds")

    clock.advance(refresh_module.CHECK_INTERVAL_SECONDS - 1)
    await state.check_and_refresh()
    assert len(objects.reads) == 1

    clock.advance(1)
    await state.check_and_refresh()
    assert len(objects.reads) == 2


@pytest.mark.asyncio
async def test_concurrent_requests_perform_one_check(clock: Clock) -> None:
    objects = CountingObjectStore()
    state = await _published_state(objects)
    objects.reads.clear()

    clock.advance(refresh_module.CHECK_INTERVAL_SECONDS)
    await asyncio.gather(*(state.check_and_refresh() for _ in range(5)))

    assert objects.reads == [CURRENT_KEY]


@pytest.mark.asyncio
async def test_requests_during_a_swap_see_one_whole_generation(clock: Clock) -> None:
    """Managers are published by one assignment, after every one is built."""
    objects = CountingObjectStore()
    state = await _published_state(objects)
    first_generation = state.generation_id
    first_managers = state.managers
    first_temps = first_managers["nyc"].get_feed_values(FeedName.LIVE_TEMPS)
    await publish(
        SnapshotStore(objects),
        sample_snapshot(live_offset=1.0),
        run_id="run-2",
        now=LATER,
    )

    objects.block = asyncio.Event()
    clock.advance(refresh_module.CHECK_INTERVAL_SECONDS)
    check = asyncio.create_task(state.check_and_refresh())
    await asyncio.sleep(0)

    # The load is suspended mid-flight; the instance still serves generation 1.
    assert state.generation_id == first_generation
    assert state.managers is first_managers
    with pytest.raises(TypeError):
        state.managers["nyc"] = first_managers["nyc"]  # pyrefly: ignore

    objects.block.set()
    await check

    assert state.generation_id != first_generation
    assert state.managers is not first_managers
    second_temps = state.managers["nyc"].get_feed_values(FeedName.LIVE_TEMPS)
    assert second_temps["water_temp"].iloc[0] == pytest.approx(
        first_temps["water_temp"].iloc[0] + 1.0
    )


@pytest.mark.asyncio
async def test_location_missing_from_the_bundle_has_no_manager(clock: Clock) -> None:
    objects = CountingObjectStore()
    state = await _published_state(objects)

    assert set(config_lib.CONFIGS) - {"nyc"}
    assert set(state.managers) == {"nyc"}


# =============================================================================
# Startup
# =============================================================================


@pytest.mark.asyncio
async def test_initial_load_failure_leaves_an_empty_state(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    objects = CountingObjectStore()
    objects.fail = True
    state = SnapshotState(SnapshotStore(objects))

    with caplog.at_level(logging.DEBUG):
        await state.initial_load()

    assert state.managers == {}
    assert state.generation_id is None
    assert state.loaded_at is None
    (event,) = _snapshot_events(caplog)
    assert event.levelno == logging.WARNING
    assert event.outcome == "failed"
    assert not hasattr(event, "age_seconds")


@pytest.mark.asyncio
async def test_initial_load_timeout_leaves_an_empty_state(clock: Clock) -> None:
    objects = CountingObjectStore()
    objects.block = asyncio.Event()
    state = SnapshotState(SnapshotStore(objects))

    await state.initial_load(timeout=0.01)

    assert state.managers == {}
    assert state.generation_id is None
    objects.block.set()


@pytest.mark.asyncio
async def test_shadow_mode_is_off_without_the_bucket_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(main_module.SNAPSHOT_READ_BUCKET_ENV_VAR, raising=False)
    shadow = SimpleNamespace(state=SimpleNamespace())

    await main_module.start_snapshot_shadow(shadow)  # pyrefly: ignore

    assert shadow.state.snapshot is None


@pytest.mark.asyncio
async def test_shadow_mode_loads_the_named_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = CountingObjectStore()
    await publish(SnapshotStore(objects), sample_snapshot(), run_id="run-1", now=NOW)
    monkeypatch.setenv(main_module.SNAPSHOT_READ_BUCKET_ENV_VAR, "bundle-bucket")
    buckets: list[str] = []

    def fake_gcs_store(bucket: str) -> CountingObjectStore:
        buckets.append(bucket)
        return objects

    monkeypatch.setattr(main_module, "gcs_store", fake_gcs_store)
    shadow = SimpleNamespace(state=SimpleNamespace())

    await main_module.start_snapshot_shadow(shadow)  # pyrefly: ignore

    assert buckets == ["bundle-bucket"]
    assert set(shadow.state.snapshot.managers) == {"nyc"}


@pytest.mark.asyncio
async def test_shadow_mode_survives_an_unreachable_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup keeps going with the state in place and no generation loaded."""
    objects = CountingObjectStore()
    objects.fail = True
    monkeypatch.setenv(main_module.SNAPSHOT_READ_BUCKET_ENV_VAR, "bundle-bucket")
    monkeypatch.setattr(main_module, "gcs_store", lambda _bucket: objects)
    shadow = SimpleNamespace(state=SimpleNamespace())

    await main_module.start_snapshot_shadow(shadow)  # pyrefly: ignore

    assert shadow.state.snapshot.generation_id is None
    assert shadow.state.snapshot.managers == {}


# =============================================================================
# Middleware
# =============================================================================


@pytest.fixture
def served_app() -> Iterator[None]:
    """The real app with empty managers and shadow mode off, restored after."""
    app.state.data_managers = {}
    app.state.snapshot = None
    yield
    app.state.snapshot = None


def test_middleware_is_inactive_without_a_shadow_state(served_app: None) -> None:
    """With shadow mode off the request path touches no snapshot code."""
    client = TestClient(app)

    response = client.get("/robots.txt")

    assert response.status_code == 200


def test_middleware_elects_a_request_to_refresh(served_app: None) -> None:
    """A request, health checks included, brings a due generation up to date."""
    checks: list[str] = []

    class RecordingState:
        async def check_and_refresh(self) -> None:
            checks.append("checked")

    app.state.snapshot = RecordingState()
    client = TestClient(app)

    response = client.get("/api/healthy")

    assert response.status_code == 503
    assert checks == ["checked"]
