"""The local entry point: one process publishing into a local store, serving it.

These tests drive the two halves the entry point composes. The job half is
`update.publish_locations` against a locator rather than a bucket; the web half
is the app's own load, which must find exactly what the job half published, and
must have found it before the first request arrives. Mocked clients keep every
run offline, and the store variables are set through the entry point's own
helper so nothing here can reach the operator's bucket.
"""

import contextlib
import datetime
import json
import os
import threading
from collections.abc import AsyncGenerator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import aiohttp
import fastapi
import pandas as pd
import pytest
import pytz
from fastapi.testclient import TestClient

from shallweswim import config, local, update
from shallweswim import web as web_module
from shallweswim.archive import store as archive_store
from shallweswim.clients.base import BaseApiClient
from shallweswim.clients.coops import CoopsApi
from shallweswim.config import CoopsTempFeedConfig, CoopsTideFeedConfig, LocationConfig
from shallweswim.core import manager as manager_module
from shallweswim.types import TIDE_TYPE_CATEGORIES
from shallweswim.util import utc_now

# A short configured history keeps the hydration test to two provider years.
LAST_YEAR = utc_now().year - 1

HISTORY_CONFIG = LocationConfig(
    code="loc",
    name="Local Store Location",
    swim_location="Local Beach",
    swim_location_link="http://example.com/local",
    description="Test location with a two-year configured history",
    latitude=40.0,
    longitude=-74.0,
    timezone=pytz.timezone("US/Eastern"),
    default_temperature_unit="F",
    live_temp_source=CoopsTempFeedConfig(station=1234567, name="Local Temp"),
    historic_temp_source=CoopsTempFeedConfig(
        station=1234567, name="Local Temp", start_year=LAST_YEAR
    ),
    tide_source=CoopsTideFeedConfig(station=1234567, name="Local Tide"),
    enabled=True,
)


class MockCoopsApi(CoopsApi):
    """CO-OPS client returning frames inside the requested window.

    Historical years must land in the year they were requested for, because
    hydration reads the archive partition that year wrote.
    """

    def __init__(self) -> None:
        super().__init__(session=cast(aiohttp.ClientSession, None))
        self.live_temperature_calls = 0
        self.historic_temperature_calls = 0
        self.historic_years: list[int] = []

    async def temperature(
        self,
        station: int,
        product: str,
        begin_date: object,
        end_date: object,
        timezone: str,
        interval: str | None = None,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return a day of hourly readings ending at the requested window end."""
        end = pd.Timestamp(str(end_date)).tz_localize("UTC").floor("h")
        water_temp = 60.5
        if interval == "6-min":
            self.live_temperature_calls += 1
            # Live readings move between cycles, as a real station's do, so a
            # second cycle publishes a genuinely new generation.
            water_temp += self.live_temperature_calls
        else:
            self.historic_temperature_calls += 1
            self.historic_years.append(end.year)
        index = pd.date_range(end=end, periods=24, freq="h", name="time")
        return pd.DataFrame({"water_temp": [water_temp] * len(index)}, index=index)

    async def tides(
        self, station: int, timezone: str, location_code: str = "unknown"
    ) -> pd.DataFrame:
        """Return a minimal high/low event frame around now."""
        index = pd.date_range(
            pd.Timestamp(utc_now()).floor("h") - pd.Timedelta(hours=6),
            periods=4,
            freq="6h",
            tz="UTC",
            name="time",
        )
        return pd.DataFrame(
            {
                "prediction": [-0.5, 1.2, -0.3, 1.4],
                "type": pd.Categorical(
                    ["low", "high", "low", "high"], categories=TIDE_TYPE_CATEGORIES
                ),
            },
            index=index,
        )


@pytest.fixture(autouse=True)
def clear_store_cache() -> Iterator[None]:
    """Keep the process-wide memory store from leaking between tests."""
    archive_store.memory_store.cache_clear()
    yield
    archive_store.memory_store.cache_clear()


@pytest.fixture
def store_locator(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], str]:
    """Apply a locator through the entry point's helper, restored afterwards."""

    def use(locator: str) -> str:
        for name in local.STORE_ENV_VARS:
            # Registers the operator's value, whatever it is, for restoration.
            monkeypatch.setenv(name, "unset-by-test")
        local.apply_store_locator(locator)
        return locator

    return use


@pytest.fixture
def cycle_clients(monkeypatch: pytest.MonkeyPatch) -> MockCoopsApi:
    """Install one fake location and mocked clients, with plots stubbed out."""
    configs = MappingProxyType({HISTORY_CONFIG.code: HISTORY_CONFIG})
    monkeypatch.setattr(config, "CONFIGS", configs)
    # `config.get` reads the real mapping in its own module, so the routes need
    # their own patch to resolve the fake location.
    monkeypatch.setattr(config, "get", lambda code: configs.get(code.lower()))
    monkeypatch.setattr(
        manager_module, "_generate_live_temp_plot", lambda *args: b"<svg>live</svg>"
    )
    monkeypatch.setattr(
        manager_module,
        "_generate_historic_temp_plots",
        lambda *args: {"2mo": b"<svg>2mo</svg>", "12mo": b"<svg>12mo</svg>"},
    )
    return MockCoopsApi()


async def _run_cycle(
    client: MockCoopsApi, run_id: str, locator: str, *, full_history: bool = False
) -> str:
    """Run one publishing cycle in a thread pool instead of a process pool."""
    clients: dict[str, BaseApiClient] = {"coops": client}
    with ThreadPoolExecutor() as pool:
        _, outcome = await update.publish_locations(
            clients, run_id, pool=pool, locator=locator, full_history=full_history
        )
    return outcome


@pytest.mark.asyncio
async def test_memory_cycle_publishes_a_generation_the_app_loads(
    cycle_clients: MockCoopsApi, store_locator: Callable[[str], str]
) -> None:
    """The default locator is one store: the job half writes what the web half reads."""
    locator = store_locator(archive_store.MEMORY_LOCATOR)

    outcome = await _run_cycle(cycle_clients, "run-memory", locator)

    assert outcome == "success"
    served = SimpleNamespace(state=SimpleNamespace())
    await web_module.start_snapshot_serving(served)  # pyrefly: ignore
    assert set(served.state.snapshot.managers) == {HISTORY_CONFIG.code}
    assert served.state.snapshot.generation_id is not None


@pytest.mark.asyncio
async def test_filesystem_store_persists_and_hydrates_the_second_cycle(
    cycle_clients: MockCoopsApi,
    store_locator: Callable[[str], str],
    tmp_path: Path,
) -> None:
    """A store directory keeps the archive, so past years stop being refetched."""
    root = tmp_path / "store"
    locator = store_locator(str(root))

    assert await _run_cycle(cycle_clients, "run-one", locator) == "success"
    first_years = list(cycle_clients.historic_years)
    # A second cycle one moment later would hold every feed on the schedule the
    # first generation published, which says nothing about the archive, so this
    # one ignores the schedule as a backfill does.
    assert (
        await _run_cycle(cycle_clients, "run-two", locator, full_history=True)
        == "success"
    )

    assert first_years == [LAST_YEAR, utc_now().year]
    # Only the current year reaches the provider again; last year hydrates.
    assert cycle_clients.historic_years[len(first_years) :] == [utc_now().year]
    assert sorted(path.name for path in (root / "archive").rglob("*.parquet")) == [
        f"{LAST_YEAR}.parquet",
        f"{utc_now().year}.parquet",
    ]
    assert (root / "published" / "current.json").is_file()
    assert len(list((root / "published" / "manifests").glob("*.json"))) == 2

    served = SimpleNamespace(state=SimpleNamespace())
    await web_module.start_snapshot_serving(served)  # pyrefly: ignore
    assert set(served.state.snapshot.managers) == {HISTORY_CONFIG.code}


@pytest.mark.asyncio
async def test_a_second_cycle_sweeps_the_generation_it_supersedes(
    cycle_clients: MockCoopsApi,
    store_locator: Callable[[str], str],
    tmp_path: Path,
) -> None:
    """A store directory does not grow without bound: old manifests are swept."""
    root = tmp_path / "store"
    locator = store_locator(str(root))
    manifests = root / "published" / "manifests"

    assert await _run_cycle(cycle_clients, "run-one", locator) == "success"
    (first_manifest,) = list(manifests.glob("*.json"))
    # Age the first generation past the retention window without waiting a day.
    # Only its published_at decides retention, so the rest is left as published.
    published = json.loads(first_manifest.read_text())
    published["published_at"] = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=3)
    ).isoformat()
    first_manifest.write_text(json.dumps(published))

    # A backfill cycle ignores the published schedule, so it fetches and
    # publishes a second generation that supersedes the first.
    assert (
        await _run_cycle(cycle_clients, "run-two", locator, full_history=True)
        == "success"
    )

    remaining = list(manifests.glob("*.json"))
    assert first_manifest not in remaining
    assert len(remaining) == 1
    # The current pointer still names a generation the app can serve, and no
    # object was swept: they are all inside the one-hour safety window.
    served = SimpleNamespace(state=SimpleNamespace())
    await web_module.start_snapshot_serving(served)  # pyrefly: ignore
    assert served.state.snapshot.generation_id is not None
    assert set(served.state.snapshot.managers) == {HISTORY_CONFIG.code}


def _run_entry_point(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> dict[str, object]:
    """Run the entry point with the server and the updater stubbed out."""
    installed: dict[str, object] = {}
    monkeypatch.setattr(local, "setup_logging", lambda: "console")
    monkeypatch.setattr(
        local.web_module, "start_app", lambda **kwargs: fastapi.FastAPI()
    )
    monkeypatch.setattr(
        local,
        "install_updater",
        lambda app, **kwargs: installed.update(kwargs),
    )
    monkeypatch.setattr(local.uvicorn, "run", lambda app, **kwargs: None)

    assert local.main(argv) == 0
    return installed


def test_entry_point_overrides_bucket_variables_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A locator replaces whatever the operator's `.env` left in the shell."""
    for name in local.STORE_ENV_VARS:
        monkeypatch.setenv(name, "operator-archive-bucket")

    installed = _run_entry_point(
        monkeypatch, ["--store-dir", str(tmp_path), "--cadence", "2"]
    )

    expected = str(tmp_path.resolve())
    assert [os.environ[name] for name in local.STORE_ENV_VARS] == [expected] * 3
    assert installed == {"locator": expected, "cadence_seconds": 120.0}


def test_entry_point_defaults_to_the_memory_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no store directory nothing outlives the process, and no bucket is read."""
    for name in local.STORE_ENV_VARS:
        monkeypatch.setenv(name, "operator-archive-bucket")

    installed = _run_entry_point(monkeypatch, [])

    assert [os.environ[name] for name in local.STORE_ENV_VARS] == [
        archive_store.MEMORY_LOCATOR
    ] * 3
    assert installed["cadence_seconds"] == local.DEFAULT_CADENCE_MINUTES * 60


def test_lifespan_composition_runs_the_updater_between_startup_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app's own lifespan still brackets everything the updater does."""
    order: list[str] = []
    cycled = threading.Event()

    async def fake_publish(
        clients: dict[str, BaseApiClient],
        run_id: str,
        *,
        pool: object,
        locator: str,
    ) -> tuple[list[object], str]:
        order.append(f"cycle {locator}")
        cycled.set()
        return [], "success"

    monkeypatch.setattr(local.update, "publish_locations", fake_publish)
    monkeypatch.setattr(local, "create_api_clients", lambda session: {})

    loads: list[str] = []

    class FakeSnapshotState:
        """Stands in for the state the app's own lifespan builds."""

        async def initial_load(self) -> None:
            loads.append("load")

    @contextlib.asynccontextmanager
    async def app_lifespan(app: fastapi.FastAPI) -> AsyncGenerator[None]:
        order.append("app start")
        app.state.process_pool = None
        app.state.snapshot = FakeSnapshotState()
        yield
        order.append("app stop")

    app = fastapi.FastAPI(lifespan=app_lifespan)
    # An hour's cadence leaves the task asleep after its startup cycle, so the
    # shutdown path is what ends it.
    local.install_updater(app, locator="memory", cadence_seconds=3600)

    with TestClient(app):
        assert cycled.wait(timeout=10)
    task = app.state.local_updater

    assert order == ["app start", "cycle memory", "app stop"]
    # The first generation is loaded before the server accepts a request.
    assert loads == ["load"]
    assert task.cancelled()


def test_store_variables_are_the_three_the_stores_read() -> None:
    """The locator must cover capture, hydration, and snapshot reading alike."""
    assert local.STORE_ENV_VARS == (
        "SHALLWESWIM_ARCHIVE_BUCKET",
        "SHALLWESWIM_ARCHIVE_READ_BUCKET",
        "SHALLWESWIM_SNAPSHOT_READ_BUCKET",
    )


def test_first_request_is_served_from_the_first_published_generation(
    cycle_clients: MockCoopsApi,
    store_locator: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed lifespan publishes and loads before the server serves.

    This is the whole point of the composition: the process fetches once, and
    the first request already has a generation behind it rather than the 503
    an instance answers before the job has published anything.
    """
    locator = store_locator(archive_store.MEMORY_LOCATOR)
    monkeypatch.setattr(
        local, "create_api_clients", lambda _session: {"coops": cycle_clients}
    )
    # The cycle's plot stubs are patched in this process, so the plots must run
    # here rather than in a real process pool's workers.
    monkeypatch.setattr(web_module, "ProcessPoolExecutor", ThreadPoolExecutor)

    app = web_module.start_app()
    original_lifespan = app.router.lifespan_context
    local.install_updater(app, locator=locator, cadence_seconds=3600)
    try:
        with TestClient(app) as client:
            conditions = client.get(f"/api/{HISTORY_CONFIG.code}/conditions")
            status = client.get("/api/status")
    finally:
        app.router.lifespan_context = original_lifespan
        app.state.snapshot = None

    assert conditions.status_code == 200
    assert conditions.json()["temperature"]["water_temp_f"] is not None
    assert status.json()[HISTORY_CONFIG.code]["generation_id"] is not None
    # One process, one fetch of each feed.
    assert cycle_clients.live_temperature_calls == 1
