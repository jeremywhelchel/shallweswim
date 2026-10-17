"""The manager's single serving cycle and bounded plot wait.

`update_once` is the publishing job's serving cycle; `wait_for_plots` lets the
job collect every submitted plot before it reads them.
"""

import asyncio
import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shallweswim.clients.coops import CoopsApi
from shallweswim.core import feeds
from shallweswim.core import manager as manager_module
from shallweswim.core.manager import LocationDataManager
from shallweswim.update import serve_location, serve_location_and_collect_plots
from shallweswim.util import utc_now
from tests.conftest import TEST_CONFIG_FULL

EXPECTED_CYCLE = [
    ("update", feeds.FEED_TIDES),
    ("update", feeds.FEED_CURRENTS),
    ("update", feeds.FEED_LIVE_TEMPS),
    ("update", feeds.FEED_HISTORIC_TEMPS),
    ("precompute_tides",),
    ("precompute_currents",),
    ("generate_plots",),
]


def _manager() -> LocationDataManager:
    return LocationDataManager(
        TEST_CONFIG_FULL,
        clients={"coops": MagicMock(spec=CoopsApi)},
        process_pool=MagicMock(spec=ProcessPoolExecutor),
    )


def _record_cycle(
    manager: LocationDataManager, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[Any, ...]]:
    """Replace the cycle's steps with recorders and return their call log."""
    calls: list[tuple[Any, ...]] = []

    async def update_dataset(
        feeds_dict: object, clients: object, feed_name: feeds.FeedName
    ) -> None:
        assert feeds_dict is manager._feeds
        assert clients is manager.clients
        calls.append(("update", feed_name))

    monkeypatch.setattr(manager_module.updater, "update_dataset", update_dataset)
    monkeypatch.setattr(
        manager,
        "_precompute_tide_predictions",
        lambda: calls.append(("precompute_tides",)),
    )
    monkeypatch.setattr(
        manager,
        "_precompute_current_predictions",
        lambda: calls.append(("precompute_currents",)),
    )

    def generate_plots(loop: asyncio.AbstractEventLoop) -> None:
        assert loop is asyncio.get_running_loop()
        calls.append(("generate_plots",))

    monkeypatch.setattr(manager, "_generate_plots", generate_plots)
    return calls


@pytest.mark.asyncio
async def test_update_once_runs_the_cycle_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    calls = _record_cycle(manager, monkeypatch)

    await manager.update_once()

    assert calls == EXPECTED_CYCLE


@pytest.mark.asyncio
async def test_update_once_propagates_a_feed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising feed ends the cycle before the later steps run."""
    manager = _manager()
    calls = _record_cycle(manager, monkeypatch)
    failing = AsyncMock(side_effect=RuntimeError("currents boom"))
    monkeypatch.setattr(manager_module.updater, "update_dataset", failing)

    with pytest.raises(RuntimeError, match="currents boom"):
        await manager.update_once()
    assert calls == []


@pytest.mark.asyncio
async def test_wait_for_plots_returns_once_pending_plots_complete() -> None:
    manager = _manager()
    loop = asyncio.get_running_loop()
    live: asyncio.Future[bytes] = loop.create_future()
    historic: asyncio.Future[dict[str, bytes]] = loop.create_future()
    submitted_at = utc_now()
    manager._pending_plot_futures[feeds.FEED_LIVE_TEMPS] = (live, submitted_at)
    manager._pending_plot_futures[feeds.FEED_HISTORIC_TEMPS] = (historic, submitted_at)
    loop.call_later(0.01, live.set_result, b"<svg>live</svg>")
    loop.call_later(
        0.05,
        historic.set_result,
        {"2mo": b"<svg>2mo</svg>", "12mo": b"<svg>12mo</svg>"},
    )

    await asyncio.wait_for(manager.wait_for_plots(timeout=5.0), timeout=2.0)

    assert manager._pending_plot_futures == {}
    assert manager.get_plot(feeds.PLOT_LIVE_TEMPS) == b"<svg>live</svg>"
    assert manager.get_plot(feeds.PLOT_HISTORIC_TEMPS_2MO) == b"<svg>2mo</svg>"
    assert manager.get_plot(feeds.PLOT_HISTORIC_TEMPS_12MO) == b"<svg>12mo</svg>"


@pytest.mark.asyncio
async def test_wait_for_plots_returns_immediately_with_nothing_pending() -> None:
    manager = _manager()

    await asyncio.wait_for(manager.wait_for_plots(timeout=60.0), timeout=1.0)

    assert manager._pending_plot_futures == {}


@pytest.mark.asyncio
async def test_wait_for_plots_warns_on_timeout_and_leaves_the_future(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _manager()
    loop = asyncio.get_running_loop()
    stuck: asyncio.Future[bytes] = loop.create_future()
    manager._pending_plot_futures[feeds.FEED_LIVE_TEMPS] = (stuck, utc_now())

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(manager.wait_for_plots(timeout=0.05), timeout=2.0)

    assert not stuck.cancelled()
    assert feeds.FEED_LIVE_TEMPS in manager._pending_plot_futures
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Plots for ['live_temps'] still pending after 0.05s" in (
        warnings[0].getMessage()
    )
    stuck.set_result(b"<svg>late</svg>")
    manager._collect_completed_plots()
    assert manager.get_plot(feeds.PLOT_LIVE_TEMPS) == b"<svg>late</svg>"


@pytest.mark.asyncio
async def test_wait_for_plots_abandons_a_hung_worker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A future past the hard timeout is dropped by the shared harvest."""
    manager = _manager()
    loop = asyncio.get_running_loop()
    hung: asyncio.Future[bytes] = loop.create_future()
    manager._pending_plot_futures[feeds.FEED_LIVE_TEMPS] = (
        hung,
        utc_now() - datetime.timedelta(seconds=manager_module.PLOT_HARD_TIMEOUT + 1),
    )

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(manager.wait_for_plots(timeout=5.0), timeout=2.0)

    assert manager._pending_plot_futures == {}
    assert any("abandoning" in record.getMessage() for record in caplog.records)
    hung.cancel()


@pytest.mark.asyncio
async def test_a_slow_location_does_not_delay_another_locations_plots() -> None:
    fast, slow = _manager(), _manager()
    slow_may_finish = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def fast_cycle() -> None:
        plot: asyncio.Future[bytes] = loop.create_future()
        plot.set_result(b"<svg>live</svg>")
        fast._pending_plot_futures[feeds.FEED_LIVE_TEMPS] = (plot, utc_now())

    fast.update_once = AsyncMock(side_effect=fast_cycle)  # type: ignore[method-assign]
    slow.update_once = AsyncMock(side_effect=slow_may_finish.wait)  # type: ignore[method-assign]

    run = asyncio.gather(
        serve_location_and_collect_plots(fast),
        serve_location_and_collect_plots(slow),
    )
    try:
        async with asyncio.timeout(2.0):
            while fast._pending_plot_futures or not fast.update_once.await_count:
                await asyncio.sleep(0.01)
        # The fast location's plot is collected while the slow one still runs.
        assert fast.get_plot(feeds.PLOT_LIVE_TEMPS) == b"<svg>live</svg>"
        assert not run.done()
    finally:
        slow_may_finish.set()
        await run


@pytest.mark.asyncio
async def test_serving_cycle_error_names_the_feed_whose_failure_it_restates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _manager()
    error = RuntimeError("tides boom")
    tides = manager._feeds[feeds.FEED_TIDES]
    assert tides is not None
    tides._last_error = error
    manager.update_once = AsyncMock(side_effect=[error, None])  # type: ignore[method-assign]

    with caplog.at_level(logging.ERROR):
        await serve_location(manager)

    [record] = caplog.records
    assert (record.component, record.location, record.feed, record.outcome) == (  # type: ignore[attr-defined]
        "updater",
        manager.config.code,
        feeds.FEED_TIDES.value,
        "failed",
    )
    assert not hasattr(record, "operation")


@pytest.mark.asyncio
async def test_serving_cycle_error_outside_a_feed_carries_no_feed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _manager()
    manager.update_once = AsyncMock(  # type: ignore[method-assign]
        side_effect=[RuntimeError("precompute boom"), None]
    )

    with caplog.at_level(logging.ERROR):
        await serve_location(manager)

    [record] = caplog.records
    # A defect in the cycle itself must stay an unattributed application error.
    assert not hasattr(record, "feed")
    assert not hasattr(record, "outcome")
