"""The manager's single serving cycle and bounded plot wait.

`update_once` is the body of the background loop's tick, shared with the
publishing job; `wait_for_plots` lets a bounded host collect every submitted
plot before it reads them, which the loop never needs.
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
async def test_background_loop_tick_is_update_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop's first tick performs exactly the sequence update_once does."""
    manager = _manager()
    calls = _record_cycle(manager, monkeypatch)

    manager.start()
    try:
        assert await manager.wait_until_ready(timeout=5.0)
    finally:
        await manager.stop()

    assert calls[: len(EXPECTED_CYCLE)] == EXPECTED_CYCLE
    # Any later ticks repeat the same cycle; the loop adds nothing between them.
    assert len(calls) % len(EXPECTED_CYCLE) == 0


@pytest.mark.asyncio
async def test_update_once_propagates_a_feed_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A raising feed ends the cycle; the loop logs it and still becomes ready."""
    manager = _manager()
    calls = _record_cycle(manager, monkeypatch)
    failing = AsyncMock(side_effect=RuntimeError("currents boom"))
    monkeypatch.setattr(manager_module.updater, "update_dataset", failing)

    with pytest.raises(RuntimeError, match="currents boom"):
        await manager.update_once()
    assert calls == []

    with caplog.at_level(logging.ERROR):
        manager.start()
        try:
            assert await manager.wait_until_ready(timeout=5.0)
        finally:
            await manager.stop()
    assert any(
        "Error in data update loop: currents boom" in record.getMessage()
        for record in caplog.records
    )


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
async def test_wait_for_plots_abandons_a_hung_worker_like_the_loop(
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
