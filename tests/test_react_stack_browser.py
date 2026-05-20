"""Optional React/FastAPI browser integration tests.

Run with:
    corepack pnpm@10.18.3 --dir frontend build
    uv run pytest tests/test_react_stack_browser.py -v --run-browser
"""

import datetime
import os
import re
import socket
import threading
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import requests
import uvicorn
from playwright.sync_api import (
    Browser,
    Page,
    Playwright,
    Route,
    expect,
    sync_playwright,
)
from playwright.sync_api import Error as PlaywrightError

from shallweswim import config, types
from shallweswim.api import routes
from shallweswim.core.feeds import (
    FEED_CURRENTS,
    FEED_LIVE_TEMPS,
    FEED_TIDES,
    PLOT_HISTORIC_TEMPS_2MO,
    PLOT_HISTORIC_TEMPS_12MO,
    PLOT_LIVE_TEMPS,
)
from shallweswim.main import app

pytestmark = pytest.mark.browser


@dataclass
class ReactStackServer:
    """Local server details for the React/FastAPI stack test."""

    base_url: str
    manager: "FakeNycDataManager"
    server: uvicorn.Server
    thread: threading.Thread


@dataclass
class FakeNycDataManager:
    """Small fake at the same boundary used by API route handlers."""

    expected_at: datetime.datetime
    has_data: bool = True
    tide_info_calls: list[datetime.datetime] = field(default_factory=list)
    tide_state_calls: list[datetime.datetime] = field(default_factory=list)
    current_calls: list[datetime.datetime] = field(default_factory=list)
    feed_value_calls: list[str] = field(default_factory=list)

    def has_feed_data(self, feed_name: str) -> bool:
        return feed_name in {FEED_LIVE_TEMPS, FEED_TIDES, FEED_CURRENTS}

    def get_current_temperature(self) -> types.TemperatureReading:
        return types.TemperatureReading(
            timestamp=self.expected_at - datetime.timedelta(minutes=20),
            temperature=61.4,
        )

    def get_tide_info_at_time(self, timestamp: datetime.datetime) -> types.TideInfo:
        self.tide_info_calls.append(timestamp)
        return types.TideInfo(
            past=[
                types.TideEntry(
                    time=timestamp - datetime.timedelta(hours=2, minutes=30),
                    type=types.TideCategory.LOW,
                    prediction=0.2,
                )
            ],
            next=[
                types.TideEntry(
                    time=timestamp + datetime.timedelta(hours=3, minutes=15),
                    type=types.TideCategory.HIGH,
                    prediction=4.8,
                )
            ],
        )

    def predict_tide_at_time(self, timestamp: datetime.datetime) -> types.TideState:
        self.tide_state_calls.append(timestamp)
        return types.TideState(
            timestamp=timestamp,
            estimated_height=2.2,
            trend=types.TideTrend.RISING,
            units="ft",
            height_pct=0.52,
        )

    def predict_flow_at_time(self, timestamp: datetime.datetime) -> types.CurrentInfo:
        self.current_calls.append(timestamp)
        return types.CurrentInfo(
            timestamp=timestamp,
            source_type=types.DataSourceType.PREDICTION,
            magnitude=1.4,
            direction=types.CurrentDirection.EBBING,
            phase=types.CurrentPhase.EBB,
            strength=types.CurrentStrength.STRONG,
            trend=types.CurrentTrend.BUILDING,
            magnitude_pct=0.75,
            state_description="strong ebb and building",
            range=types.CurrentRange(
                slack=types.CurrentRangePoint(
                    timestamp=timestamp - datetime.timedelta(hours=1),
                    magnitude=0.0,
                    phase=types.CurrentPhase.SLACK,
                ),
                peak=types.CurrentRangePoint(
                    timestamp=timestamp + datetime.timedelta(hours=2),
                    magnitude=1.8,
                    phase=types.CurrentPhase.EBB,
                ),
            ),
        )

    def get_feed_values(self, feed_name: str) -> list[int]:
        self.feed_value_calls.append(feed_name)
        return [1, 2, 3]

    def get_plot(self, plot_name: str) -> bytes | None:
        if plot_name in {
            PLOT_LIVE_TEMPS,
            PLOT_HISTORIC_TEMPS_2MO,
            PLOT_HISTORIC_TEMPS_12MO,
        }:
            return b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        return None


class FakeFigure:
    """Minimal Matplotlib-like figure used by the plot endpoint."""

    def __init__(self, timestamp: datetime.datetime) -> None:
        self.timestamp = timestamp

    def savefig(self, output: Any, **_kwargs: Any) -> None:
        output.write(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            f"<text>{self.timestamp.isoformat()}</text>"
            "</svg>"
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _target_local_time() -> datetime.datetime:
    cfg = config.get("nyc")
    if cfg is None:
        raise RuntimeError("NYC config missing")
    target_utc = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=2)
    target_local = target_utc.astimezone(cfg.timezone)
    return target_local.replace(tzinfo=None, second=0, microsecond=0)


def _skip_locally_fail_in_ci(message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS"):
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture
def react_stack_server(monkeypatch: pytest.MonkeyPatch) -> Generator[ReactStackServer]:
    """Serve the built React app through the real FastAPI app with fake data."""
    frontend_dist = Path("frontend/dist")
    if not (frontend_dist / "index.html").is_file():
        _skip_locally_fail_in_ci(
            "React app is not built. Run "
            "`corepack pnpm@10.18.3 --dir frontend build` first."
        )

    target_at = _target_local_time()
    manager = FakeNycDataManager(expected_at=target_at)
    executor = ThreadPoolExecutor(max_workers=1)
    previous_state = dict(app.state._state)

    def fake_create_tide_current_plot(
        _tides_data: Any,
        _currents_data: Any,
        timestamp: datetime.datetime,
        _cfg: Any,
    ) -> FakeFigure:
        return FakeFigure(timestamp)

    monkeypatch.setattr(
        routes, "_create_tide_current_plot", fake_create_tide_current_plot
    )
    app.state.frontend_dist = str(frontend_dist)
    app.state.data_managers = {"nyc": manager}
    app.state.process_pool = executor

    port = _free_port()
    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="off",
    )
    server = uvicorn.Server(server_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/app/nyc", timeout=0.2)
            if response.status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        executor.shutdown(wait=True)
        app.state._state.clear()
        app.state._state.update(previous_state)
        raise RuntimeError("React stack test server did not start")

    yield ReactStackServer(
        base_url=base_url,
        manager=manager,
        server=server,
        thread=thread,
    )

    server.should_exit = True
    thread.join(timeout=5)
    executor.shutdown(wait=True)
    app.state._state.clear()
    app.state._state.update(previous_state)


def _launch_chromium() -> tuple[Playwright, Browser]:
    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch()
    except PlaywrightError as exc:
        playwright.stop()
        if "Executable doesn't exist" in str(
            exc
        ) or "Host system is missing dependencies" in str(exc):
            _skip_locally_fail_in_ci(f"Playwright host dependencies are missing: {exc}")
        raise
    return playwright, browser


def _block_external_requests(route: Route) -> None:
    request = route.request
    if request.url.startswith("http://127.0.0.1:"):
        route.continue_()
    else:
        route.abort()


def test_react_planner_uses_conditions_and_plot_with_same_at(
    react_stack_server: ReactStackServer,
) -> None:
    """React URL state, rendered data, and real API handlers agree on ``at``."""
    playwright, browser = _launch_chromium()
    target_at = react_stack_server.manager.expected_at.isoformat()
    api_urls: list[str] = []

    try:
        page: Page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.route("**/*", _block_external_requests)
        page.on(
            "request",
            lambda request: (
                api_urls.append(request.url)
                if urlparse(request.url).path.startswith("/api/")
                else None
            ),
        )

        page.goto(
            f"{react_stack_server.base_url}/app/nyc"
            f"?planner=open&detail=open&at={target_at}"
        )

        expect(page.get_by_role("heading", name="Water Movement")).to_be_visible()
        expect(page.get_by_role("region", name="Planner mode")).to_be_visible()
        expect(page.get_by_text("2.2 ft")).to_be_visible()
        expect(page.get_by_text("1.4 kt")).to_be_visible()
        expect(page.get_by_text("1.6 ft")).to_have_count(0)
        expect(page.get_by_text(re.compile(r"water is going out fast"))).to_be_visible()

        detail_chart = page.get_by_role(
            "img", name=re.compile(r"^Tide and current plot")
        )
        expect(detail_chart).to_be_visible()
        expect(detail_chart).to_have_attribute(
            "src",
            f"/api/nyc/plots/current_tide?at={target_at.replace(':', '%3A')}",
        )

        condition_urls = [
            url for url in api_urls if urlparse(url).path == "/api/nyc/conditions"
        ]
        plot_urls = [
            url
            for url in api_urls
            if urlparse(url).path == "/api/nyc/plots/current_tide"
        ]
        current_urls = [
            url for url in api_urls if urlparse(url).path == "/api/nyc/currents"
        ]

        assert len(condition_urls) >= 1
        assert len(plot_urls) >= 1
        assert current_urls == []
        assert all(
            parse_qs(urlparse(url).query).get("at") == [target_at]
            for url in condition_urls
        )
        assert all(
            parse_qs(urlparse(url).query).get("at") == [target_at] for url in plot_urls
        )

        manager = react_stack_server.manager
        assert manager.tide_info_calls == [manager.expected_at]
        assert manager.tide_state_calls == [manager.expected_at]
        assert manager.current_calls == [manager.expected_at]
        assert manager.feed_value_calls.count(FEED_TIDES) >= 1
        assert manager.feed_value_calls.count(FEED_CURRENTS) >= 1
    finally:
        browser.close()
        playwright.stop()
