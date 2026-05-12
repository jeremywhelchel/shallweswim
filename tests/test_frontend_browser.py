"""Optional browser smoke tests for frontend behavior.

Run with:
    uv run pytest tests/test_frontend_browser.py -v --run-browser
"""

import socket
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import fastapi
import pytest
import requests
import uvicorn
from fastapi import responses
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Route, expect, sync_playwright

from shallweswim import canonical, config
from shallweswim.main import templates

pytestmark = pytest.mark.browser


@dataclass
class BrowserSmokeServer:
    """Local test server details for browser smoke tests."""

    base_url: str
    request_counts: dict[str, int]
    server: uvicorn.Server
    thread: threading.Thread


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _conditions_payload() -> dict[str, Any]:
    return {
        "location": {"code": "nyc", "name": "New York"},
        "temperature": {
            "water_temp": 53.1,
            "units": "F",
            "timestamp": "2026-05-12T09:06:00-04:00",
            "station_name": "The Battery, NY",
        },
        "tides": {
            "past": [
                {"type": "high", "time": "2026-05-12T04:09:00-04:00"},
            ],
            "next": [
                {"type": "low", "time": "2026-05-12T10:31:00-04:00"},
                {"type": "high", "time": "2026-05-12T16:48:00-04:00"},
            ],
        },
        "current": {
            "magnitude": 1.4,
            "direction": "EBB",
            "state_description": "strong ebb and easing",
        },
    }


@pytest.fixture
def browser_smoke_server() -> Generator[BrowserSmokeServer]:
    """Serve a small app with real templates/static assets and mocked API data."""
    app = fastapi.FastAPI()
    app.mount("/static", StaticFiles(directory="shallweswim/static"), name="static")
    request_counts = {"conditions": 0}

    @app.get("/nyc")
    async def location_page(request: fastapi.Request) -> responses.HTMLResponse:
        cfg = config.get("nyc")
        if not cfg:
            raise RuntimeError("NYC config missing")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "config": cfg,
                "all_locations": config.CONFIGS,
                "canonical_url": canonical.canonical_url("/nyc"),
            },
        )

    @app.get("/api/nyc/conditions")
    async def conditions() -> dict[str, Any]:
        request_counts["conditions"] += 1
        return _conditions_payload()

    @app.get("/api/nyc/plots/{_plot_name}")
    async def plot_placeholder(_plot_name: str) -> responses.Response:
        return responses.Response(status_code=204)

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
            response = requests.get(f"{base_url}/nyc", timeout=0.2)
            if response.status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Browser smoke test server did not start")

    yield BrowserSmokeServer(
        base_url=base_url,
        request_counts=request_counts,
        server=server,
        thread=thread,
    )

    server.should_exit = True
    thread.join(timeout=5)


def _block_external_requests(route: Route) -> None:
    request = route.request
    if request.url.startswith("http://127.0.0.1:"):
        route.continue_()
    else:
        route.abort()


def _launch_chromium() -> Any:
    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch()
    except PlaywrightError as exc:
        playwright.stop()
        if "Host system is missing dependencies" in str(exc):
            pytest.skip(f"Playwright host dependencies are missing: {exc}")
        raise
    return playwright, browser


def test_location_page_updates_conditions_once(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """The main page loads mocked API data and updates condition placeholders."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page()
        page.route("**/*", _block_external_requests)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        expect(page.locator("#temp-station-info")).to_contain_text("The Battery, NY")
        expect(page.locator("#past-tide-type")).to_have_text("high")
        expect(page.locator("#next-tide-0-type")).to_have_text("low")
        expect(page.locator("#next-tide-1-type")).to_have_text("high")
        expect(page.locator("#current-state-summary")).to_have_text(
            "strong ebb and easing"
        )
        expect(page.locator("#current-magnitude")).to_have_text("1.4")

        assert browser_smoke_server.request_counts["conditions"] == 1
    finally:
        browser.close()
        playwright.stop()


def test_windy_embed_uses_expected_parameters_and_layout(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """The Windy iframe keeps our expected URL contract and responsive layout."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.route("**/*", _block_external_requests)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        windy_frame = page.locator("iframe.windyframe")
        expect(windy_frame).to_have_count(1)
        expect(windy_frame).to_have_attribute("title", "Windy forecast")

        src = windy_frame.get_attribute("src")
        assert src is not None
        parsed_src = urlparse(src)
        query = parse_qs(parsed_src.query, keep_blank_values=True)

        assert parsed_src.scheme == "https"
        assert parsed_src.netloc == "embed.windy.com"
        assert parsed_src.path == "/embed2.html"
        assert query["overlay"] == ["waves"]
        assert query["product"] == ["ecmwfWaves"]
        assert query["detail"] == ["true"]
        assert query["width"] == ["950"]
        assert query["height"] == ["350"]
        assert query["metricTemp"] == ["°F"]

        box = windy_frame.bounding_box()
        assert box is not None
        assert box["width"] <= 950
        assert box["width"] > 900
        assert box["height"] == 350
    finally:
        browser.close()
        playwright.stop()
