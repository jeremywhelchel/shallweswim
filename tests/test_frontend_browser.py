"""Optional browser smoke tests for frontend behavior.

Run with:
    uv run pytest tests/test_frontend_browser.py -v --run-browser
"""

import re
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
from playwright.sync_api import (
    Browser,
    Page,
    Playwright,
    Route,
    expect,
    sync_playwright,
)
from playwright.sync_api import Error as PlaywrightError

from shallweswim import canonical, config
from shallweswim.main import templates

pytestmark = pytest.mark.browser


@dataclass
class BrowserSmokeServer:
    """Local test server details for browser smoke tests."""

    base_url: str
    request_counts: dict[str, int]
    request_order: list[str]
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


def _currents_payload() -> dict[str, Any]:
    return {
        "location": {"code": "nyc", "name": "New York"},
        "timestamp": "2026-05-12T11:15:00-04:00",
        "current": {
            "timestamp": "2026-05-12T11:15:00-04:00",
            "direction": "EBB",
            "phase": "ebb",
            "strength": "moderate",
            "trend": "building",
            "magnitude": 1.2,
            "magnitude_pct": 0.5,
            "state_description": "moderate ebb and building",
            "source_type": "prediction",
        },
        "legacy_chart": {
            "hours_since_last_tide": 1.1,
            "last_tide_type": "high",
            "chart_filename": "legacy-current.svg",
            "map_title": "NY Harbor",
        },
        "current_chart_filename": "/static/current-map.svg",
        "navigation": {
            "shift": 0,
            "next_hour": 60,
            "prev_hour": -60,
            "current_api_url": "/api/nyc/currents",
            "plot_url": "/api/nyc/plots/current_tide?shift=0",
        },
    }


def _transit_payload(status: str = "Good Service") -> dict[str, Any]:
    return {
        "status": status,
        "direction_statuses": {"south": status},
        "destinations": {"south": ["Coney Island-Stillwell Av"]},
        "delay_summaries": {"south": []},
        "service_change_summaries": {"both": "", "south": ""},
        "service_irregularity_summaries": {"south": []},
    }


@pytest.fixture
def browser_smoke_server() -> Generator[BrowserSmokeServer]:
    """Serve a small app with real templates/static assets and mocked API data."""
    app = fastapi.FastAPI()
    app.mount("/static", StaticFiles(directory="shallweswim/static"), name="static")
    request_counts = {"conditions": 0, "currents": 0, "plots": 0}
    request_order: list[str] = []

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

    @app.get("/nyc/currents")
    async def currents_page(request: fastapi.Request) -> responses.HTMLResponse:
        cfg = config.get("nyc")
        if not cfg:
            raise RuntimeError("NYC config missing")
        return templates.TemplateResponse(
            request=request,
            name="current.html",
            context={
                "config": cfg,
                "all_locations": config.CONFIGS,
                "canonical_url": canonical.canonical_url("/nyc/currents"),
            },
        )

    @app.get("/api/nyc/conditions")
    async def conditions() -> dict[str, Any]:
        request_counts["conditions"] += 1
        request_order.append("conditions")
        return _conditions_payload()

    @app.get("/api/nyc/currents")
    async def currents() -> dict[str, Any]:
        request_counts["currents"] += 1
        request_order.append("currents")
        return _currents_payload()

    @app.get("/api/nyc/plots/{_plot_name}")
    async def plot_placeholder(_plot_name: str) -> responses.Response:
        request_counts["plots"] += 1
        request_order.append("plot")
        return responses.Response(
            content=(
                '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8">'
                '<rect width="8" height="8" fill="blue"/></svg>'
            ),
            media_type="image/svg+xml",
        )

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
        request_order=request_order,
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


def _launch_chromium() -> tuple[Playwright, Browser]:
    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch()
    except PlaywrightError as exc:
        playwright.stop()
        if "Host system is missing dependencies" in str(exc):
            pytest.skip(f"Playwright host dependencies are missing: {exc}")
        raise
    return playwright, browser


def _block_external_and_fail_conditions(route: Route) -> None:
    request = route.request
    if request.url.startswith("http://127.0.0.1:") and request.url.endswith(
        "/api/nyc/conditions"
    ):
        route.fulfill(status=503, body="Service unavailable")
    elif request.url.startswith("http://127.0.0.1:"):
        route.continue_()
    else:
        route.abort()


def _block_external_and_fail_currents(route: Route) -> None:
    request = route.request
    if request.url.startswith("http://127.0.0.1:") and urlparse(
        request.url
    ).path.endswith("/api/nyc/currents"):
        route.fulfill(status=503, body="Service unavailable")
    elif request.url.startswith("http://127.0.0.1:"):
        route.continue_()
    else:
        route.abort()


def _mock_transit_success(route: Route) -> None:
    request = route.request
    if request.url.startswith("https://goodservice.io/api/routes/"):
        route.fulfill(json=_transit_payload())
    elif request.url.startswith("http://127.0.0.1:"):
        route.continue_()
    else:
        route.abort()


def _mock_transit_failure(route: Route) -> None:
    request = route.request
    if request.url.startswith("https://goodservice.io/api/routes/"):
        route.fulfill(status=503, body="Transit unavailable")
    elif request.url.startswith("http://127.0.0.1:"):
        route.continue_()
    else:
        route.abort()


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
        expect(page.locator("#conditions-status")).to_be_hidden()

        assert browser_smoke_server.request_counts["conditions"] == 1
    finally:
        browser.close()
        playwright.stop()


def test_transit_status_updates_when_goodservice_responds(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """NYC transit cards render train status independently from conditions."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page()
        page.route("**/*", _mock_transit_success)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        expect(page.locator("#Q_status")).to_have_text("Good Service")
        expect(page.locator("#Q_destination")).to_have_text("Coney Island-Stillwell Av")
        expect(page.locator("#B_status")).to_have_text("Good Service")
        expect(page.locator("#B_destination")).to_have_text("Coney Island-Stillwell Av")
        expect(page.locator("#Q_delay_div")).to_be_hidden()
        expect(page.locator("#B_delay_div")).to_be_hidden()
    finally:
        browser.close()
        playwright.stop()


def test_transit_failure_shows_unavailable_state(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """A failed transit lookup does not leave train cards stuck on placeholders."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page()
        page.route("**/*", _mock_transit_failure)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        expect(page.locator("#Q_status")).to_have_text("Unavailable")
        expect(page.locator("#Q_destination")).to_have_text("unavailable")
        expect(page.locator("#B_status")).to_have_text("Unavailable")
        expect(page.locator("#B_destination")).to_have_text("unavailable")
        expect(page.locator("#Q_delay_div")).to_be_hidden()
        expect(page.locator("#B_delay_div")).to_be_hidden()
    finally:
        browser.close()
        playwright.stop()


def test_deferred_plots_load_after_conditions(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """Temperature plots wait until the first conditions request has completed."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page()
        page.route("**/*", _block_external_requests)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        live_plot = page.locator('img[data-src="/api/nyc/plots/live_temps"]')
        expect(live_plot).to_have_attribute(
            "src", re.compile(r"/api/nyc/plots/live_temps$")
        )

        assert browser_smoke_server.request_order[0] == "conditions"
        assert browser_smoke_server.request_counts["conditions"] == 1
        assert browser_smoke_server.request_counts["plots"] >= 1
    finally:
        browser.close()
        playwright.stop()


def test_deferred_plot_loading_retries_transient_failures(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """A cold-start plot 503 is retried instead of sticking as a broken image."""
    playwright, browser = _launch_chromium()
    live_plot_attempts = 0

    def fail_first_live_plot_request(route: Route) -> None:
        nonlocal live_plot_attempts
        request = route.request
        if request.url.startswith("http://127.0.0.1:") and request.url.endswith(
            "/api/nyc/plots/live_temps"
        ):
            live_plot_attempts += 1
            if live_plot_attempts == 1:
                route.fulfill(status=503, body="Plot warming up")
            else:
                route.continue_()
        elif request.url.startswith("http://127.0.0.1:"):
            route.continue_()
        else:
            route.abort()

    try:
        page: Page = browser.new_page()
        page.route("**/*", fail_first_live_plot_request)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        live_plot = page.locator('img[data-src="/api/nyc/plots/live_temps"]')
        expect(live_plot).to_have_attribute(
            "src", re.compile(r"/api/nyc/plots/live_temps$")
        )
        expect(live_plot).to_have_attribute("data-status", "loaded")

        assert live_plot_attempts >= 2
        assert browser_smoke_server.request_counts["plots"] >= 1
    finally:
        browser.close()
        playwright.stop()


def test_deferred_plot_loading_shows_unavailable_after_retries(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """A plot that never becomes ready gets a quiet per-plot unavailable note."""
    playwright, browser = _launch_chromium()

    def fail_live_plot_request(route: Route) -> None:
        request = route.request
        if request.url.startswith("http://127.0.0.1:") and request.url.endswith(
            "/api/nyc/plots/live_temps"
        ):
            route.fulfill(status=503, body="Plot unavailable")
        elif request.url.startswith("http://127.0.0.1:"):
            route.continue_()
        else:
            route.abort()

    try:
        page: Page = browser.new_page()
        page.add_init_script("window.SWS_DEFERRED_PLOT_RETRY_DELAYS = [10];")
        page.route("**/*", fail_live_plot_request)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        live_plot = page.locator('img[data-src="/api/nyc/plots/live_temps"]')
        expect(live_plot).to_have_attribute("data-status", "unavailable")
        expect(
            page.locator('img[data-src="/api/nyc/plots/live_temps"] + .plot-status')
        ).to_have_text("Plot unavailable")
    finally:
        browser.close()
        playwright.stop()


def test_currents_page_updates_prediction(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """The currents page loads mocked prediction data and clears status text."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page()
        page.route("**/*", _block_external_requests)

        page.goto(f"{browser_smoke_server.base_url}/nyc/currents")

        expect(page.locator("#timestamp")).to_contain_text("5/12/2026")
        expect(page.locator("#state")).to_have_text("moderate ebb and building")
        expect(page.locator("#magnitude")).to_have_text("1.2")
        expect(page.locator("#currents-status")).to_be_hidden()
        expect(page.locator("#prev-hour-link")).to_have_attribute(
            "href", re.compile(r"/nyc/currents\?shift=-60$")
        )
        expect(page.locator("#next-hour-link")).to_have_attribute(
            "href", re.compile(r"/nyc/currents\?shift=60$")
        )

        assert browser_smoke_server.request_counts["currents"] == 1
    finally:
        browser.close()
        playwright.stop()


def test_initial_currents_failure_shows_unavailable_state(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """A failed first currents load does not leave placeholders spinning."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page()
        page.route("**/*", _block_external_and_fail_currents)

        page.goto(f"{browser_smoke_server.base_url}/nyc/currents")

        expect(page.locator("#currents-status")).to_have_text(
            "Unable to load current prediction. Please try again later."
        )
        expect(page.locator("#timestamp")).to_have_text("unavailable")
        expect(page.locator("#state")).to_have_text("unavailable")
        expect(page.locator("#magnitude")).to_have_text("N/A")

        assert browser_smoke_server.request_counts["currents"] == 0
    finally:
        browser.close()
        playwright.stop()


def test_currents_refresh_failure_keeps_loaded_data(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """A currents refresh failure keeps prior values and marks them as stale."""
    playwright, browser = _launch_chromium()
    current_attempts = 0

    def route_currents_once_then_fail(route: Route) -> None:
        nonlocal current_attempts
        request = route.request
        if request.url.startswith("http://127.0.0.1:") and urlparse(
            request.url
        ).path.endswith("/api/nyc/currents"):
            current_attempts += 1
            if current_attempts == 1:
                route.continue_()
            else:
                route.fulfill(status=503, body="Service unavailable")
        elif request.url.startswith("http://127.0.0.1:"):
            route.continue_()
        else:
            route.abort()

    try:
        page: Page = browser.new_page()
        page.route("**/*", route_currents_once_then_fail)

        page.goto(f"{browser_smoke_server.base_url}/nyc/currents")

        expect(page.locator("#state")).to_have_text("moderate ebb and building")
        expect(page.locator("#magnitude")).to_have_text("1.2")
        expect(page.locator("#currents-status")).to_be_hidden()

        page.evaluate("fetchCurrentsData('nyc')")

        expect(page.locator("#currents-status")).to_have_text(
            "Could not refresh current prediction. Showing last loaded data."
        )
        expect(page.locator("#state")).to_have_text("moderate ebb and building")
        expect(page.locator("#magnitude")).to_have_text("1.2")

        assert browser_smoke_server.request_counts["currents"] == 1
        assert current_attempts == 2
    finally:
        browser.close()
        playwright.stop()


def test_initial_conditions_failure_shows_unavailable_state(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """A failed first conditions load does not leave placeholders spinning."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page()
        page.route("**/*", _block_external_and_fail_conditions)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#conditions-status")).to_have_text(
            "Unable to load latest conditions. Please try again later."
        )
        expect(page.locator("#water-temp")).to_have_text("Unavailable")
        expect(page.locator("#temp-station-info")).to_have_text(
            "Current water temperature is unavailable."
        )
        expect(page.locator("#past-tide-type")).to_have_text("Unavailable")
        expect(page.locator("#past-tide-date")).to_have_text("Unavailable")
        expect(page.locator("#past-tide-time")).to_have_text("Unavailable")
        expect(page.locator("#next-tide-0-type")).to_have_text("Unavailable")
        expect(page.locator("#next-tide-1-type")).to_have_text("Unavailable")
        expect(page.locator("#current-state-summary")).to_have_text("unavailable")
        expect(page.locator("#current-magnitude")).to_have_text("N/A")

        assert browser_smoke_server.request_counts["conditions"] == 0
    finally:
        browser.close()
        playwright.stop()


def test_conditions_refresh_failure_keeps_loaded_data(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """A refresh failure keeps prior values and marks them as stale."""
    playwright, browser = _launch_chromium()
    condition_attempts = 0

    def route_conditions_once_then_fail(route: Route) -> None:
        nonlocal condition_attempts
        request = route.request
        if request.url.startswith("http://127.0.0.1:") and request.url.endswith(
            "/api/nyc/conditions"
        ):
            condition_attempts += 1
            if condition_attempts == 1:
                route.continue_()
            else:
                route.fulfill(status=503, body="Service unavailable")
        elif request.url.startswith("http://127.0.0.1:"):
            route.continue_()
        else:
            route.abort()

    try:
        page: Page = browser.new_page()
        page.route("**/*", route_conditions_once_then_fail)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        expect(page.locator("#past-tide-type")).to_have_text("high")
        expect(page.locator("#current-state-summary")).to_have_text(
            "strong ebb and easing"
        )
        expect(page.locator("#conditions-status")).to_be_hidden()

        page.evaluate("fetchAndUpdateConditions('nyc')")

        expect(page.locator("#conditions-status")).to_have_text(
            "Could not refresh latest conditions. Showing last loaded data."
        )
        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        expect(page.locator("#past-tide-type")).to_have_text("high")
        expect(page.locator("#current-state-summary")).to_have_text(
            "strong ebb and easing"
        )
        expect(page.locator("#current-magnitude")).to_have_text("1.4")

        assert browser_smoke_server.request_counts["conditions"] == 1
        assert condition_attempts == 2
    finally:
        browser.close()
        playwright.stop()


def test_debug_tool_is_quiet_without_debug_parameter(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """The debug script passively tracks data without visible UI or console noise."""
    playwright, browser = _launch_chromium()
    console_messages: list[str] = []

    try:
        page: Page = browser.new_page()
        page.on("console", lambda message: console_messages.append(message.text))
        page.route("**/*", _block_external_requests)

        page.goto(f"{browser_smoke_server.base_url}/nyc")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        expect(page.locator("#sws-debug-btn")).to_have_count(0)
        expect(page.locator("#sws-debug-panel")).to_have_count(0)

        debug_api_call_count = page.evaluate("window.SWS_DEBUG_STATE.apiCalls.length")
        assert debug_api_call_count >= 1
        assert not any(message.startswith("[DEBUG:") for message in console_messages)
        assert "Debug mode disabled. Add ?debug=1 to URL to enable debugger." not in (
            console_messages
        )
    finally:
        browser.close()
        playwright.stop()


def test_debug_tool_panel_opens_with_debug_parameter(
    browser_smoke_server: BrowserSmokeServer,
) -> None:
    """The opt-in debug UI opens and renders captured state safely."""
    playwright, browser = _launch_chromium()

    try:
        page: Page = browser.new_page()
        page.route("**/*", _block_external_requests)

        page.goto(f"{browser_smoke_server.base_url}/nyc?debug=1")

        expect(page.locator("#water-temp")).to_have_text("53.1°F")
        debug_button = page.locator("#sws-debug-btn")
        expect(debug_button).to_be_visible()
        expect(page.locator("#sws-debug-panel")).to_have_count(0)

        debug_button.click()

        debug_panel = page.locator("#sws-debug-panel")
        expect(debug_panel).to_be_visible()
        expect(debug_panel).to_contain_text("Debug Info")
        expect(debug_panel).to_contain_text("Browser Information")
        expect(debug_panel).to_contain_text("Recent API Calls")
        expect(debug_panel).to_contain_text("/api/nyc/conditions")
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
