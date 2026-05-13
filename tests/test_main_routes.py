"""Tests for top-level HTML and crawler routes."""

from fastapi.testclient import TestClient

from shallweswim import canonical, config
from shallweswim.main import app, start_app


def test_www_host_redirects_to_canonical_apex() -> None:
    """The duplicate www host redirects to the canonical apex host."""
    client = TestClient(app)

    response = client.get(
        "/nyc?foo=bar",
        headers={"host": "www.shallweswim.today"},
        follow_redirects=False,
    )

    assert response.status_code == 301
    assert response.headers["location"] == f"{canonical.CANONICAL_BASE_URL}/nyc?foo=bar"


def test_robots_txt_points_to_sitemap() -> None:
    """robots.txt advertises the canonical sitemap."""
    client = TestClient(app)

    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "User-agent: *" in response.text
    assert f"Sitemap: {canonical.CANONICAL_BASE_URL}/sitemap.xml" in response.text


def test_sitemap_lists_canonical_location_pages() -> None:
    """Sitemap contains only canonical apex-host URLs for indexable pages."""
    client = TestClient(app)

    response = client.get("/sitemap.xml")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert f"<loc>{canonical.CANONICAL_BASE_URL}/all</loc>" in response.text
    assert "www.shallweswim.today" not in response.text

    for loc_code in config.CONFIGS:
        assert f"<loc>{canonical.CANONICAL_BASE_URL}/{loc_code}</loc>" in response.text


def test_app_route_returns_clear_not_built_response_when_dist_missing(tmp_path) -> None:
    """Local /app requests are explicit when the frontend build is missing."""
    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(tmp_path / "missing-dist")

    try:
        response = client.get("/app")
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    assert response.status_code == 404
    assert "Frontend app shell has not been built" in response.text
    assert response.headers["cache-control"] == "no-cache, must-revalidate"


def test_app_routes_serve_built_shell_and_assets(tmp_path) -> None:
    """Built frontend shell is served for client routes under /app."""
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id='root'></div>")
    (dist / "manifest.webmanifest").write_text('{"name":"Shall We Swim"}')
    (assets / "app.js").write_text("console.log('app')")

    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(dist)

    try:
        shell = client.get("/app/nyc/currents")
        manifest = client.get("/app/manifest.webmanifest")
        asset = client.get("/app/assets/app.js")
        missing_asset = client.get("/app/assets/missing.js")
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    assert shell.status_code == 200
    assert "<div id='root'></div>" in shell.text
    assert shell.headers["cache-control"] == "no-cache, must-revalidate"
    assert manifest.status_code == 200
    assert manifest.headers["cache-control"] == "no-cache, must-revalidate"
    assert asset.status_code == 200
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert missing_asset.status_code == 404


def test_app_assets_reject_encoded_path_traversal(tmp_path) -> None:
    """Encoded asset paths cannot escape the built frontend assets directory."""
    secret = tmp_path / "secret.txt"
    secret.write_text("do not expose")

    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id='root'></div>")
    (assets / "app.js").write_text("console.log('app')")

    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(dist)

    traversal_paths = [
        "/app/assets/%2e%2e/index.html",
        "/app/assets/%2e%2e/%2e%2e/secret.txt",
        "/app/assets/..%2F..%2Fsecret.txt",
    ]

    try:
        responses = [client.get(path) for path in traversal_paths]
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    for response in responses:
        assert response.status_code == 404
        assert "do not expose" not in response.text
        assert "<div id='root'></div>" not in response.text


def test_start_app_requires_frontend_dist_when_configured(tmp_path) -> None:
    """Production startup fails loudly if the built frontend shell is absent."""
    original_frontend_dist = getattr(app.state, "frontend_dist", None)

    try:
        try:
            start_app(
                frontend_dist=str(tmp_path / "missing-dist"),
                require_frontend_dist=True,
            )
        except RuntimeError as exc:
            assert "Frontend app shell is missing" in str(exc)
        else:
            raise AssertionError("start_app should fail when frontend dist is required")
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist


def test_location_page_has_canonical_url() -> None:
    """Location pages identify their canonical apex-host URL."""
    client = TestClient(app)

    response = client.get("/nyc")

    assert response.status_code == 200
    assert (
        f'<link rel="canonical" href="{canonical.CANONICAL_BASE_URL}/nyc"'
        in response.text
    )


def test_location_page_renders_frontend_bootstrap() -> None:
    """Location pages include the app script and data placeholders."""
    client = TestClient(app)

    response = client.get("/nyc")

    assert response.status_code == 200
    assert 'locationCode: "nyc"' in response.text
    assert 'src="/static/main.js"' in response.text
    assert 'id="conditions-status" hidden' in response.text
    assert 'id="water-temp"' in response.text
    assert 'id="tides-section"' in response.text
    assert 'id="past-tide-type">...</strong>' in response.text
    assert 'id="current-magnitude"' in response.text
    assert 'id="current-state-summary">...</span>' in response.text


def test_location_page_defers_temperature_plot_loading() -> None:
    """Temperature plot images do not race conditions during initial HTML parse."""
    client = TestClient(app)

    response = client.get("/nyc")

    assert response.status_code == 200
    assert 'class="plot deferred-plot"' in response.text
    assert 'data-src="/api/nyc/plots/live_temps"' in response.text
    assert 'data-src="/api/nyc/plots/historic_temps?period=2mo"' in response.text
    assert 'data-src="/api/nyc/plots/historic_temps?period=12mo"' in response.text
    assert ' src="/api/nyc/plots/live_temps"' not in response.text
    assert ' src="/api/nyc/plots/historic_temps?period=2mo"' not in response.text
    assert ' src="/api/nyc/plots/historic_temps?period=12mo"' not in response.text


def test_location_page_renders_windy_embed_with_layout_dimensions() -> None:
    """Windy embed dimensions match the responsive iframe's desktop layout."""
    client = TestClient(app)

    response = client.get("/nyc")

    assert response.status_code == 200
    assert 'class="windyframe"' in response.text
    assert "https://embed.windy.com/embed2.html" in response.text
    assert "width=950" in response.text
    assert "height=350" in response.text
    assert "overlay=waves" in response.text
    assert "product=ecmwfWaves" in response.text


def test_currents_page_renders_without_empty_image_sources() -> None:
    """Currents page defers chart image sources until API data is loaded."""
    client = TestClient(app)

    response = client.get("/nyc/currents")

    assert response.status_code == 200
    assert 'locationCode: "nyc"' in response.text
    assert 'src="/static/main.js"' in response.text
    assert 'id="currents-status" hidden' in response.text
    assert 'id="timestamp">...</span>' in response.text
    assert 'id="state">...</span>' in response.text
    assert 'id="magnitude">...</span>' in response.text
    assert 'id="current-chart"' in response.text
    assert 'id="tide-current-plot"' in response.text
    assert 'id="legacy-chart"' in response.text
    assert 'src=""' not in response.text


def test_all_locations_page_renders_widgets() -> None:
    """All-locations page renders one frontend widget per enabled location."""
    client = TestClient(app)

    response = client.get("/all")

    assert response.status_code == 200
    assert "/api/${locationCode}/conditions" in response.text
    for loc_code in config.CONFIGS:
        assert f'data-location="{loc_code}"' in response.text


def test_widget_page_renders_standalone_widget() -> None:
    """Standalone widget page includes its location and API loader."""
    client = TestClient(app)

    response = client.get("/nyc/widget")

    assert response.status_code == 200
    assert 'data-location="nyc"' in response.text
    assert 'id="widget-water-temp"' in response.text
    assert "/api/${locationCode}/conditions" in response.text
