"""Tests for top-level HTML and crawler routes."""

import pytest
from fastapi.testclient import TestClient

from shallweswim import canonical, config
from shallweswim.main import app, start_app


def _write_fake_frontend_dist(dist) -> None:
    """Create a minimal Vite-like app shell for route tests."""
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "<title>generic shell</title>",
                '<script type="module" crossorigin src="/assets/index-abc.js"></script>',
                '<link rel="stylesheet" crossorigin href="/assets/index-def.css">',
                "</head>",
                "<body>",
                '<div id="root"></div>',
                "</body>",
                "</html>",
            ]
        )
    )
    (assets / "app.js").write_text("console.log('app')")


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


def test_root_manifest_uses_root_app_scope() -> None:
    """The canonical web manifest launches the root-mounted React app."""
    client = TestClient(app)

    response = client.get("/manifest.json")

    assert response.status_code == 200
    assert not response.history
    assert response.headers["content-type"].startswith("application/manifest+json")
    assert response.headers["cache-control"] == "no-cache, must-revalidate"
    data = response.json()
    assert data["name"] == "shall we swim?"
    assert data["short_name"] == "shallweswim"
    assert data["start_url"] == "/?source=pwa-react"
    assert data["scope"] == "/"
    assert data["background_color"] == "#000099"


def test_sitemap_lists_canonical_location_pages() -> None:
    """Sitemap contains only canonical apex-host URLs for indexable pages."""
    client = TestClient(app)

    response = client.get("/sitemap.xml")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert f"<loc>{canonical.CANONICAL_BASE_URL}/locations</loc>" in response.text
    assert "www.shallweswim.today" not in response.text

    for loc_code in config.CONFIGS:
        assert f"<loc>{canonical.CANONICAL_BASE_URL}/{loc_code}</loc>" in response.text


def test_root_app_route_returns_clear_not_built_response_when_dist_missing(
    tmp_path,
) -> None:
    """Local app requests are explicit when the frontend build is missing."""
    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(tmp_path / "missing-dist")

    try:
        response = client.get("/")
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    assert response.status_code == 404
    assert "Frontend app shell has not been built" in response.text
    assert response.headers["cache-control"] == "no-cache, must-revalidate"


def test_app_routes_serve_built_shell_and_assets(tmp_path) -> None:
    """Built frontend shell is served for canonical root app routes."""
    dist = tmp_path / "dist"
    _write_fake_frontend_dist(dist)

    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(dist)

    try:
        root = client.get("/")
        location_shell = client.get("/nyc")
        locations_shell = client.get("/locations")
        removed_app_route = client.get("/app")
        invalid_location = client.get("/zzz")
        asset = client.get("/assets/app.js")
        missing_asset = client.get("/assets/missing.js")
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    for shell in [root, location_shell, locations_shell]:
        assert shell.status_code == 200
        assert '<div id="root"></div>' in shell.text
        assert 'src="/assets/index-abc.js"' in shell.text
        assert 'href="/assets/index-def.css"' in shell.text
        assert "<title>generic shell</title>" not in shell.text
        assert shell.headers["cache-control"] == "no-cache, must-revalidate"

    assert removed_app_route.status_code == 404
    assert invalid_location.status_code == 404
    assert asset.status_code == 200
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert missing_asset.status_code == 404


def test_root_app_route_renders_default_location_metadata(tmp_path) -> None:
    """The root app shell exposes useful default-location HTML before JS."""
    dist = tmp_path / "dist"
    _write_fake_frontend_dist(dist)
    cfg = config.CONFIGS["nyc"]

    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(dist)

    try:
        response = client.get("/")
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    assert response.status_code == 200
    assert (
        f"<title>{cfg.swim_location} swim conditions | shall we swim?</title>"
        in response.text
    )
    assert (
        f'<link rel="canonical" href="{canonical.CANONICAL_BASE_URL}/">'
        in response.text
    )
    assert (
        '<link rel="alternate" type="application/json" '
        f'href="{canonical.CANONICAL_BASE_URL}/api/nyc/conditions">'
    ) in response.text
    assert '<meta property="og:type" content="website">' in response.text
    assert '<meta property="og:site_name" content="shall we swim?">' in response.text
    assert (
        '<meta property="og:image" '
        f'content="{canonical.CANONICAL_BASE_URL}/static/android-chrome-512x512.png">'
    ) in response.text
    assert '<meta name="twitter:card" content="summary">' in response.text
    assert (
        '<link rel="icon" type="image/png" sizes="32x32" '
        'href="/static/favicon-32x32.png">'
    ) in response.text
    assert '"@type": "WebSite"' in response.text
    assert '"@type": "WebPage"' in response.text
    assert '"@type": "Place"' in response.text
    assert "<noscript>" in response.text
    assert 'class="durable-fallback"' in response.text
    assert "text-decoration:underline" in response.text
    assert cfg.name in response.text
    assert cfg.swim_location in response.text
    assert cfg.swim_location_link in response.text
    assert cfg.description in response.text
    assert (
        '<li><a href="/api/nyc/conditions">Condition data as JSON</a></li>'
        in response.text
    )
    assert (
        '<li><a href="/api/locations">All locations as JSON</a></li>' in response.text
    )
    assert '<li><a href="/locations">All locations</a></li>' in response.text


def test_location_app_route_renders_location_metadata(tmp_path) -> None:
    """Configured location routes expose location-specific durable HTML."""
    dist = tmp_path / "dist"
    _write_fake_frontend_dist(dist)
    cfg = config.CONFIGS["nyc"]

    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(dist)

    try:
        response = client.get("/nyc")
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    assert response.status_code == 200
    assert (
        f"<title>{cfg.swim_location} swim conditions | shall we swim?</title>"
        in response.text
    )
    assert (
        f'<link rel="canonical" href="{canonical.CANONICAL_BASE_URL}/nyc">'
        in response.text
    )
    assert (
        '<link rel="alternate" type="application/json" '
        f'href="{canonical.CANONICAL_BASE_URL}/api/nyc/conditions">'
    ) in response.text
    assert '"latitude": 40.573' in response.text
    assert '"longitude": -73.954' in response.text
    assert '<div id="root"></div><noscript>' in response.text
    assert 'class="durable-fallback"' in response.text
    assert cfg.swim_location_link in response.text
    assert cfg.description in response.text
    assert 'href="/api/nyc/conditions"' in response.text


def test_locations_app_route_renders_all_locations_metadata(tmp_path) -> None:
    """The all-locations app shell advertises location discovery."""
    dist = tmp_path / "dist"
    _write_fake_frontend_dist(dist)

    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(dist)

    try:
        response = client.get("/locations")
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    assert response.status_code == 200
    assert (
        "<title>Open water swimming locations | shall we swim?</title>" in response.text
    )
    assert (
        f'<link rel="canonical" href="{canonical.CANONICAL_BASE_URL}/locations">'
        in response.text
    )
    assert (
        '<link rel="alternate" type="application/json" '
        f'href="{canonical.CANONICAL_BASE_URL}/api/locations">'
    ) in response.text
    assert '"@type": "WebPage"' in response.text
    assert '"@type": "Place"' not in response.text
    assert "Open water swimming locations" in response.text
    assert 'href="/api/locations"' in response.text
    for loc_code, cfg in config.CONFIGS.items():
        assert f'href="/{loc_code}"' in response.text
        assert cfg.swim_location in response.text


def test_app_assets_reject_encoded_path_traversal(tmp_path) -> None:
    """Encoded asset paths cannot escape the built frontend assets directory."""
    secret = tmp_path / "secret.txt"
    secret.write_text("do not expose")

    dist = tmp_path / "dist"
    _write_fake_frontend_dist(dist)

    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(dist)

    traversal_paths = [
        "/assets/%2e%2e/index.html",
        "/assets/%2e%2e/%2e%2e/secret.txt",
        "/assets/..%2F..%2Fsecret.txt",
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


@pytest.mark.integration
def test_canonical_html_routes_expose_real_config_metadata(tmp_path) -> None:
    """Integration coverage for durable HTML using production location config."""
    dist = tmp_path / "dist"
    _write_fake_frontend_dist(dist)

    client = TestClient(app)
    original_frontend_dist = getattr(app.state, "frontend_dist", None)
    app.state.frontend_dist = str(dist)

    try:
        responses = {
            "/": client.get("/"),
            "/locations": client.get("/locations"),
            "/nyc": client.get("/nyc"),
        }
    finally:
        if original_frontend_dist is None:
            del app.state.frontend_dist
        else:
            app.state.frontend_dist = original_frontend_dist

    for path, response in responses.items():
        assert response.status_code == 200
        assert f'href="{canonical.canonical_url(path)}"' in response.text
        assert '<link rel="alternate" type="application/json"' in response.text
        assert '<script type="application/ld+json">' in response.text


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


def test_legacy_location_page_has_root_canonical_url() -> None:
    """Legacy location pages point canonical metadata at the root app route."""
    client = TestClient(app)

    response = client.get("/legacy/nyc")

    assert response.status_code == 200
    assert (
        f'<link rel="canonical" href="{canonical.CANONICAL_BASE_URL}/nyc"'
        in response.text
    )


def test_legacy_location_page_renders_frontend_bootstrap() -> None:
    """Legacy location pages include the app script and data placeholders."""
    client = TestClient(app)

    response = client.get("/legacy/nyc")

    assert response.status_code == 200
    assert 'locationCode: "nyc"' in response.text
    assert 'src="/static/main.js"' in response.text
    assert 'id="conditions-status" hidden' in response.text
    assert 'id="water-temp"' in response.text
    assert 'id="tides-section"' in response.text
    assert 'id="past-tide-type">...</strong>' in response.text
    assert 'id="current-magnitude"' in response.text
    assert 'id="current-state-summary">...</span>' in response.text


def test_legacy_location_page_defers_temperature_plot_loading() -> None:
    """Temperature plot images do not race conditions during initial HTML parse."""
    client = TestClient(app)

    response = client.get("/legacy/nyc")

    assert response.status_code == 200
    assert 'class="plot deferred-plot"' in response.text
    assert 'data-src="/api/nyc/plots/live_temps"' in response.text
    assert 'data-src="/api/nyc/plots/historic_temps?period=2mo"' in response.text
    assert 'data-src="/api/nyc/plots/historic_temps?period=12mo"' in response.text
    assert ' src="/api/nyc/plots/live_temps"' not in response.text
    assert ' src="/api/nyc/plots/historic_temps?period=2mo"' not in response.text
    assert ' src="/api/nyc/plots/historic_temps?period=12mo"' not in response.text


def test_legacy_location_page_renders_windy_embed_with_layout_dimensions() -> None:
    """Windy embed dimensions match the responsive iframe's desktop layout."""
    client = TestClient(app)

    response = client.get("/legacy/nyc")

    assert response.status_code == 200
    assert 'class="windyframe"' in response.text
    assert "https://embed.windy.com/embed2.html" in response.text
    assert "width=950" in response.text
    assert "height=350" in response.text
    assert "overlay=waves" in response.text
    assert "product=ecmwfWaves" in response.text


def test_existing_embed_route_still_serves_legacy_embed_content() -> None:
    """Existing external embed URLs keep serving the legacy embed page."""
    client = TestClient(app)

    response = client.get("/nyc/embed")

    assert response.status_code == 200
    assert 'class="embed-page"' in response.text
    assert (
        f'<link rel="canonical" href="{canonical.CANONICAL_BASE_URL}/nyc/embed"'
        in response.text
    )
    assert "https://embed.windy.com/embed2.html" in response.text
    assert "width=950" in response.text
    assert "height=350" in response.text


def test_root_embed_route_keeps_existing_redirect() -> None:
    """The historical /embed shortcut still redirects to the default embed."""
    client = TestClient(app, follow_redirects=False)

    response = client.get("/embed")

    assert response.status_code == 301
    assert response.headers["location"] == "/nyc/embed"


def test_currents_page_renders_without_empty_image_sources() -> None:
    """Currents page defers chart image sources until API data is loaded."""
    client = TestClient(app)

    response = client.get("/legacy/nyc/currents")

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
    """Legacy all-locations page renders one frontend widget per enabled location."""
    client = TestClient(app)

    response = client.get("/legacy/all")

    assert response.status_code == 200
    assert (
        f'<link rel="canonical" href="{canonical.CANONICAL_BASE_URL}/locations"'
        in response.text
    )
    assert "/api/${locationCode}/conditions" in response.text
    for loc_code in config.CONFIGS:
        assert f'data-location="{loc_code}"' in response.text


def test_widget_page_renders_standalone_widget() -> None:
    """Legacy standalone widget page includes its location and API loader."""
    client = TestClient(app)

    response = client.get("/legacy/nyc/widget")

    assert response.status_code == 200
    assert 'data-location="nyc"' in response.text
    assert 'id="widget-water-temp"' in response.text
    assert "/api/${locationCode}/conditions" in response.text
