"""Tests for top-level HTML and crawler routes."""

from fastapi.testclient import TestClient

from shallweswim import canonical, config
from shallweswim.main import app


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
