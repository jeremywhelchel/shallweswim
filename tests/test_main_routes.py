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
