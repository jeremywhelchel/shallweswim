"""Tests for canonical URL helper functions."""

from xml.etree import ElementTree

from shallweswim import canonical


def test_canonical_redirect_url_for_www_host_preserves_path_and_query() -> None:
    """Duplicate www host redirects to apex host without dropping the query."""
    assert (
        canonical.canonical_redirect_url(
            hostname="www.shallweswim.today",
            path="/nyc",
            query="utm_source=test",
        )
        == "https://shallweswim.today/nyc?utm_source=test"
    )


def test_canonical_redirect_url_ignores_canonical_host() -> None:
    """The canonical host should not redirect to itself."""
    assert (
        canonical.canonical_redirect_url(
            hostname="shallweswim.today",
            path="/nyc",
            query="",
        )
        is None
    )


def test_sitemap_xml_uses_xml_serializer() -> None:
    """Sitemap XML is serialized with proper escaping and namespace handling."""
    sitemap = canonical.sitemap_xml(
        [
            "https://shallweswim.today/nyc",
            "https://shallweswim.today/search?q=a&b=c",
        ]
    )

    root = ElementTree.fromstring(sitemap)
    namespace = {"sitemap": canonical.SITEMAP_NAMESPACE}
    locs = [loc.text for loc in root.findall("sitemap:url/sitemap:loc", namespace)]

    assert root.tag == f"{{{canonical.SITEMAP_NAMESPACE}}}urlset"
    assert locs == [
        "https://shallweswim.today/nyc",
        "https://shallweswim.today/search?q=a&b=c",
    ]
