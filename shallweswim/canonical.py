"""Canonical URL and crawler metadata helpers."""

from collections.abc import Iterable
from xml.etree import ElementTree

CANONICAL_HOST = "shallweswim.today"
CANONICAL_BASE_URL = f"https://{CANONICAL_HOST}"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"


def canonical_url(path: str) -> str:
    """Build a canonical absolute URL for an app path."""
    return f"{CANONICAL_BASE_URL}{path}"


def canonical_redirect_url(hostname: str | None, path: str, query: str) -> str | None:
    """Return the canonical URL for duplicate production host requests."""
    if hostname != f"www.{CANONICAL_HOST}":
        return None

    query_string = f"?{query}" if query else ""
    return f"{CANONICAL_BASE_URL}{path or '/'}{query_string}"


def robots_txt() -> str:
    """Build crawler directives for the canonical site."""
    return "\n".join(
        [
            "User-agent: *",
            "Allow: /",
            f"Sitemap: {canonical_url('/sitemap.xml')}",
            "",
        ]
    )


def sitemap_xml(urls: Iterable[str]) -> str:
    """Build a sitemap XML document for canonical URLs."""
    ElementTree.register_namespace("", SITEMAP_NAMESPACE)
    urlset = ElementTree.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")

    for url in urls:
        url_element = ElementTree.SubElement(urlset, f"{{{SITEMAP_NAMESPACE}}}url")
        loc_element = ElementTree.SubElement(url_element, f"{{{SITEMAP_NAMESPACE}}}loc")
        loc_element.text = url

    return ElementTree.tostring(
        urlset,
        encoding="unicode",
        xml_declaration=True,
    )
