"""HTTP response compression helpers."""

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipResponder
from starlette.types import ASGIApp, Receive, Scope, Send

GZIP_EXCLUDED_CONTENT_TYPES = (
    "text/event-stream",
    "image/avif",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/x-icon",
    "application/gzip",
    "application/pdf",
    "application/x-gzip",
    "application/x-zip-compressed",
    "application/zip",
    "font/woff",
    "font/woff2",
)


def is_gzip_excluded_content_type(content_type: str) -> bool:
    """Return whether a response content type should skip gzip compression."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type in GZIP_EXCLUDED_CONTENT_TYPES


class SelectiveGZipResponder(GZipResponder):
    """Starlette gzip responder that skips already-compressed binary responses."""

    async def send_with_compression(self, message):
        if message["type"] == "http.response.start":
            self.initial_message = message
            headers = Headers(raw=self.initial_message["headers"])
            self.content_encoding_set = "content-encoding" in headers
            self.content_type_is_excluded = is_gzip_excluded_content_type(
                headers.get("content-type", "")
            )
            return

        await super().send_with_compression(message)


class SelectiveGZipMiddleware:
    """Gzip compress eligible responses while skipping already-compressed assets."""

    def __init__(
        self, app: ASGIApp, minimum_size: int = 500, compresslevel: int = 9
    ) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.compresslevel = compresslevel

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # pragma: no cover
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if "gzip" not in headers.get("Accept-Encoding", ""):
            await self.app(scope, receive, send)
            return

        responder = SelectiveGZipResponder(
            self.app, self.minimum_size, compresslevel=self.compresslevel
        )
        await responder(scope, receive, send)
