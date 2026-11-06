"""HTTP response compression settings."""

# Media types the gzip middleware passes through uncompressed: event streams,
# and formats that are already compressed.
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
