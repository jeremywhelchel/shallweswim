"""Provider-neutral logging configuration for ShallWeSwim."""

import json
import logging
import os
import sys
from typing import Any, TextIO

# Determine project root for relative log paths (directory containing this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FORMAT_ENV_VAR = "SHALLWESWIM_LOG_FORMAT"
STRUCTURED_FIELDS = (
    "component",
    "operation",
    "location",
    "feed",
    "provider",
    "outcome",
    "run_id",
    "generation_id",
    "duration_ms",
    "record_count",
)


class RelativePathFilter(logging.Filter):
    """Logging filter to add relative path attribute to LogRecords."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Calculate and add 'relativepath' to the log record.

        Args:
            record: The log record to process

        Returns:
            Always True to process the record
        """
        try:
            # Use os.path.normpath for cross-platform compatibility
            record.relativepath = os.path.relpath(
                os.path.normpath(record.pathname), PROJECT_ROOT
            )
        except ValueError:
            record.relativepath = record.pathname  # Fallback if calculation fails
        return True  # Always process the record


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible value without failing application logging."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


class JsonLogFormatter(logging.Formatter):
    """Format each log event as one JSON object for container platforms."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "source": {
                "file": getattr(record, "relativepath", record.pathname),
                "line": record.lineno,
                "function": record.funcName,
            },
        }
        for field in STRUCTURED_FIELDS:
            if hasattr(record, field):
                payload[field] = _json_safe(getattr(record, field))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )


def _create_handler(log_format: str, stream: TextIO | None = None) -> logging.Handler:
    """Create the requested stream handler."""
    if log_format == "console":
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(relativepath)s:%(lineno)d: %(message)s")
        )
    elif log_format == "json":
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(JsonLogFormatter())
    else:
        raise ValueError(
            f"Invalid {LOG_FORMAT_ENV_VAR} value {log_format!r}; expected 'console' or 'json'"
        )
    handler.addFilter(RelativePathFilter())
    return handler


def setup_logging(log_format: str | None = None) -> str:
    """Configure logging and return the validated selected format."""
    selected_format = log_format or os.environ.get(LOG_FORMAT_ENV_VAR, "console")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.filters.clear()
    root_logger.addHandler(_create_handler(selected_format))
    root_logger.setLevel(logging.INFO)
    logging.info("Logging configured with %s format", selected_format)
    return selected_format
