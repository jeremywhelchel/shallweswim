import json
import logging
from io import StringIO
from pathlib import Path

import pytest

from shallweswim.logging_utils import PROJECT_ROOT, _create_handler, setup_logging


def _record(**kwargs: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="shallweswim.core.feeds",
        level=logging.INFO,
        pathname=str(Path(PROJECT_ROOT) / "shallweswim/core/feeds.py"),
        lineno=123,
        msg="Temperature %s",
        args=("updated",),
        exc_info=None,
        func="update",
    )
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


def test_json_log_has_cloud_severity_message_and_source() -> None:
    stream = StringIO()
    handler = _create_handler("json", stream)

    handler.handle(_record())

    payload = json.loads(stream.getvalue())
    assert payload == {
        "severity": "INFO",
        "message": "Temperature updated",
        "logger": "shallweswim.core.feeds",
        "source": {
            "file": "shallweswim/core/feeds.py",
            "line": 123,
            "function": "update",
        },
    }
    assert stream.getvalue().count("\n") == 1


def test_json_log_includes_only_approved_extra_fields() -> None:
    stream = StringIO()
    handler = _create_handler("json", stream)

    handler.handle(_record(location="nyc", duration_ms=12.5, api_token="secret"))

    payload = json.loads(stream.getvalue())
    assert payload["location"] == "nyc"
    assert payload["duration_ms"] == 12.5
    assert "api_token" not in payload


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_json_log_converts_non_finite_numbers_to_valid_json(value: float) -> None:
    stream = StringIO()
    handler = _create_handler("json", stream)

    handler.handle(_record(duration_ms=value))

    payload = json.loads(
        stream.getvalue(),
        parse_constant=lambda constant: pytest.fail(f"invalid JSON: {constant}"),
    )
    assert payload["duration_ms"] == str(value)


def test_console_log_remains_human_readable() -> None:
    stream = StringIO()
    handler = _create_handler("console", stream)

    handler.handle(_record())

    assert stream.getvalue() == (
        "INFO:shallweswim/core/feeds.py:123: Temperature updated\n"
    )


def test_invalid_log_format_fails_fast() -> None:
    with pytest.raises(ValueError, match="expected 'console' or 'json'"):
        _create_handler("xml")


def test_setup_logging_returns_validated_selected_format() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_filters = root_logger.filters.copy()
    original_level = root_logger.level
    try:
        assert setup_logging("console") == "console"
    finally:
        root_logger.handlers = original_handlers
        root_logger.filters = original_filters
        root_logger.setLevel(original_level)
