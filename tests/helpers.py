"""Utility functions for tests."""

import json
from typing import Any


def assert_json_serializable(obj: Any) -> None:
    """Assert that an object is JSON serializable.

    Args:
        obj: The object to check for JSON serializability.

    Raises:
        AssertionError: If the object is not JSON serializable.
    """
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"Object is not JSON serializable: {e}") from e
