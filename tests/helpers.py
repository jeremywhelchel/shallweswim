"""Utility functions for tests."""

import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse


def create_test_app(**kwargs: Any) -> FastAPI:
    """Create a FastAPI app configured for testing.

    Uses ORJSONResponse to match production config (handles NaN -> null).

    Args:
        **kwargs: Additional arguments passed to FastAPI constructor.

    Returns:
        Configured FastAPI application.
    """
    return FastAPI(default_response_class=ORJSONResponse, **kwargs)


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
