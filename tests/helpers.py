"""Utility functions for tests."""

import json
from typing import Any

from fastapi import FastAPI


def create_test_app(**kwargs: Any) -> FastAPI:
    """Create a FastAPI app configured for testing.

    FastAPI 0.130+ uses Pydantic's native JSON serialization which handles
    NaN -> null conversion automatically when response_model is set.

    Args:
        **kwargs: Additional arguments passed to FastAPI constructor.

    Returns:
        Configured FastAPI application.
    """
    return FastAPI(**kwargs)


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
