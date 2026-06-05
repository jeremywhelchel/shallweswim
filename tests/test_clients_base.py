"""Tests for shared API client infrastructure."""

import asyncio

import aiohttp
import pytest

from shallweswim.clients.base import (
    RetryableClientError,
    provider_request_slot,
    raise_if_retryable_http_status,
    request_timeout,
    retryable_network_error,
    retryable_timeout_error,
)


@pytest.mark.asyncio
async def test_provider_request_slot_limits_active_requests() -> None:
    """Provider gates limit active work while preserving async fanout."""
    active_requests = 0
    max_active_requests = 0

    async def simulated_request() -> None:
        nonlocal active_requests, max_active_requests
        async with provider_request_slot("test-provider-limit", 2):
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            await asyncio.sleep(0)
            active_requests -= 1

    await asyncio.gather(*(simulated_request() for _ in range(10)))

    assert max_active_requests == 2


@pytest.mark.asyncio
async def test_provider_request_slot_rejects_invalid_limit() -> None:
    """Provider gates require at least one active request slot."""
    with pytest.raises(ValueError, match="at least 1"):
        async with provider_request_slot("test-provider-invalid", 0):
            pass


def test_request_timeout_builds_total_timeout() -> None:
    """Shared timeout helper creates the aiohttp timeout object clients use."""
    timeout = request_timeout(12.5)

    assert isinstance(timeout, aiohttp.ClientTimeout)
    assert timeout.total == 12.5


def test_retryable_timeout_error_uses_consistent_message() -> None:
    """Shared timeout errors keep provider/resource details."""
    error = retryable_timeout_error(
        timeout_seconds=30.0,
        provider="NWIS",
        resource="site 08155500",
    )

    assert isinstance(error, RetryableClientError)
    assert str(error) == "Request timed out after 30.0s for NWIS site 08155500"


def test_retryable_network_error_includes_exception_details() -> None:
    """Shared network errors preserve exception type and message."""
    source_error = aiohttp.ClientPayloadError("bad gzip")

    error = retryable_network_error(
        provider="NWIS",
        action="for site 08155500",
        error=source_error,
    )

    assert isinstance(error, RetryableClientError)
    assert (
        str(error)
        == "Network error during NWIS request for site 08155500: ClientPayloadError: bad gzip"
    )


def test_raise_if_retryable_http_status_raises_for_retryable_status() -> None:
    """Retryable HTTP statuses are converted to retryable client errors."""
    with pytest.raises(RetryableClientError, match="HTTP 503"):
        raise_if_retryable_http_status(503, "HTTP 503")


def test_raise_if_retryable_http_status_ignores_terminal_status() -> None:
    """Non-retryable HTTP statuses remain client-specific terminal errors."""
    raise_if_retryable_http_status(404, "HTTP 404")
