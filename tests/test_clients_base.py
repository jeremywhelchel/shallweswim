"""Tests for shared API client infrastructure."""

import asyncio

import pytest

from shallweswim.clients.base import provider_request_slot


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
