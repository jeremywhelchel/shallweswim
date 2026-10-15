"""Tests for the provider-outage skip used by live client integration tests."""

import pytest

from shallweswim.clients.base import BaseClientError, RetryableClientError
from tests.helpers import skip_on_provider_outage


@pytest.mark.parametrize(
    "error",
    [
        RetryableClientError("Request timed out after 30.0s"),
        RetryableClientError("HTTP error 503", status=503),
    ],
)
def test_outage_skips(error: RetryableClientError) -> None:
    """A network failure or a 5xx after retries skips with the error as reason."""
    with (
        pytest.raises(pytest.skip.Exception, match="Provider outage"),
        skip_on_provider_outage("Provider"),
    ):
        raise error


@pytest.mark.parametrize(
    "error",
    [
        RetryableClientError("HTTP error 429", status=429),
        BaseClientError("HTTP error 404", status=404),
        AssertionError("changed format"),
    ],
)
def test_other_errors_propagate(error: Exception) -> None:
    """Rate limiting, a 4xx, and a failed assertion still fail the test."""
    with pytest.raises(type(error)) as raised, skip_on_provider_outage("Provider"):
        raise error
    assert raised.value is error
