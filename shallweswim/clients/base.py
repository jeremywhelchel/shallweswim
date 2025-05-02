"""Base class for API clients."""

import abc
import logging
import aiohttp
from typing import Optional


class BaseClientError(Exception):
    """Base exception for all API client errors."""


class BaseApiClient(abc.ABC):
    """Abstract base class for API clients."""

    _session: aiohttp.ClientSession

    @property
    @abc.abstractmethod
    def client_type(self) -> str:
        """Return the string identifier for the client type (e.g., 'coops', 'ndbc')."""
        raise NotImplementedError

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Initialize the base client with an aiohttp session.

        Args:
            session: The aiohttp client session to use for requests.
        """
        self._session = session

    def log(
        self,
        message: str,
        level: int = logging.INFO,
        location_code: Optional[str] = None,
    ) -> None:
        """Log a message, automatically prepending client type and optional location code."""
        client_tag = self.client_type
        if location_code:
            prefix = f"[{location_code}][{client_tag}]"
        else:
            prefix = f"[{client_tag}]"
        formatted_message = f"{prefix} {message}"
        logging.log(level, formatted_message)

    # TODO: Define common methods and properties here, e.g.,
    # - Shared request logic (e.g., _make_request)
    # - Standardized logging methods
