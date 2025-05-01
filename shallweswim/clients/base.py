"""Base class for API clients."""

import abc
import logging
import aiohttp


class BaseClientError(Exception):
    """Base exception for all API client errors."""


class BaseApiClient(abc.ABC):
    """Abstract base class for API clients."""

    _session: aiohttp.ClientSession

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Initialize the base client with an aiohttp session.

        Args:
            session: The aiohttp client session to use for requests.
        """
        self._session = session

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log a message using the class's logger.

        Subclasses should ideally implement a more specific logger,
        perhaps incorporating a location code or client name.
        """
        # Basic implementation - subclasses might override
        logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        logger.log(level, message)

    # TODO: Define common methods and properties here, e.g.,
    # - Shared request logic (e.g., _make_request)
    # - Standardized logging methods
