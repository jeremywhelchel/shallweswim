"""Base class for API clients."""

import abc
import logging


class BaseApiClient(abc.ABC):
    """Abstract base class for API clients."""

    # TODO: Define common methods and properties here, e.g.,
    # - __init__ with shared session/config
    # - Shared request logic (e.g., _make_request)
    # - Standardized logging methods

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log a message using the class's logger.

        Subclasses should ideally implement a more specific logger,
        perhaps incorporating a location code or client name.
        """
        # Basic implementation - subclasses might override
        logger = logging.getLogger(self.__class__.__name__)
        logger.log(level, message)
