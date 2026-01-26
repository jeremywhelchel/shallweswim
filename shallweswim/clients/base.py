"""Base class for API clients.

Architecture
------------
All API clients inherit from BaseApiClient and implement `_execute_request`.
The base class provides retry logic via `request_with_retry`.

Implementation Pattern for _execute_request
-------------------------------------------
Subclasses MUST structure `_execute_request` with TWO SEPARATE PHASES:

1. **FETCH PHASE** (inside try/except):
   - Perform the actual API call or library function
   - Catch ONLY network/connection errors here
   - Raise RetryableClientError for transient errors (timeouts, connection refused)
   - Raise client-specific *ApiError for unexpected fetch failures

2. **VALIDATION PHASE** (OUTSIDE try/except):
   - Check if the response indicates "no data" (expected condition)
   - Raise StationUnavailableError for confirmed "no data" conditions
   - Raise client-specific *DataError for unexpected formats

Example structure:
    async def _execute_request(self, ..., location_code: str) -> pd.DataFrame:
        # --- FETCH PHASE ---
        try:
            raw_result = await self._do_api_call(...)
        except NetworkErrors as e:
            raise RetryableClientError(...) from e
        except Exception as e:
            raise MyClientApiError(...) from e

        # --- VALIDATION PHASE (outside try!) ---
        if raw_result is empty:
            self.log(..., level=WARNING)
            raise StationUnavailableError(...)  # Propagates naturally
        if raw_result has unexpected format:
            self.log(..., level=ERROR)
            raise MyClientDataError(...)

        return raw_result

WHY THIS PATTERN MATTERS:
- StationUnavailableError raised in validation phase propagates naturally
- If validation were inside the try block, `except Exception` would catch it
- This caused bugs where expected "no data" was logged as ERROR instead of WARNING
- See CONVENTIONS.md Section 5 for full error handling documentation

Error Hierarchy
---------------
- BaseClientError: Base for all client errors
  - RetryableClientError: Transient errors, will be retried
  - StationUnavailableError: Expected "no data" condition (WARNING, not ERROR)
  - *ApiError (per client): Unexpected errors during fetch
  - *DataError (per client): Unexpected response format
"""

import abc
import logging
import aiohttp
from typing import Optional, Any, TypeVar
import tenacity


class BaseClientError(Exception):
    """Base exception for all client-related errors."""


class ClientConnectionError(BaseClientError):
    """Indicates a connection error after exhausting retries."""


class RetryableClientError(BaseClientError):
    """Exception raised when a client encounters a transient, retryable error."""


class StationUnavailableError(BaseClientError):
    """Station is temporarily unavailable or has no data.

    This is an EXPECTED operational condition, not a bug.
    Use ONLY when we're confident the station is offline or has no data,
    NOT for unexpected data formats or parsing failures.

    Examples:
    - NDBC returns empty dict {}
    - COOPS returns "No data was found" message
    - Empty DataFrame with zero rows for requested time range
    """


# Type variable for the result of the API call
T = TypeVar("T")


class BaseApiClient(abc.ABC):
    """Abstract base class for API clients with built-in retry logic."""

    # Tenacity configuration
    MAX_RETRIES: int = 4  # Total attempts = MAX_RETRIES + 1
    INITIAL_RETRY_DELAY: float = 0.5  # seconds
    MAX_RETRY_DELAY: float = 5.0  # seconds

    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abc.abstractmethod
    def client_type(self) -> str:
        """Return a string identifier for the client type (e.g., 'coops', 'nwis')."""
        pass

    def log(
        self,
        message: str,
        level: int = logging.INFO,
        location_code: Optional[str] = None,
    ) -> None:
        """Log a message with standardized formatting including client type and location."""
        prefix = f"[{location_code or 'general'}][{self.client_type}]"
        formatted_message = f"{prefix} {message}"
        logging.log(level, formatted_message)

    def _log_retry(self, retry_state: tenacity.RetryCallState) -> None:
        """Log retry attempts using tenacity's state and the instance's log method."""
        # Extract location_code from the arguments passed to the wrapped function
        # This assumes 'location_code' is passed as a keyword argument to request_with_retry
        location_code = retry_state.kwargs.get("location_code", "unknown")
        attempt_num = retry_state.attempt_number
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        wait_time = (
            getattr(retry_state.next_action, "sleep", None)
            if retry_state.next_action
            else None
        )
        self.log(
            (
                f"Retryable error encountered (Attempt {attempt_num}/{self.MAX_RETRIES + 1}): "
                f"{exception.__class__.__name__ if exception else 'Unknown'}: {exception if exception else 'Unknown error'}. Retrying in {wait_time:.2f}s..."
                if wait_time
                else "Retrying..."
            ),
            level=logging.WARNING,
            location_code=location_code,
        )

    @abc.abstractmethod
    async def _execute_request(self, *args: Any, **kwargs: Any) -> T:
        """Perform the client-specific API request. See module docstring for pattern.

        IMPORTANT: Structure this method with TWO SEPARATE PHASES:

        1. FETCH PHASE (inside try/except):
           - Perform API call
           - Catch network errors → raise RetryableClientError
           - Catch other fetch errors → raise *ApiError

        2. VALIDATION PHASE (OUTSIDE try/except - this is critical!):
           - Check for "no data" → raise StationUnavailableError (WARNING)
           - Check for bad format → raise *DataError (ERROR)

        The validation phase MUST be outside try/except so that
        StationUnavailableError propagates without being caught by
        a generic `except Exception` handler.

        Args:
            *args: Client-specific positional arguments.
            **kwargs: Must include 'location_code' for logging.

        Returns:
            The successfully retrieved data (typically pd.DataFrame).

        Raises:
            RetryableClientError: Transient network error (will be retried).
            StationUnavailableError: Station has no data (expected, WARNING).
            *ApiError: Unexpected error during fetch (ERROR).
            *DataError: Unexpected response format (ERROR).
        """
        pass

    async def request_with_retry(
        self, location_code: str, *args: Any, **kwargs: Any
    ) -> T:
        """Executes the client-specific request with retry logic using tenacity.

        Args:
            location_code: The location code for logging.
            *args: Positional arguments to pass to _execute_request.
            **kwargs: Keyword arguments to pass to _execute_request.

        Returns:
            The result of the _execute_request on success.

        Raises:
            RetryableClientError: If the request fails after all retries due to transient issues.
            Exception: Any non-retryable exception raised by _execute_request.
        """

        # Define the retry strategy using tenacity
        retry_decorator = tenacity.retry(
            stop=tenacity.stop_after_attempt(self.MAX_RETRIES + 1),
            wait=tenacity.wait_random_exponential(
                multiplier=self.INITIAL_RETRY_DELAY, max=self.MAX_RETRY_DELAY
            ),
            retry=tenacity.retry_if_exception_type(RetryableClientError),
            before_sleep=self._log_retry,
            reraise=True,  # Re-raise the last exception if all retries fail
        )

        # Apply the decorator dynamically to the _execute_request call
        # We need to wrap it in an inner async function to apply the decorator correctly
        # Pass location_code explicitly for the logger
        async def decorated_execute(*inner_args: Any, **inner_kwargs: Any) -> T:
            return await self._execute_request(*inner_args, **inner_kwargs)

        # Add location_code to kwargs if not already present, for the logger
        if "location_code" not in kwargs:
            kwargs["location_code"] = location_code

        try:
            # Call the decorated execution function
            result: T = await retry_decorator(decorated_execute)(*args, **kwargs)
            return result
        except RetryableClientError as e:
            # Log final failure after all retries
            self.log(
                f"API request failed after {self.MAX_RETRIES + 1} attempts due to retryable errors: "
                f"{e.__class__.__name__}: {e}",
                level=logging.ERROR,
                location_code=location_code,
            )
            raise  # Re-raise the final RetryableClientError
        except StationUnavailableError:
            # Expected operational condition - already logged as WARNING in _execute_request
            # Just propagate to _handle_task_exception which logs at WARNING level
            raise
        except Exception as e:
            # Log non-retryable failures immediately
            self.log(
                f"API request failed due to a non-retryable error: {e.__class__.__name__}: {e}",
                level=logging.ERROR,
                location_code=location_code,
            )
            raise  # Re-raise other exceptions

    # TODO: Define common methods and properties here, e.g.,
    # - Common utility functions (e.g., date formatting) -> COULD BE ADDED LATER
