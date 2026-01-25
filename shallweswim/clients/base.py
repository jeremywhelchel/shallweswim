"""Base class for API clients."""

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
        """Subclasses must implement this method to perform the actual client-specific request.

        This method should:
        1. Perform the necessary API call or library function execution.
        2. Catch specific, transient, retryable errors (e.g., connection errors, timeouts)
           and raise RetryableClientError from them.
        3. Handle non-retryable errors (e.g., bad HTTP status, data parsing errors)
           by raising appropriate subclass-specific exceptions (e.g., CoopsDataError) or
           letting other unexpected errors propagate.
        4. Return the successfully retrieved and parsed data on success.
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
