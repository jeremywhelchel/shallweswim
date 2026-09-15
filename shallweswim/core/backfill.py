"""Walk a location's historical temperature source back through the years.

An ordinary capture run archives the current year. This module is the job-side
walk that archives everything a provider still holds: for one location it
fetches a single station-local calendar year at a time, newest first, archives
whatever comes back, and stops once the provider has answered with no data for
several years running. The walk therefore discovers a source's depth itself
rather than taking a range an operator found by hand.

``shallweswim.update`` hosts the walk: it parses the arguments, runs the
selected locations concurrently, and logs the run summary. A backfill only
writes: it publishes nothing, reads nothing back from the archive, and keeps no
fetched data in memory beyond the request that archived it.
"""

import calendar
import dataclasses
import datetime
import logging
from enum import StrEnum

from shallweswim import config as config_lib
from shallweswim.archive.capture import CaptureResult
from shallweswim.clients.base import BaseApiClient, StationUnavailableError
from shallweswim.core.feeds import TempFeed, create_temp_feed
from shallweswim.util import utc_now

# Floor of the walk when ``--backfill-from`` names no year. No provider in this
# project holds anything nearly this old, so a walk from this floor ends by
# running out of data rather than by reaching the floor.
BACKFILL_FLOOR_YEAR = 1900

# Consecutive empty years, going backwards, that end a source's walk. Provider
# records have short gaps; this is the bound on how long a walk keeps asking
# past one before deciding the record has ended.
BACKFILL_EMPTY_YEARS_STOP = 5


class BackfillStop(StrEnum):
    """Why one source's walk ended."""

    EMPTY_YEARS = "empty years"
    FLOOR = "floor"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class SourceBackfillResult:
    """What one source's walk archived, and why it ended.

    Attributes:
        location_code: Code of the location the source belongs to.
        source_identity: The source's stable citation key, as archived.
        years_archived: Rows added and revised, keyed by the year that
            returned data.
        years_empty: Years every request of which returned no data.
        stop: What ended the walk.
    """

    location_code: str
    source_identity: str
    years_archived: dict[int, CaptureResult]
    years_empty: tuple[int, ...]
    stop: BackfillStop

    @property
    def earliest_year_with_data(self) -> int | None:
        """The earliest year that returned data, or None when none did."""
        return min(self.years_archived) if self.years_archived else None

    @property
    def archived(self) -> CaptureResult:
        """Rows this walk added to and revised in the archive."""
        return CaptureResult(
            sum(result.new_count for result in self.years_archived.values()),
            sum(result.revised_count for result in self.years_archived.values()),
        )

    def summary(self) -> str:
        """Return this walk's one-line description for the run summary."""
        archived = _year_list(sorted(self.years_archived))
        empty = _year_list(sorted(self.years_empty))
        earliest = self.earliest_year_with_data
        return (
            f"[{self.location_code}] {self.source_identity}: "
            f"archived {archived}; empty {empty}; "
            f"earliest {earliest if earliest is not None else 'none'}; "
            f"stopped at {self.stop.value}"
        )


def _year_list(years: list[int]) -> str:
    """Render a list of years for a log message, naming an empty list."""
    return ", ".join(str(year) for year in years) if years else "none"


@dataclasses.dataclass(frozen=True)
class _YearRequest:
    """One provider request in a year's walk, with a label for its logs."""

    label: str
    feed: TempFeed


def _window(
    year: int, month: int | None, now: datetime.datetime
) -> tuple[datetime.datetime, datetime.datetime] | None:
    """Return the timezone-naive edges of one station-local calendar window.

    The edges are built exactly as ``HistoricalTempsFeed._get_feeds`` builds a
    year's: naive local midnight through the last second of the period, capped
    at now so no request asks a provider for the future.

    Args:
        year: Calendar year the window covers.
        month: Calendar month the window covers, or None for the whole year.
        now: Current time, as the naive UTC clock feeds keep.

    Returns:
        The window's start and end, or None when the window lies entirely in
        the future and there is nothing to request.
    """
    start = datetime.datetime(year, month or 1, 1)
    if start > now:
        return None
    if month is None:
        end = datetime.datetime(year, 12, 31, 23, 59, 59)
    else:
        last_day = calendar.monthrange(year, month)[1]
        end = datetime.datetime(year, month, last_day, 23, 59, 59)
    return start, min(end, now)


def _year_requests(
    location_config: config_lib.LocationConfig,
    temp_config: config_lib.TempFeedConfig,
    clients: dict[str, BaseApiClient],
    year: int,
    now: datetime.datetime,
) -> list[_YearRequest]:
    """Build the feeds that archive one year of one source.

    CO-OPS is fetched as both of its products, because neither covers the
    other: the hourly product reaches further back and is complete across
    stretches where six-minute is empty. The six-minute product is requested
    one calendar month at a time rather than as a year, because a year spans
    more than one client request window and an empty window aborts the whole
    call. Every other source is one request per year at the cadence its client
    returns, which is what the historical feed relies on today.

    The feeds are the one-year feeds ``HistoricalTempsFeed._get_feeds`` builds,
    with no expiration interval: nothing schedules or caches them, and each is
    fetched exactly once.

    Args:
        location_config: Location the source belongs to.
        temp_config: The location's historical temperature source.
        clients: Provider API clients keyed by provider name.
        year: Calendar year to archive.
        now: Current time, as the naive UTC clock feeds keep.

    Returns:
        The year's requests in fetch order, hourly first, empty when the whole
        year is still in the future.
    """
    year_window = _window(year, None, now)
    if year_window is None:
        return []
    requests = [
        _YearRequest(
            label=f"{year} hourly",
            feed=create_temp_feed(
                location_config=location_config,
                temp_config=temp_config,
                start=year_window[0],
                end=year_window[1],
                interval="h",
                expiration_interval=None,
                clients=clients,
            ),
        )
    ]
    if isinstance(temp_config, config_lib.CoopsTempFeedConfig):
        for month in range(1, 13):
            month_window = _window(year, month, now)
            if month_window is None:
                continue
            requests.append(
                _YearRequest(
                    label=f"{year}-{month:02d} six-minute",
                    feed=create_temp_feed(
                        location_config=location_config,
                        temp_config=temp_config,
                        start=month_window[0],
                        end=month_window[1],
                        interval="6-min",
                        expiration_interval=None,
                        clients=clients,
                    ),
                )
            )
    return requests


async def _archive_year(
    requests: list[_YearRequest],
    clients: dict[str, BaseApiClient],
    location_code: str,
) -> tuple[CaptureResult, int]:
    """Fetch and archive one year's requests, one at a time.

    A request the provider answers with no data is counted and skipped: a
    six-minute month a station lacks while its hourly year has data is the
    ordinary case. Any other failure propagates, because it ends this source's
    walk rather than one of its years.

    Args:
        requests: The year's requests, in fetch order.
        clients: Provider API clients keyed by provider name.
        location_code: Location code for log context.

    Returns:
        The rows this year added to and revised in the archive, and how many of
        its requests returned no data.
    """
    new_count = 0
    revised_count = 0
    empty_requests = 0
    for request in requests:
        try:
            captured = await request.feed.archive_once(clients)
        except StationUnavailableError as error:
            empty_requests += 1
            logging.debug(f"[{location_code}] No data for {request.label}: {error}")
            continue
        if captured is None:
            continue
        new_count += captured.new_count
        revised_count += captured.revised_count
    return CaptureResult(new_count, revised_count), empty_requests


def _year_message(
    location_code: str,
    source_identity: str,
    year: int,
    captured: CaptureResult,
    empty_requests: int,
    request_count: int,
) -> str:
    """Return the progress line for one completed year of a walk."""
    if empty_requests == request_count:
        detail = "empty"
    else:
        detail = f"{captured.new_count} rows added, {captured.revised_count} revised"
        if request_count > 1:
            detail += f" ({empty_requests} of {request_count} requests empty)"
    return f"[{location_code}] Backfill {source_identity} {year}: {detail}"


async def backfill_location(
    location_config: config_lib.LocationConfig,
    clients: dict[str, BaseApiClient],
    *,
    floor_year: int,
) -> SourceBackfillResult | None:
    """Archive every year the location's historical temperature source holds.

    Years are walked from the current UTC year down to the floor, newest
    first, one request at a time. A year is empty only when every one of its
    requests returned no data; after `BACKFILL_EMPTY_YEARS_STOP` consecutive
    empty years the walk stops, and a year with data resets that count. Any
    other failure ends this source's walk and is logged at ERROR, leaving the
    years already archived archived; the caller's other locations continue.

    Args:
        location_config: Location to walk.
        clients: Provider API clients keyed by provider name.
        floor_year: Oldest year the walk will request.

    Returns:
        What this source's walk archived, or None when the location has no
        historical temperature source to walk.
    """
    temp_config = location_config.historic_temp_source
    if temp_config is None or not temp_config.historic_enabled:
        logging.info(
            f"[{location_config.code}] No historical temperature source to backfill"
        )
        return None

    source_identity = temp_config.citation_key
    # One clock for the whole walk, so its newest year and the windows that
    # year is capped to cannot disagree.
    now = utc_now()
    years_archived: dict[int, CaptureResult] = {}
    years_empty: list[int] = []
    consecutive_empty = 0
    stop = BackfillStop.FLOOR
    for year in range(now.year, floor_year - 1, -1):
        requests = _year_requests(location_config, temp_config, clients, year, now)
        if not requests:
            # The whole year is still in the future; nothing to ask for.
            continue
        try:
            captured, empty_requests = await _archive_year(
                requests, clients, location_config.code
            )
        except Exception as error:
            logging.error(
                f"[{location_config.code}] Backfill of {source_identity} failed "
                f"in {year}: {error}"
            )
            stop = BackfillStop.ERROR
            break
        logging.info(
            _year_message(
                location_config.code,
                source_identity,
                year,
                captured,
                empty_requests,
                len(requests),
            )
        )
        if empty_requests == len(requests):
            years_empty.append(year)
            consecutive_empty += 1
            if consecutive_empty >= BACKFILL_EMPTY_YEARS_STOP:
                stop = BackfillStop.EMPTY_YEARS
                break
            continue
        consecutive_empty = 0
        years_archived[year] = captured

    return SourceBackfillResult(
        location_code=location_config.code,
        source_identity=source_identity,
        years_archived=years_archived,
        years_empty=tuple(years_empty),
        stop=stop,
    )
