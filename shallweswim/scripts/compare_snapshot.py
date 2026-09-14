"""Compare bundle-backed serving with a locally fetched legacy manager.

This is the shadow-mode comparison command, and it never runs in production.
One local process holds both sides in memory exactly as a web instance would:
it builds the fetching `LocationDataManager` of every enabled location the way
the web service does, loads the current published generation from
`SHALLWESWIM_SNAPSHOT_READ_BUCKET` into a `SnapshotLocationManager` per
location, and asks both sides the same questions at one location-local instant.

The command answers whether bundle-backed serving is equivalent to today's
serving. It exits zero only when every configured feed matches (or is present
on neither side, or present only in the bundle because the local fetch failed),
so it can run in a loop for days and its exit status is the verdict.

Usage:
    SHALLWESWIM_SNAPSHOT_READ_BUCKET=my-bucket \\
      uv run python -m shallweswim.scripts.compare_snapshot
    ... compare_snapshot --location nyc --location san
    ... compare_snapshot --at 2026-06-01T04:00:00
"""

import argparse
import asyncio
import dataclasses
import datetime
import logging
import math
import os
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from enum import StrEnum

import aiohttp
import numpy as np
import pandas as pd

from shallweswim import config as config_lib
from shallweswim import logging_utils
from shallweswim.archive.store import gcs_store

# The job's serving cycle, reused rather than restated: the legacy side must be
# the state the publisher itself would have published from.
from shallweswim.capture import serve_location
from shallweswim.clients import create_api_clients
from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.core.manager import PLOT_HARD_TIMEOUT, LocationDataManager
from shallweswim.core.serving import LocationServing
from shallweswim.snapshot.load import LoadedSnapshot, load_current
from shallweswim.snapshot.manager import SnapshotLocationManager
from shallweswim.snapshot.model import LocationManifest
from shallweswim.snapshot.store import SNAPSHOT_READ_BUCKET_ENV_VAR, SnapshotStore
from shallweswim.types import DataSourceType

# Archive variables the comparison run must not act on; see `_isolate_legacy`.
ARCHIVE_BUCKET_ENV_VAR = "SHALLWESWIM_ARCHIVE_BUCKET"
ARCHIVE_READ_BUCKET_ENV_VAR = "SHALLWESWIM_ARCHIVE_READ_BUCKET"

# Float columns agree within this relative tolerance; the comparison adds no
# absolute tolerance, because both sides read the same numbers from the same
# provider and any absolute floor would hide a real difference near zero.
FLOAT_RTOL = 1e-6

# A mismatch report is a diagnosis, not a dump: enough rows to see the pattern.
MAX_REPORTED_DIFFERENCES = 20


class FeedOutcome(StrEnum):
    """What comparing one configured feed's two sides found.

    `ABSENT` is not one of the contract's five outcomes: it is the case the
    contract's exit rule names, a configured feed that neither side holds, and
    naming it keeps `missing` meaning what the contract says it means.
    """

    MISSING = "missing"
    EXTRA = "extra"
    DISJOINT = "disjoint"
    MISMATCH = "mismatch"
    MATCH = "match"
    ABSENT = "absent"


# Outcomes that do not fail the run: equal data, data only the bundle has
# (the local fetch failed), and a feed neither side holds.
PASSING_OUTCOMES = frozenset({FeedOutcome.MATCH, FeedOutcome.EXTRA, FeedOutcome.ABSENT})


@dataclasses.dataclass(frozen=True)
class ValueDifference:
    """One differing cell on a shared timestamp, or one differing column."""

    timestamp: datetime.datetime | None
    column: str
    legacy: object
    bundle: object


@dataclasses.dataclass(frozen=True)
class DerivedDifference:
    """One route-facing answer the two sides give differently."""

    answer: str
    legacy: str
    bundle: str


@dataclasses.dataclass(frozen=True)
class FeedComparison:
    """One configured feed's outcome and the numbers behind it."""

    feed: FeedName
    outcome: FeedOutcome
    shared_count: int
    differing_count: int
    largest_difference: float | None
    legacy_first: datetime.datetime | None
    legacy_last: datetime.datetime | None
    bundle_first: datetime.datetime | None
    bundle_last: datetime.datetime | None
    lag: datetime.timedelta | None
    differences: tuple[ValueDifference, ...] = ()
    derived_differences: tuple[DerivedDifference, ...] = ()


@dataclasses.dataclass(frozen=True)
class LocationComparison:
    """Every configured feed of one location, compared at one local instant."""

    code: str
    at: datetime.datetime
    feeds: tuple[FeedComparison, ...]
    legacy_plots: tuple[PlotName, ...]
    bundle_plots: tuple[PlotName, ...]

    @property
    def ok(self) -> bool:
        """Whether every feed's outcome passes."""
        return all(feed.outcome in PASSING_OUTCOMES for feed in self.feeds)


@dataclasses.dataclass(frozen=True)
class _Raised:
    """A derived answer that raised; equal to another of the same failure."""

    name: str
    message: str

    def __str__(self) -> str:
        return f"{self.name}: {self.message}"


@dataclasses.dataclass(frozen=True)
class _FrameDifference:
    """What comparing two frames on their shared timestamps found."""

    shared_count: int
    differing_count: int
    largest_difference: float | None
    differences: tuple[ValueDifference, ...]


def is_configured(config: config_lib.LocationConfig, feed_name: FeedName) -> bool:
    """Whether the location configures the feed, as `build_feeds` decides it.

    Args:
        config: The location's configuration.
        feed_name: The feed to test.

    Returns:
        True when the location's configuration builds this feed.

    Raises:
        ValueError: If the feed name is not one this function knows.
    """
    if feed_name is FeedName.TIDES:
        return config.tide_source is not None
    if feed_name is FeedName.CURRENTS:
        return config.currents_source is not None
    if feed_name is FeedName.LIVE_TEMPS:
        live = config.live_temp_source
        return live is not None and live.live_enabled
    if feed_name is FeedName.HISTORIC_TEMPS:
        historic = config.historic_temp_source
        return historic is not None and historic.historic_enabled
    raise ValueError(f"Unhandled feed: {feed_name}")


def _window(
    frame: pd.DataFrame,
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """Return the frame's first and last timestamp, or Nones when it is empty."""
    if frame.empty:
        return (None, None)
    index = frame.index
    return (index[0].to_pydatetime(), index[-1].to_pydatetime())


def _scalar(value: object) -> object:
    """Return a numpy scalar as the plain Python value the report prints."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _is_float_column(series: pd.Series) -> bool:
    """Whether a column is compared with the float tolerance."""
    return bool(pd.api.types.is_float_dtype(series))


def _compare_frames(legacy: pd.DataFrame, bundle: pd.DataFrame) -> _FrameDifference:
    """Compare two served frames on the timestamps they share.

    Float columns agree within `FLOAT_RTOL` relative tolerance and treat two
    NaNs as equal; every other column must agree exactly. A column only one
    side has is itself a difference, reported with no timestamp.

    Args:
        legacy: The locally fetched frame.
        bundle: The frame loaded from the published generation.

    Returns:
        The shared timestamp count, how many of those timestamps differ in any
        column, the largest absolute difference across numeric columns, and the
        first `MAX_REPORTED_DIFFERENCES` differences.
    """
    shared = legacy.index.intersection(bundle.index)
    left = legacy.loc[shared]
    right = bundle.loc[shared]
    differing = np.zeros(len(shared), dtype=bool)
    largest: float | None = None
    differences: list[ValueDifference] = []

    for column in legacy.columns:
        if column not in bundle.columns:
            differences.append(
                ValueDifference(None, str(column), "<column>", "<absent>")
            )
            continue
        left_column = left[column]
        right_column = right[column]
        if _is_float_column(left_column) and _is_float_column(right_column):
            left_values = left_column.to_numpy(dtype=float)
            right_values = right_column.to_numpy(dtype=float)
            equal = np.isclose(
                left_values, right_values, rtol=FLOAT_RTOL, atol=0.0, equal_nan=True
            )
            absolute = np.abs(left_values - right_values)
            finite = absolute[np.isfinite(absolute)]
            if finite.size:
                column_largest = float(finite.max())
                largest = (
                    column_largest if largest is None else max(largest, column_largest)
                )
        else:
            left_values = left_column.astype(object).to_numpy()
            right_values = right_column.astype(object).to_numpy()
            both_null = left_column.isna().to_numpy() & right_column.isna().to_numpy()
            equal = (left_values == right_values) | both_null
        column_differs = ~equal
        differing = differing | column_differs
        for position in np.flatnonzero(column_differs):
            if len(differences) >= MAX_REPORTED_DIFFERENCES:
                break
            differences.append(
                ValueDifference(
                    shared[position].to_pydatetime(),
                    str(column),
                    _scalar(left_values[position]),
                    _scalar(right_values[position]),
                )
            )

    for column in bundle.columns:
        if column not in legacy.columns:
            differences.append(
                ValueDifference(None, str(column), "<absent>", "<column>")
            )

    return _FrameDifference(
        shared_count=len(shared),
        differing_count=int(differing.sum()),
        largest_difference=largest,
        differences=tuple(differences[:MAX_REPORTED_DIFFERENCES]),
    )


def _answers_equal(left: object, right: object) -> bool:
    """Whether two derived answers agree, floats within `FLOAT_RTOL`.

    Answers are dataclasses of scalars, enums, datetimes, and nested
    dataclasses. Two floats that differ only in their last bits, which the same
    arithmetic over frames that round-trip through Parquet can produce, are the
    same answer; everything else must be equal exactly.
    """
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=FLOAT_RTOL, abs_tol=0.0) or (
            math.isnan(left) and math.isnan(right)
        )
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        if type(left) is not type(right):
            return False
        return all(
            _answers_equal(getattr(left, field.name), getattr(right, field.name))
            for field in dataclasses.fields(left)
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _answers_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _answers_equal(left[key], right[key]) for key in left
        )
    return left == right


def _answer(
    serving: LocationServing, call: Callable[[LocationServing], object]
) -> object:
    """Return one derived answer, or the failure it raised.

    A raised answer compares equal to the same failure on the other side, so a
    question neither side can answer is not reported as a difference.
    """
    try:
        return call(serving)
    except Exception as error:
        # The failure is the answer: a route would return it to a user.
        return _Raised(type(error).__name__, str(error))


def _derived_checks(
    config: config_lib.LocationConfig,
    feed_name: FeedName,
    now_local: datetime.datetime,
    legacy_frame: pd.DataFrame,
    bundle_frame: pd.DataFrame,
) -> list[tuple[str, Callable[[LocationServing], object]]]:
    """Return the route-facing answers to compare for one feed.

    Args:
        config: The location's configuration.
        feed_name: The feed being compared.
        now_local: The naive location-local instant both sides answer at.
        legacy_frame: The locally fetched frame.
        bundle_frame: The frame loaded from the published generation.

    Returns:
        Named callables, each asked of both sides. Live temperature is asked
        only when both frames end at the same timestamp, because the latest
        reading is by definition a different answer when they do not.
    """
    if feed_name is FeedName.TIDES and config.tide_source is not None:
        return [
            ("get_tide_info_at_time", lambda s: s.get_tide_info_at_time(now_local)),
            ("predict_tide_at_time", lambda s: s.predict_tide_at_time(now_local)),
        ]
    currents = config.currents_source
    if (
        feed_name is FeedName.CURRENTS
        and currents is not None
        and currents.source_type == DataSourceType.PREDICTION
    ):
        return [("predict_flow_at_time", lambda s: s.predict_flow_at_time(now_local))]
    if (
        feed_name is FeedName.LIVE_TEMPS
        and _window(legacy_frame)[1] == (_window(bundle_frame)[1])
    ):
        return [("get_current_temperature", lambda s: s.get_current_temperature())]
    return []


def _compare_feed(
    legacy: LocationServing,
    bundle: LocationServing,
    config: config_lib.LocationConfig,
    feed_name: FeedName,
    now_local: datetime.datetime,
) -> FeedComparison:
    """Compare one configured feed on both sides at `now_local`."""
    legacy_has = legacy.has_feed_data(feed_name)
    bundle_has = bundle.has_feed_data(feed_name)
    legacy_frame = legacy.get_feed_values(feed_name) if legacy_has else pd.DataFrame()
    bundle_frame = bundle.get_feed_values(feed_name) if bundle_has else pd.DataFrame()
    legacy_first, legacy_last = _window(legacy_frame)
    bundle_first, bundle_last = _window(bundle_frame)

    if not (legacy_has and bundle_has):
        if not legacy_has and not bundle_has:
            outcome = FeedOutcome.ABSENT
        elif legacy_has:
            outcome = FeedOutcome.MISSING
        else:
            outcome = FeedOutcome.EXTRA
        return FeedComparison(
            feed=feed_name,
            outcome=outcome,
            shared_count=0,
            differing_count=0,
            largest_difference=None,
            legacy_first=legacy_first,
            legacy_last=legacy_last,
            bundle_first=bundle_first,
            bundle_last=bundle_last,
            lag=None,
        )

    frame_difference = _compare_frames(legacy_frame, bundle_frame)
    derived: list[DerivedDifference] = []
    for name, call in _derived_checks(
        config, feed_name, now_local, legacy_frame, bundle_frame
    ):
        legacy_answer = _answer(legacy, call)
        bundle_answer = _answer(bundle, call)
        if not _answers_equal(legacy_answer, bundle_answer):
            derived.append(
                DerivedDifference(name, str(legacy_answer), str(bundle_answer))
            )

    if frame_difference.differing_count or frame_difference.differences or derived:
        # A derived answer can differ while the overlap is empty or equal, and
        # that is still a mismatch: it is what a request would return.
        outcome = FeedOutcome.MISMATCH
    elif frame_difference.shared_count == 0:
        outcome = FeedOutcome.DISJOINT
    else:
        outcome = FeedOutcome.MATCH

    lag = (
        None
        if legacy_last is None or bundle_last is None
        else legacy_last - bundle_last
    )
    return FeedComparison(
        feed=feed_name,
        outcome=outcome,
        shared_count=frame_difference.shared_count,
        differing_count=frame_difference.differing_count,
        largest_difference=frame_difference.largest_difference,
        legacy_first=legacy_first,
        legacy_last=legacy_last,
        bundle_first=bundle_first,
        bundle_last=bundle_last,
        lag=lag,
        differences=frame_difference.differences,
        derived_differences=tuple(derived),
    )


def compare_location(
    legacy: LocationServing,
    bundle: LocationServing,
    config: config_lib.LocationConfig,
    now_local: datetime.datetime,
) -> LocationComparison:
    """Compare one location's two serving states at one local instant.

    Every feed the configuration builds is compared, plus any feed either side
    holds data for that the configuration no longer builds. Plot bytes are not
    compared, because the two sides draw different fetch windows; only which
    plots exist on each side is reported.

    Args:
        legacy: The locally fetched serving state.
        bundle: The serving state loaded from the published generation.
        config: The location's configuration.
        now_local: The naive location-local instant both sides answer at.

    Returns:
        One outcome per compared feed, and each side's plot names.

    Raises:
        ValueError: If `now_local` is timezone-aware; served frames and every
            query are naive location-local.
    """
    if now_local.tzinfo is not None:
        raise ValueError("now_local must be naive location-local time")
    compared = [
        _compare_feed(legacy, bundle, config, feed_name, now_local)
        for feed_name in FeedName
        if is_configured(config, feed_name)
        or legacy.has_feed_data(feed_name)
        or bundle.has_feed_data(feed_name)
    ]
    return LocationComparison(
        code=config.code,
        at=now_local,
        feeds=tuple(compared),
        legacy_plots=tuple(
            name for name in PlotName if legacy.get_plot(name) is not None
        ),
        bundle_plots=tuple(
            name for name in PlotName if bundle.get_plot(name) is not None
        ),
    )


def exit_code(comparisons: Sequence[LocationComparison]) -> int:
    """Return 0 when every feed of every location passes, 1 otherwise."""
    return 0 if all(comparison.ok for comparison in comparisons) else 1


def _format_time(value: datetime.datetime | None) -> str:
    return "-" if value is None else value.strftime("%Y-%m-%d %H:%M")


def _format_row(feed: FeedComparison) -> str:
    """Return one feed's table row."""
    largest = (
        "-" if feed.largest_difference is None else f"{feed.largest_difference:.3e}"
    )
    return (
        f"  {feed.feed.value:<15}{feed.outcome.value:<10}"
        f"{feed.shared_count:>8}{feed.differing_count:>8}{largest:>12}  "
        f"{_format_time(feed.legacy_first)}..{_format_time(feed.legacy_last)}  "
        f"{_format_time(feed.bundle_first)}..{_format_time(feed.bundle_last)}  "
        f"{'-' if feed.lag is None else feed.lag}"
    )


def _format_details(comparison: LocationComparison, feed: FeedComparison) -> list[str]:
    """Return the differing rows and derived answers of one mismatching feed."""
    lines: list[str] = []
    if feed.differences:
        lines.append(
            f"{comparison.code}/{feed.feed.value}: {feed.differing_count} of "
            f"{feed.shared_count} shared timestamps differ "
            f"(showing {len(feed.differences)})"
        )
        lines.extend(
            f"  {_format_time(difference.timestamp)}  {difference.column}  "
            f"legacy={difference.legacy!r}  bundle={difference.bundle!r}"
            for difference in feed.differences
        )
    for derived in feed.derived_differences:
        lines.append(f"{comparison.code}/{feed.feed.value}: {derived.answer} differs")
        lines.append(f"  legacy: {derived.legacy}")
        lines.append(f"  bundle: {derived.bundle}")
    return lines


def format_report(comparisons: Sequence[LocationComparison]) -> str:
    """Render the readable report: one table per location, then differing rows.

    Args:
        comparisons: One comparison per location, in report order.

    Returns:
        The report text, ending with the verdict line the exit code follows.
    """
    lines: list[str] = []
    details: list[str] = []
    for comparison in comparisons:
        lines.append(
            f"{comparison.code} at {comparison.at.isoformat(sep=' ')} local "
            f"({'ok' if comparison.ok else 'FAILED'})"
        )
        lines.append(
            f"  {'feed':<15}{'outcome':<10}{'shared':>8}{'differ':>8}{'max diff':>12}  "
            f"{'legacy window':<36}{'bundle window':<36}lag"
        )
        lines.extend(_format_row(feed) for feed in comparison.feeds)
        lines.append(
            f"  plots: legacy={_plot_list(comparison.legacy_plots)} "
            f"bundle={_plot_list(comparison.bundle_plots)}"
        )
        for feed in comparison.feeds:
            details.extend(_format_details(comparison, feed))
        lines.append("")
    lines.extend(details)
    if details:
        lines.append("")
    failing = sum(1 for comparison in comparisons if not comparison.ok)
    lines.append(
        f"{len(comparisons)} location(s) compared, {failing} failing: "
        f"{'MATCH' if failing == 0 else 'MISMATCH'}"
    )
    return "\n".join(lines)


def _plot_list(plots: Sequence[PlotName]) -> str:
    return ",".join(plot.value for plot in plots) if plots else "-"


def _isolate_legacy() -> None:
    """Make the legacy side a pure provider fetch that writes nothing.

    The comparison exists to prove the bundle equals what the providers return
    right now, so the historical feed must not hydrate past years from the
    archive: `SHALLWESWIM_ARCHIVE_READ_BUCKET` is removed from this process's
    environment even when the operator's `.env` sets it for ordinary local
    runs. The write bucket is removed too, because a comparison run must never
    add to the archive.
    """
    os.environ.pop(ARCHIVE_READ_BUCKET_ENV_VAR, None)
    os.environ.pop(ARCHIVE_BUCKET_ENV_VAR, None)


async def _fetch_legacy(
    configs: Sequence[config_lib.LocationConfig],
) -> dict[str, LocationDataManager]:
    """Build and run one fetching manager per location, as the job does.

    Args:
        configs: The locations to fetch.

    Returns:
        The managers, keyed by location code, after their serving cycle and
        their plots have completed.
    """
    pool = ProcessPoolExecutor(max_workers=os.cpu_count())
    try:
        async with aiohttp.ClientSession() as session:
            clients = create_api_clients(session)
            managers = [
                LocationDataManager(location_config, clients, pool)
                for location_config in configs
            ]
            await asyncio.gather(*(serve_location(manager) for manager in managers))
            await asyncio.gather(
                *(manager.wait_for_plots(PLOT_HARD_TIMEOUT) for manager in managers)
            )
    finally:
        pool.shutdown(wait=True)
    return {manager.config.code: manager for manager in managers}


def _bundle_manager(
    config: config_lib.LocationConfig,
    loaded: LoadedSnapshot,
    loaded_at: datetime.datetime,
) -> SnapshotLocationManager:
    """Return the location's serving state from the loaded generation.

    A location the generation does not hold becomes an empty manager, so every
    one of its feeds is reported `missing` rather than the location vanishing
    from the report.
    """
    location = loaded.manifest.locations.get(config.code)
    if location is None:
        return SnapshotLocationManager(
            config, LocationManifest(feeds={}, plots={}), {}, {}, loaded_at
        )
    return SnapshotLocationManager(
        config,
        location,
        loaded.frames[config.code],
        loaded.plots[config.code],
        loaded_at,
    )


async def _run(
    configs: Sequence[config_lib.LocationConfig],
    bucket: str,
    at: datetime.datetime | None,
) -> int:
    """Fetch, load, compare, and print.

    Args:
        configs: The locations to compare.
        bucket: The bucket whose current generation is the bundle side.
        at: The naive local instant to compare at, or None for each location's
            current local time.

    Returns:
        The process exit code.
    """
    _isolate_legacy()
    legacy = await _fetch_legacy(configs)
    store = SnapshotStore(await asyncio.to_thread(gcs_store, bucket))
    loaded = await load_current(store)
    if loaded is None:
        logging.error(f"No published generation in {bucket}")
        return 1
    loaded_at = datetime.datetime.now(datetime.UTC)
    comparisons = [
        compare_location(
            legacy[location_config.code],
            _bundle_manager(location_config, loaded, loaded_at),
            location_config,
            at or location_config.local_now(),
        )
        for location_config in configs
    ]
    print(
        f"generation {loaded.pointer.generation_id} "
        f"published {loaded.manifest.published_at.isoformat()}"
    )
    print(format_report(comparisons))
    return exit_code(comparisons)


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, str]:
    """Parse arguments and the required bucket, failing usage before any fetch.

    Args:
        argv: Argument list to parse, or None to read from the command line.

    Returns:
        The parsed arguments and the snapshot read bucket name.

    Raises:
        SystemExit: On any usage error, including a missing read bucket and an
            `--at` value that is not a naive local timestamp.
    """
    parser = argparse.ArgumentParser(
        prog="python -m shallweswim.scripts.compare_snapshot",
        description=(
            "Compare the published snapshot bundle with a locally fetched "
            "legacy manager. Exits non-zero when any feed mismatches."
        ),
    )
    parser.add_argument(
        "--location",
        action="append",
        choices=sorted(config_lib.CONFIGS),
        help="Compare only this location; repeatable (default: every location).",
    )
    parser.add_argument(
        "--at",
        help=(
            "Naive ISO local timestamp to compare at, applied as each "
            "location's own local time (default: now)."
        ),
    )
    args = parser.parse_args(argv)
    bucket = os.environ.get(SNAPSHOT_READ_BUCKET_ENV_VAR)
    if not bucket:
        parser.error(
            f"{SNAPSHOT_READ_BUCKET_ENV_VAR} is required; it names the bucket "
            "holding the published generation to compare against"
        )
    if args.at is not None:
        try:
            args.at = datetime.datetime.fromisoformat(args.at)
        except ValueError as error:
            parser.error(f"--at is not an ISO timestamp: {error}")
        if args.at.tzinfo is not None:
            parser.error("--at must be a naive local timestamp, without an offset")
    return args, str(bucket)


def main(argv: list[str] | None = None) -> int:
    """Run one comparison.

    Args:
        argv: Argument list to parse, or None to read from the command line.

    Returns:
        Process exit code: 0 when every feed matches, is bundle-only, or is
        held by neither side; 1 otherwise.
    """
    args, bucket = _parse_args(argv)
    logging_utils.setup_logging()
    configs = [
        config_lib.CONFIGS[code]
        for code in (args.location or sorted(config_lib.CONFIGS))
    ]
    return asyncio.run(_run(configs, bucket, args.at))


if __name__ == "__main__":
    sys.exit(main())
