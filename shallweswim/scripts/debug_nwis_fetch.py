"""Debug USGS NWIS fetches without running the full service."""

import argparse
import asyncio
import contextvars
import datetime
import time
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
import pandas as pd

from shallweswim import config
from shallweswim.clients.nwis import NwisApi
from shallweswim.config.locations import (
    LocationConfig,
    NwisCurrentFeedConfig,
    NwisTempFeedConfig,
)
from shallweswim.core.manager import DEFAULT_HISTORIC_TEMPS_START_YEAR
from shallweswim.util import utc_now

FeedKind = Literal["live-temp", "historical-temp", "currents"]

_fetch_label: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nwis_debug_fetch_label", default="unlabeled"
)


@dataclass(frozen=True)
class FetchTarget:
    site_no: str
    parameter_cd: str
    timezone: str
    feed: FeedKind
    label: str
    location_code: str
    historic_enabled: bool | None = None
    live_enabled: bool | None = None
    configured_start_year: int | None = None
    configured_end_year: int | None = None


@dataclass(frozen=True)
class HttpRecord:
    label: str
    method: str
    url: str
    status: int | None
    elapsed: float
    rate_limit_limit: str | None
    rate_limit_remaining: str | None
    retry_after: str | None
    error: str | None = None


@dataclass(frozen=True)
class FetchOutcome:
    dataframes: list[pd.DataFrame]
    failures: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch USGS NWIS data through shallweswim.clients.nwis."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "site_no", nargs="?", help="USGS site number, such as 03292494."
    )
    target.add_argument(
        "--location",
        help="Location code with an NWIS temperature or currents source, such as aus or sdf.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Fetch all configured locations that have the selected NWIS feed.",
    )
    parser.add_argument(
        "--feed",
        choices=["live-temp", "historical-temp", "currents", "all"],
        default="live-temp",
        help="NWIS feed to fetch.",
    )
    parser.add_argument("--parameter-cd", help="USGS parameter code for site mode.")
    parser.add_argument("--timezone", help="IANA timezone for site mode.")
    parser.add_argument("--start", type=_date, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", type=_date, help="End date, YYYY-MM-DD.")
    parser.add_argument("--start-year", type=int, help="First year for yearly mode.")
    parser.add_argument("--end-year", type=int, help="Last year for yearly mode.")
    parser.add_argument(
        "--yearly",
        action="store_true",
        help="Split temperature fetches into one fetch per year.",
    )
    parser.add_argument(
        "--startup-workload",
        action="store_true",
        help="Fetch the configured NWIS startup workload for the selected location(s).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum concurrent yearly fetch tasks in this script.",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=5,
        help="Multiplier for estimating multi-instance startup request volume.",
    )
    return parser.parse_args()


def _date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def _target_feeds(args: argparse.Namespace) -> list[FeedKind]:
    if args.startup_workload:
        return ["live-temp", "historical-temp", "currents"]
    if args.feed == "all":
        return ["live-temp", "historical-temp", "currents"]
    return [args.feed]


def _configured_locations(args: argparse.Namespace) -> list[LocationConfig]:
    if args.location:
        location_config = config.get(args.location)
        if location_config is None:
            raise ValueError(f"Unknown location code: {args.location}")
        return [location_config]
    if args.all:
        return list(config.CONFIGS.values())
    raise ValueError("Configured locations are only available with --location or --all")


def _targets(args: argparse.Namespace) -> list[FetchTarget]:
    feeds = _target_feeds(args)
    if args.site_no:
        if len(feeds) != 1:
            raise ValueError("site mode requires one concrete --feed, not all")
        if not args.parameter_cd:
            raise ValueError("--parameter-cd is required when fetching by site")
        if not args.timezone:
            raise ValueError("--timezone is required when fetching by site")
        feed = feeds[0]
        return [
            FetchTarget(
                site_no=args.site_no,
                parameter_cd=args.parameter_cd,
                timezone=args.timezone,
                feed=feed,
                label=f"{args.site_no}:{feed}",
                location_code=args.site_no,
            )
        ]

    targets: list[FetchTarget] = []
    for location_config in _configured_locations(args):
        for feed in feeds:
            target = _target_for_location(location_config, feed, args.startup_workload)
            if target is not None:
                targets.append(target)

    if not targets:
        raise ValueError("No configured NWIS targets matched the requested feed")
    return targets


def _target_for_location(
    location_config: LocationConfig,
    feed: FeedKind,
    startup_workload: bool,
) -> FetchTarget | None:
    if feed in {"live-temp", "historical-temp"}:
        temp_source = location_config.temp_source
        if not isinstance(temp_source, NwisTempFeedConfig):
            return None
        if startup_workload and feed == "live-temp" and not temp_source.live_enabled:
            return None
        if (
            startup_workload
            and feed == "historical-temp"
            and not temp_source.historic_enabled
        ):
            return None
        return FetchTarget(
            site_no=temp_source.site_no,
            parameter_cd=temp_source.parameter_cd,
            timezone=str(location_config.timezone),
            feed=feed,
            label=f"{location_config.code}:{feed}",
            location_code=location_config.code,
            historic_enabled=temp_source.historic_enabled,
            live_enabled=temp_source.live_enabled,
            configured_start_year=temp_source.start_year,
            configured_end_year=temp_source.end_year,
        )

    currents_source = location_config.currents_source
    if not isinstance(currents_source, NwisCurrentFeedConfig):
        return None
    return FetchTarget(
        site_no=currents_source.site_no,
        parameter_cd=currents_source.parameter_cd,
        timezone=str(location_config.timezone),
        feed=feed,
        label=f"{location_config.code}:{feed}",
        location_code=location_config.code,
    )


def _range(
    args: argparse.Namespace, target: FetchTarget
) -> tuple[datetime.date, datetime.date]:
    today = utc_now().date()
    if target.feed == "currents":
        return today - datetime.timedelta(days=1), today

    if target.feed == "historical-temp" or args.start_year or args.end_year:
        start_year = (
            args.start_year
            or target.configured_start_year
            or DEFAULT_HISTORIC_TEMPS_START_YEAR
        )
        end_year = args.end_year or target.configured_end_year or today.year
        return datetime.date(start_year, 1, 1), _year_end(end_year, today)

    end = args.end or today
    start = args.start or (end - datetime.timedelta(days=8))
    return start, end


def _year_end(year: int, today: datetime.date) -> datetime.date:
    if year == today.year:
        return today
    return datetime.date(year, 12, 31)


def _year_ranges(
    start: datetime.date, end: datetime.date
) -> list[tuple[int, datetime.date, datetime.date]]:
    ranges = []
    today = utc_now().date()
    for year in range(start.year, end.year + 1):
        year_start = max(start, datetime.date(year, 1, 1))
        year_end = min(end, _year_end(year, today))
        ranges.append((year, year_start, year_end))
    return ranges


def _trace_config(records: list[HttpRecord]) -> aiohttp.TraceConfig:
    trace_config = aiohttp.TraceConfig()

    async def on_request_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: Any,
        params: aiohttp.TraceRequestStartParams,
    ) -> None:
        del session, params
        trace_config_ctx.started_at = time.monotonic()
        trace_config_ctx.label = _fetch_label.get()

    async def on_request_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: Any,
        params: aiohttp.TraceRequestEndParams,
    ) -> None:
        del session
        started_at = getattr(trace_config_ctx, "started_at", time.monotonic())
        label = getattr(trace_config_ctx, "label", _fetch_label.get())
        headers = params.response.headers
        records.append(
            HttpRecord(
                label=label,
                method=params.method,
                url=str(params.url),
                status=params.response.status,
                elapsed=time.monotonic() - started_at,
                rate_limit_limit=headers.get("x-ratelimit-limit"),
                rate_limit_remaining=headers.get("x-ratelimit-remaining"),
                retry_after=headers.get("retry-after"),
            )
        )

    async def on_request_exception(
        session: aiohttp.ClientSession,
        trace_config_ctx: Any,
        params: aiohttp.TraceRequestExceptionParams,
    ) -> None:
        del session
        started_at = getattr(trace_config_ctx, "started_at", time.monotonic())
        label = getattr(trace_config_ctx, "label", _fetch_label.get())
        records.append(
            HttpRecord(
                label=label,
                method=params.method,
                url=str(params.url),
                status=None,
                elapsed=time.monotonic() - started_at,
                rate_limit_limit=None,
                rate_limit_remaining=None,
                retry_after=None,
                error=f"{params.exception.__class__.__name__}: {params.exception}",
            )
        )

    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_exception.append(on_request_exception)
    return trace_config


async def _fetch_range(
    *,
    client: NwisApi,
    records: list[HttpRecord],
    target: FetchTarget,
    start: datetime.date,
    end: datetime.date,
    label: str,
) -> FetchOutcome:
    started = time.monotonic()
    token = _fetch_label.set(label)
    try:
        if target.feed == "currents":
            df = await client.currents(
                site_no=target.site_no,
                parameter_cd=target.parameter_cd,
                timezone=target.timezone,
                location_code=target.location_code,
            )
        else:
            df = await client.temperature(
                site_no=target.site_no,
                parameter_cd=target.parameter_cd,
                begin_date=start,
                end_date=end,
                timezone=target.timezone,
                location_code=target.location_code,
            )
    except Exception as e:
        elapsed = time.monotonic() - started
        print(f"{label}: ERROR {e.__class__.__name__}: {e} elapsed={elapsed:.2f}s")
        _print_http_summary(label, records)
        return FetchOutcome(dataframes=[], failures=1)
    finally:
        _fetch_label.reset(token)

    elapsed = time.monotonic() - started
    _print_dataframe_summary(label, df, target, elapsed)
    _print_http_summary(label, records)
    return FetchOutcome(dataframes=[df], failures=0)


def _print_dataframe_summary(
    label: str,
    df: pd.DataFrame,
    target: FetchTarget,
    elapsed: float,
) -> None:
    oldest = df.index.min() if not df.empty else None
    newest = df.index.max() if not df.empty else None
    value_column = _value_column(target)
    missing = int(df[value_column].isna().sum()) if value_column in df else 0
    value_min = df[value_column].min() if value_column in df and not df.empty else None
    value_max = df[value_column].max() if value_column in df and not df.empty else None
    print(
        f"{label}: rows={len(df)} missing={missing} oldest={oldest} newest={newest} "
        f"min={value_min} max={value_max} elapsed={elapsed:.2f}s"
    )


def _value_column(target: FetchTarget) -> str:
    if target.feed == "currents":
        return "velocity_fps"
    return "water_temp"


def _print_http_summary(label: str, records: list[HttpRecord]) -> None:
    label_records = [record for record in records if record.label == label]
    latest = label_records[-1] if label_records else None
    statuses = ",".join(
        str(record.status) if record.status is not None else "exception"
        for record in label_records
    )
    errors = "; ".join(record.error for record in label_records if record.error)
    print(
        f"{label}: http_requests={len(label_records)} statuses={statuses or 'none'} "
        f"rate_limit_limit={latest.rate_limit_limit if latest else None} "
        f"rate_limit_remaining={latest.rate_limit_remaining if latest else None} "
        f"retry_after={latest.retry_after if latest else None} "
        f"errors={errors or 'none'}"
    )


async def _fetch_target(
    *,
    client: NwisApi,
    records: list[HttpRecord],
    target: FetchTarget,
    args: argparse.Namespace,
) -> FetchOutcome:
    start, end = _range(args, target)
    print(
        f"target={target.label} site={target.site_no} parameter={target.parameter_cd} "
        f"timezone={target.timezone} feed={target.feed} start={start} end={end} "
        f"live_enabled={target.live_enabled} historic_enabled={target.historic_enabled}"
    )

    if target.feed == "currents":
        return await _fetch_range(
            client=client,
            records=records,
            target=target,
            start=start,
            end=end,
            label=target.label,
        )

    yearly = args.yearly or args.startup_workload or target.feed == "historical-temp"
    if not yearly:
        return await _fetch_range(
            client=client,
            records=records,
            target=target,
            start=start,
            end=end,
            label=f"{target.label}:{start}:{end}",
        )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def fetch_year(
        year: int, year_start: datetime.date, year_end: datetime.date
    ) -> FetchOutcome:
        async with semaphore:
            return await _fetch_range(
                client=client,
                records=records,
                target=target,
                start=year_start,
                end=year_end,
                label=f"{target.label}:{year}",
            )

    outcomes = await asyncio.gather(
        *[
            fetch_year(year, year_start, year_end)
            for year, year_start, year_end in _year_ranges(start, end)
        ]
    )
    successful_dataframes = [df for outcome in outcomes for df in outcome.dataframes]
    failures = sum(outcome.failures for outcome in outcomes)
    if not successful_dataframes:
        print(f"{target.label}: combined rows=0 successful_years=0")
        return FetchOutcome(dataframes=[], failures=failures)

    combined = pd.concat(successful_dataframes).sort_index().resample("h").first()
    value_column = _value_column(target)
    missing = (
        int(combined[value_column].isna().sum()) if value_column in combined else 0
    )
    print(
        f"{target.label}: combined rows={len(combined)} missing={missing} "
        f"oldest={combined.index.min()} newest={combined.index.max()} "
        f"successful_years={len(successful_dataframes)} failures={failures}"
    )
    return FetchOutcome(dataframes=successful_dataframes, failures=failures)


def _print_final_summary(
    *,
    records: list[HttpRecord],
    instance_count: int,
    started: float,
) -> None:
    elapsed = time.monotonic() - started
    http_failures = [
        record for record in records if record.status is None or record.status >= 400
    ]
    latest = records[-1] if records else None
    print(
        "summary: "
        f"http_requests={len(records)} "
        f"estimated_{instance_count}_instance_requests={len(records) * instance_count} "
        f"http_failures={len(http_failures)} elapsed={elapsed:.2f}s "
        f"latest_rate_limit_limit={latest.rate_limit_limit if latest else None} "
        f"latest_rate_limit_remaining={latest.rate_limit_remaining if latest else None} "
        f"latest_retry_after={latest.retry_after if latest else None}"
    )


async def _main() -> None:
    args = _parse_args()
    targets = _targets(args)
    records: list[HttpRecord] = []
    started = time.monotonic()

    async with aiohttp.ClientSession(trace_configs=[_trace_config(records)]) as session:
        client = NwisApi(session=session)
        failures = 0
        for target in targets:
            outcome = await _fetch_target(
                client=client,
                records=records,
                target=target,
                args=args,
            )
            failures += outcome.failures

    _print_final_summary(
        records=records,
        instance_count=args.instance_count,
        started=started,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_main())
