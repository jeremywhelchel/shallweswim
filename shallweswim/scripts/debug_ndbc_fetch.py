"""Debug NDBC temperature fetches without running the full service."""

import argparse
import asyncio
import datetime
import time
from dataclasses import dataclass

import aiohttp
import pandas as pd

from shallweswim import config
from shallweswim.clients.ndbc import NdbcApi
from shallweswim.config.locations import NdbcTempFeedConfig
from shallweswim.util import utc_now


@dataclass(frozen=True)
class FetchTarget:
    station: str
    timezone: str
    mode: str
    label: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NDBC temperature data through shallweswim.clients.ndbc."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("station", nargs="?", help="NDBC station id, such as 44013.")
    target.add_argument(
        "--location",
        help="Location code with an NDBC temperature source, such as bos or chi.",
    )
    parser.add_argument("--timezone", help="IANA timezone for station mode.")
    parser.add_argument(
        "--mode",
        choices=["stdmet", "ocean"],
        default="stdmet",
        help="NDBC data mode to fetch.",
    )
    parser.add_argument("--start", type=_date, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", type=_date, help="End date, YYYY-MM-DD.")
    parser.add_argument("--start-year", type=int, help="First year for yearly mode.")
    parser.add_argument("--end-year", type=int, help="Last year for yearly mode.")
    parser.add_argument(
        "--yearly",
        action="store_true",
        help="Split the requested range into one fetch per year.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum concurrent yearly fetch tasks in this script.",
    )
    return parser.parse_args()


def _date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def _target(args: argparse.Namespace) -> FetchTarget:
    if args.location:
        location_config = config.get(args.location)
        temp_source = location_config.temp_source
        if not isinstance(temp_source, NdbcTempFeedConfig):
            raise ValueError(f"{args.location} does not use an NDBC temperature source")
        return FetchTarget(
            station=temp_source.station,
            timezone=str(location_config.timezone),
            mode=args.mode,
            label=args.location,
        )

    if not args.timezone:
        raise ValueError("--timezone is required when fetching by station")
    return FetchTarget(
        station=args.station,
        timezone=args.timezone,
        mode=args.mode,
        label=args.station,
    )


def _range(args: argparse.Namespace) -> tuple[datetime.date, datetime.date]:
    today = utc_now().date()
    if args.start_year or args.end_year:
        start_year = args.start_year or args.end_year
        end_year = args.end_year or args.start_year
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


async def _fetch_range(
    *,
    client: NdbcApi,
    target: FetchTarget,
    start: datetime.date,
    end: datetime.date,
    label: str,
) -> pd.DataFrame:
    started = time.monotonic()
    df = await client.temperature(
        station_id=target.station,
        begin_date=start,
        end_date=end,
        timezone=target.timezone,
        location_code=target.label,
        mode=target.mode,
    )
    elapsed = time.monotonic() - started
    oldest = df.index.min() if not df.empty else None
    newest = df.index.max() if not df.empty else None
    missing = int(df["water_temp"].isna().sum()) if "water_temp" in df else 0
    print(
        f"{label}: rows={len(df)} missing={missing} "
        f"oldest={oldest} newest={newest} elapsed={elapsed:.2f}s"
    )
    return df


async def _main() -> None:
    args = _parse_args()
    target = _target(args)
    start, end = _range(args)

    print(
        f"target={target.label} station={target.station} mode={target.mode} "
        f"timezone={target.timezone} start={start} end={end}"
    )

    async with aiohttp.ClientSession() as session:
        client = NdbcApi(session=session)

        if not args.yearly:
            await _fetch_range(
                client=client,
                target=target,
                start=start,
                end=end,
                label=f"{start}:{end}",
            )
            return

        semaphore = asyncio.Semaphore(args.concurrency)

        async def fetch_year(
            year: int, year_start: datetime.date, year_end: datetime.date
        ) -> pd.DataFrame:
            async with semaphore:
                return await _fetch_range(
                    client=client,
                    target=target,
                    start=year_start,
                    end=year_end,
                    label=str(year),
                )

        tasks = [
            fetch_year(year, year_start, year_end)
            for year, year_start, year_end in _year_ranges(start, end)
        ]
        dataframes = await asyncio.gather(*tasks)
        combined = pd.concat(dataframes).sort_index().resample("h").first()
        missing = int(combined["water_temp"].isna().sum())
        print(
            f"combined: rows={len(combined)} missing={missing} "
            f"oldest={combined.index.min()} newest={combined.index.max()}"
        )


if __name__ == "__main__":
    asyncio.run(_main())
