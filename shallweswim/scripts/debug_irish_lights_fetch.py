"""Debug Irish Lights MetOcean temperature fetches."""

import argparse
import asyncio
import datetime
import time

import aiohttp

from shallweswim import config
from shallweswim.clients.irish_lights import IrishLightsApi
from shallweswim.config.locations import IrishLightsTempFeedConfig
from shallweswim.util import utc_now


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Irish Lights MetOcean temperature data."
    )
    parser.add_argument(
        "--location",
        default="cor",
        help="Location code with an Irish Lights temperature source.",
    )
    parser.add_argument("--start", type=_date, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", type=_date, help="End date, YYYY-MM-DD.")
    parser.add_argument("--start-year", type=int, help="First year to fetch.")
    parser.add_argument("--end-year", type=int, help="Last year to fetch.")
    return parser.parse_args()


def _date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def _year_end(year: int) -> datetime.datetime:
    if year == utc_now().year:
        return utc_now()
    return datetime.datetime(year, 12, 31, 23, 59, 59)


def _ranges(
    args: argparse.Namespace,
) -> list[tuple[str, datetime.datetime, datetime.datetime]]:
    if args.start_year or args.end_year:
        start_year = args.start_year or args.end_year
        end_year = args.end_year or args.start_year
        return [
            (
                str(year),
                datetime.datetime(year, 1, 1),
                _year_end(year),
            )
            for year in range(start_year, end_year + 1)
        ]

    end_date = args.end or utc_now().date()
    start_date = args.start or (end_date - datetime.timedelta(days=2))
    return [
        (
            f"{start_date}:{end_date}",
            datetime.datetime.combine(start_date, datetime.time.min),
            datetime.datetime.combine(end_date, datetime.time.max),
        )
    ]


async def _main() -> None:
    args = _parse_args()
    location_config = config.get(args.location)
    if location_config is None:
        raise ValueError(f"Unknown location: {args.location}")
    temp_source = (
        location_config.live_temp_source or location_config.historic_temp_source
    )
    if not isinstance(temp_source, IrishLightsTempFeedConfig):
        raise ValueError(f"{args.location} does not use an Irish Lights temp source")

    ranges = _ranges(args)
    print(
        f"target={location_config.code}:temperature mmsi={temp_source.mmsi} "
        f"timezone={location_config.timezone} ranges={len(ranges)}"
    )
    started = time.monotonic()
    total_rows = 0
    failures = 0

    async with aiohttp.ClientSession() as session:
        client = IrishLightsApi(session=session)
        for label, start, end in ranges:
            range_started = time.monotonic()
            try:
                df = await client.water_temperature(
                    mmsi=temp_source.mmsi,
                    begin_date=start,
                    end_date=end,
                    timezone=location_config.timezone,
                    location_code=location_config.code,
                    min_valid_temp_c=temp_source.min_valid_temp_c,
                    max_valid_temp_c=temp_source.max_valid_temp_c,
                )
            except Exception as e:
                failures += 1
                print(f"{label}: error={e.__class__.__name__}: {e}")
                continue

            total_rows += len(df)
            print(
                f"{label}: rows={len(df)} oldest={df.index.min()} newest={df.index.max()} "
                f"min_f={df['water_temp'].min():.2f} max_f={df['water_temp'].max():.2f} "
                f"elapsed={time.monotonic() - range_started:.2f}s"
            )

    print(
        f"summary: ranges={len(ranges)} rows={total_rows} failures={failures} "
        f"elapsed={time.monotonic() - started:.2f}s"
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
