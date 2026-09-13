"""Tests for CSPF Sandettie data client."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pandas as pd
import pytest
import pytz

from shallweswim.clients.base import StationUnavailableError
from shallweswim.clients.cspf import CspfApi, _CspfPage
from shallweswim.core.feeds import to_serving_index
from shallweswim.util import c_to_f


@pytest.fixture
def mock_session() -> MagicMock:
    """Provides a mock aiohttp ClientSession."""
    return MagicMock(spec=aiohttp.ClientSession)


@pytest.fixture
def cspf_client(mock_session: MagicMock) -> CspfApi:
    """Provides a CSPF client with a mock session."""
    return CspfApi(session=mock_session)


def cspf_page(points: list[tuple[str, float]]) -> _CspfPage:
    """Build a minimal CSPF page with the sea-temperature JS array."""
    body = ",".join(f"['{timestamp}', {value}]" for timestamp, value in points)
    return _CspfPage(
        url="https://cspf.co.uk/sandettie-data/2025",
        body=f"var a_abc123 =  [{body}];",
    )


def test_parse_temperature_page_converts_celsius_to_internal_fahrenheit() -> None:
    """CSPF sea-temperature points parse into the standard water_temp schema."""
    frame = CspfApi._parse_temperature_page(
        cspf_page(
            [
                ("1744657200000", 10.1),
                ("1744660800000", 10.2),
            ]
        )
    )

    assert list(frame.columns) == ["water_temp"]
    assert frame.index.name == "time"
    assert str(frame.index.tz) == "UTC"
    # The epoch milliseconds name 19:00Z, which is 20:00 British Summer Time.
    assert frame.index[0] == pd.Timestamp("2025-04-14 19:00:00", tz="UTC")
    assert frame["water_temp"].iloc[0] == pytest.approx(c_to_f(10.1))
    assert frame["water_temp"].iloc[1] == pytest.approx(c_to_f(10.2))


@pytest.mark.asyncio
async def test_monthly_pages_are_primary(
    cspf_client: CspfApi,
) -> None:
    """Monthly pages are denser than annual summaries and are fetched first."""
    monthly_page = cspf_page(
        [
            ("1744657200000", 10.1),
            ("1744660800000", 10.2),
        ]
    )
    empty_page = _CspfPage(url="https://cspf.co.uk/sandettie-data/2025/1", body="")

    with patch.object(
        cspf_client,
        "_fetch_page",
        new_callable=AsyncMock,
        side_effect=[monthly_page] + [empty_page for _ in range(11)],
    ) as fetch_page:
        frame = await cspf_client.sandettie_temperature(
            begin_date=datetime.datetime(2025, 1, 1),
            end_date=datetime.datetime(2025, 12, 31, 23, 59, 59),
            location_code="dov",
            timezone=pytz.timezone("Europe/London"),
        )

    assert [call.kwargs["path"] for call in fetch_page.call_args_list] == [
        f"sandettie-data/2025/{month}" for month in range(1, 13)
    ]
    assert len(frame) == 2
    assert str(frame.index.tz) == "UTC"
    assert frame["water_temp"].iloc[-1] == pytest.approx(c_to_f(10.2))


@pytest.mark.asyncio
async def test_fall_back_hour_keeps_both_folds_as_distinct_instants(
    cspf_client: CspfApi,
) -> None:
    """Both folds of a British fall-back hour survive as separate readings."""
    # 2025-10-26 01:30 local happens twice in Europe/London: once at 00:30Z on
    # British Summer Time and again at 01:30Z on Greenwich Mean Time.
    monthly_page = cspf_page(
        [
            ("1761438600000", 11.1),
            ("1761442200000", 11.2),
            ("1761445800000", 11.3),
        ]
    )
    empty_page = _CspfPage(url="https://cspf.co.uk/sandettie-data/2025/1", body="")
    timezone = pytz.timezone("Europe/London")

    with patch.object(
        cspf_client,
        "_fetch_page",
        new_callable=AsyncMock,
        side_effect=[empty_page for _ in range(9)]
        + [monthly_page]
        + [empty_page for _ in range(2)],
    ):
        frame = await cspf_client.sandettie_temperature(
            begin_date=datetime.datetime(2025, 1, 1),
            end_date=datetime.datetime(2025, 12, 31, 23, 59, 59),
            location_code="dov",
            timezone=timezone,
        )

    assert frame.index.is_unique
    assert frame.index.to_list() == [
        pd.Timestamp("2025-10-26 00:30:00", tz="UTC"),
        pd.Timestamp("2025-10-26 01:30:00", tz="UTC"),
        pd.Timestamp("2025-10-26 02:30:00", tz="UTC"),
    ]

    served = to_serving_index(frame, timezone)
    assert served.index.tz is None
    assert served.index.is_unique
    # Serving keeps the first fold, the British Summer Time reading.
    assert served.index.to_list() == [
        pd.Timestamp("2025-10-26 01:30:00"),
        pd.Timestamp("2025-10-26 02:30:00"),
    ]
    assert served["water_temp"].iloc[0] == pytest.approx(c_to_f(11.1))


@pytest.mark.asyncio
async def test_empty_monthly_pages_fall_back_to_annual_page(
    cspf_client: CspfApi,
) -> None:
    """Annual pages remain a fallback when monthly pages have no data."""
    annual_page = cspf_page([("1744657200000", 10.1)])
    empty_page = _CspfPage(url="https://cspf.co.uk/sandettie-data/2025/1", body="")

    with patch.object(
        cspf_client,
        "_fetch_page",
        new_callable=AsyncMock,
        side_effect=[empty_page for _ in range(12)] + [annual_page],
    ) as fetch_page:
        frame = await cspf_client.sandettie_temperature(
            begin_date=datetime.datetime(2025, 1, 1),
            end_date=datetime.datetime(2025, 12, 31, 23, 59, 59),
            location_code="dov",
            timezone=pytz.timezone("Europe/London"),
        )

    assert fetch_page.call_args_list[-1].kwargs["path"] == "sandettie-data/2025"
    assert fetch_page.call_count == 13
    assert len(frame) == 1
    assert str(frame.index.tz) == "UTC"
    assert frame["water_temp"].iloc[0] == pytest.approx(c_to_f(10.1))


@pytest.mark.asyncio
async def test_empty_year_raises_station_unavailable(cspf_client: CspfApi) -> None:
    """No parsed temperature points is a station-unavailable condition."""
    with patch.object(
        cspf_client,
        "_fetch_page",
        new_callable=AsyncMock,
        return_value=_CspfPage(url="https://cspf.co.uk/sandettie-data/2023", body=""),
    ):
        with pytest.raises(StationUnavailableError):
            await cspf_client.sandettie_temperature(
                begin_date=datetime.datetime(2023, 1, 1),
                end_date=datetime.datetime(2023, 12, 31, 23, 59, 59),
                location_code="dov",
                timezone=pytz.timezone("Europe/London"),
            )
