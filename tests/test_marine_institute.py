"""Tests for the Marine Institute Ireland API client."""

import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytz

from shallweswim.clients.base import StationUnavailableError
from shallweswim.clients.marine_institute import (
    MARINE_INSTITUTE_TIDE_HIGH_LOW_URL,
    METERS_TO_FEET,
    MarineInstituteApi,
    MarineInstituteDataError,
    _high_low_tide_predictions_to_feed,
)


def test_build_tide_high_low_url_encodes_station_and_time_constraints() -> None:
    """Marine Institute high/low URLs use explicit tabledap constraints."""
    client = MarineInstituteApi(session=MagicMock())

    url = client._build_tide_high_low_url(
        station_id="Kinsale",
        begin_utc=datetime.datetime(2026, 6, 14, tzinfo=datetime.UTC),
        end_utc=datetime.datetime(2026, 6, 16, tzinfo=datetime.UTC),
    )

    assert url.startswith(MARINE_INSTITUTE_TIDE_HIGH_LOW_URL)
    assert "time,stationID,longitude,latitude,tide_time_category" in url
    assert "Water_Level_ODMalin" in url
    assert "stationID=%22Kinsale%22" in url
    assert "time%3E=2026-06-14T00:00:00Z" in url
    assert "time%3C=2026-06-16T00:00:00Z" in url


def test_high_low_tide_predictions_convert_units_datum_and_timezone() -> None:
    """OD Malin high/low rows become local-naive LAT-like feet events."""
    raw_df = pd.DataFrame(
        {
            "time": [
                "2026-06-14T04:00:00Z",
                "2026-06-14T10:20:00Z",
            ],
            "Water_Level_ODMalin": [1.564, -1.482],
            "tide_time_category": ["HIGH", "LOW"],
        }
    )

    result = _high_low_tide_predictions_to_feed(
        raw_df=raw_df,
        timezone=pytz.timezone("Europe/Dublin"),
        height_offset_m=2.01,
    )

    assert list(result["type"].astype(str)) == ["high", "low"]
    assert result["prediction"].to_list() == pytest.approx(
        [(1.564 + 2.01) * METERS_TO_FEET, (-1.482 + 2.01) * METERS_TO_FEET]
    )
    assert result.index.to_list() == [
        datetime.datetime(2026, 6, 14, 5, 0),
        datetime.datetime(2026, 6, 14, 11, 20),
    ]
    assert result.index.tz is None


def test_high_low_tide_predictions_require_expected_columns() -> None:
    """Unexpected Marine Institute tide response schemas fail loudly."""
    raw_df = pd.DataFrame({"time": ["2026-06-14T00:00:00Z"]})

    with pytest.raises(MarineInstituteDataError, match="Water_Level_ODMalin"):
        _high_low_tide_predictions_to_feed(
            raw_df=raw_df,
            timezone=pytz.UTC,
            height_offset_m=2.01,
        )


def test_high_low_tide_predictions_require_usable_rows() -> None:
    """Rows without parseable time, height, or tide type are unavailable."""
    raw_df = pd.DataFrame(
        {
            "time": ["not-a-time"],
            "Water_Level_ODMalin": ["not-a-height"],
            "tide_time_category": ["UNKNOWN"],
        }
    )

    with pytest.raises(StationUnavailableError, match="no usable high/low events"):
        _high_low_tide_predictions_to_feed(
            raw_df=raw_df,
            timezone=pytz.UTC,
            height_offset_m=2.01,
        )
