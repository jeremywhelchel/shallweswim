"""Tests for the Irish Lights MetOcean API client."""

import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
import pytz

from shallweswim.clients.base import StationUnavailableError
from shallweswim.clients.irish_lights import (
    IRISH_LIGHTS_METOCEAN_URL,
    IrishLightsApi,
    IrishLightsDataError,
    _metocean_temperature_to_feed,
)
from shallweswim.util import c_to_f


def test_build_metocean_url_uses_public_token_mmsi_and_time_bounds() -> None:
    """Irish Lights URLs include the required MMSI and bounded date params."""
    client = IrishLightsApi(session=MagicMock())

    url = client._build_metocean_url(
        mmsi="992501100",
        begin_utc=datetime.datetime(2026, 6, 15, tzinfo=datetime.UTC),
        end_utc=datetime.datetime(2026, 6, 16, tzinfo=datetime.UTC),
    )

    assert url.startswith(IRISH_LIGHTS_METOCEAN_URL)
    assert "accesstoken=" in url
    assert "MMSI=992501100" in url
    assert "FromDate=2026-06-15T00%3A00%3A00.000Z" in url
    assert "ToDate=2026-06-16T00%3A00%3A00.000Z" in url


def test_temperature_payload_converts_to_local_naive_fahrenheit_and_filters_qc() -> (
    None
):
    """MetOcean temperature rows become local-naive Fahrenheit rows."""
    payload = {
        "MetOceanData": [
            {
                "hour": "2026-06-15T01:00:00Z",
                "WaterTemperature": 50.1,
            },
            {
                "hour": "2026-06-15T00:00:00Z",
                "WaterTemperature": 13.0,
            },
            {
                "hour": "2026-06-14T23:00:00Z",
                "WaterTemperature": None,
            },
        ]
    }

    frame = _metocean_temperature_to_feed(
        payload=payload,
        timezone=pytz.timezone("Europe/Dublin"),
        min_valid_temp_c=0.0,
        max_valid_temp_c=25.0,
        mmsi="992501100",
    )

    assert list(frame.columns) == ["water_temp"]
    assert frame.index.name == "time"
    assert frame.index.to_list() == [pd.Timestamp("2026-06-15 01:00:00")]
    assert frame.index.tz is None
    assert frame["water_temp"].to_list() == pytest.approx([c_to_f(13.0)])


def test_temperature_payload_rejects_endpoint_error_rows() -> None:
    """Irish Lights error rows are treated as provider data errors."""
    payload = {"MetOceanData": [{"Error": "Maximun duration of 30000 hours exceeded"}]}

    with pytest.raises(IrishLightsDataError, match="30000 hours"):
        _metocean_temperature_to_feed(
            payload=payload,
            timezone=pytz.UTC,
            min_valid_temp_c=0.0,
            max_valid_temp_c=25.0,
            mmsi="992501100",
        )


def test_temperature_payload_requires_expected_columns() -> None:
    """Irish Lights schema drift fails as a data error, not no-data."""
    payload = {"MetOceanData": [{"hour": "2026-06-15T00:00:00Z"}]}

    with pytest.raises(IrishLightsDataError, match="WaterTemperature"):
        _metocean_temperature_to_feed(
            payload=payload,
            timezone=pytz.UTC,
            min_valid_temp_c=0.0,
            max_valid_temp_c=25.0,
            mmsi="992501100",
        )


def test_temperature_payload_requires_usable_temperature_rows() -> None:
    """Empty or fully filtered MetOcean payloads are station-unavailable."""
    payload = {
        "MetOceanData": [
            {"hour": "2026-06-15T00:00:00Z", "WaterTemperature": 50.1},
        ]
    }

    with pytest.raises(StationUnavailableError, match="no usable"):
        _metocean_temperature_to_feed(
            payload=payload,
            timezone=pytz.UTC,
            min_valid_temp_c=0.0,
            max_valid_temp_c=25.0,
            mmsi="992501100",
        )
