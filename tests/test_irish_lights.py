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
from shallweswim.core.feeds import to_serving_index
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


def test_temperature_payload_converts_to_utc_fahrenheit_and_filters_qc() -> None:
    """MetOcean temperature rows become UTC-indexed Fahrenheit rows."""
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
        min_valid_temp_c=0.0,
        max_valid_temp_c=25.0,
        mmsi="992501100",
    )

    assert list(frame.columns) == ["water_temp"]
    assert frame.index.name == "time"
    assert frame.index.to_list() == [pd.Timestamp("2026-06-15 00:00:00", tz="UTC")]
    assert str(frame.index.tz) == "UTC"
    assert frame["water_temp"].to_list() == pytest.approx([c_to_f(13.0)])


def test_fall_back_hour_keeps_both_folds_as_distinct_instants() -> None:
    """Both folds of an Irish fall-back hour survive as separate readings."""
    # 2025-10-26 01:30 local happens twice in Europe/Dublin: once at 00:30Z on
    # Irish Standard Time and again at 01:30Z on Greenwich Mean Time.
    payload = {
        "MetOceanData": [
            {"hour": "2025-10-26T00:30:00Z", "WaterTemperature": 11.1},
            {"hour": "2025-10-26T01:30:00Z", "WaterTemperature": 11.2},
            {"hour": "2025-10-26T02:30:00Z", "WaterTemperature": 11.3},
        ]
    }
    timezone = pytz.timezone("Europe/Dublin")

    frame = _metocean_temperature_to_feed(
        payload=payload,
        min_valid_temp_c=0.0,
        max_valid_temp_c=25.0,
        mmsi="992501100",
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
    # Serving keeps the first fold, the Irish Standard Time reading.
    assert served.index.to_list() == [
        pd.Timestamp("2025-10-26 01:30:00"),
        pd.Timestamp("2025-10-26 02:30:00"),
    ]
    assert served["water_temp"].iloc[0] == pytest.approx(c_to_f(11.1))


def test_temperature_payload_rejects_endpoint_error_rows() -> None:
    """Irish Lights error rows are treated as provider data errors."""
    payload = {"MetOceanData": [{"Error": "Maximun duration of 30000 hours exceeded"}]}

    with pytest.raises(IrishLightsDataError, match="30000 hours"):
        _metocean_temperature_to_feed(
            payload=payload,
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
            min_valid_temp_c=0.0,
            max_valid_temp_c=25.0,
            mmsi="992501100",
        )
