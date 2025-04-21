"""Tests for NOAA API client."""

# Standard library imports
import datetime
import urllib.error
from typing import Literal, cast
from unittest.mock import patch

# Third-party imports
import pandas as pd
import pytest

# Local imports
from shallweswim.noaa import NoaaApi, NoaaConnectionError, NoaaDataError
from shallweswim.noaa_types import ProductType


@pytest.fixture
def mock_tide_data() -> pd.DataFrame:
    """Mock tide prediction data."""
    return pd.DataFrame(
        {
            "Date Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
            " Prediction": [5.2, 1.3],
            " Type": ["H", "L"],
        }
    )


@pytest.fixture
def mock_current_data() -> pd.DataFrame:
    """Mock current prediction data."""
    return pd.DataFrame(
        {
            "Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
            " Velocity_Major": [2.5, -1.8],
            " Depth": [10.0, 10.0],
            " Type": ["flood", "ebb"],
            " meanFloodDir": [45.0, 45.0],
            " Bin": [1, 1],
        }
    )


@pytest.fixture
def mock_temperature_data() -> pd.DataFrame:
    """Mock temperature data."""
    return pd.DataFrame(
        {
            "Date Time": ["2025-04-19 10:00", "2025-04-19 16:00"],
            " Water Temperature": [62.5, 63.2],
            " Air Temperature": [68.0, 70.5],
            " X": [1, 1],
            " N": [1, 1],
            " R ": [1, 1],
        }
    )


def test_tides_success(mock_tide_data: pd.DataFrame) -> None:
    """Test successful tide prediction fetch."""
    with patch("pandas.read_csv", return_value=mock_tide_data):
        df = NoaaApi.Tides(station=9414290)

    assert len(df) == 2
    assert list(df.columns) == ["prediction", "type"]
    assert df["type"].tolist() == ["high", "low"]
    assert df["prediction"].tolist() == [5.2, 1.3]


def test_currents_success(mock_current_data: pd.DataFrame) -> None:
    """Test successful current prediction fetch."""
    with patch("pandas.read_csv", return_value=mock_current_data):
        df = NoaaApi.Currents(station="SFB1201", interpolate=False)

    assert len(df) == 2
    assert list(df.columns) == ["velocity"]
    assert df["velocity"].tolist() == [2.5, -1.8]


def test_temperature_success(mock_temperature_data: pd.DataFrame) -> None:
    """Test successful temperature fetch."""
    with patch("pandas.read_csv", return_value=mock_temperature_data):
        df = NoaaApi.Temperature(
            station=9414290,
            product="water_temperature",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
        )

    assert len(df) == 2
    assert "water_temp" in df.columns
    assert df["water_temp"].tolist() == [62.5, 63.2]


def test_connection_error() -> None:
    """Test handling of connection errors."""
    with patch("pandas.read_csv", side_effect=urllib.error.URLError("Network error")):
        with pytest.raises(NoaaConnectionError, match="Failed to connect to NOAA API"):
            NoaaApi.Tides(station=9414290)


def test_data_error() -> None:
    """Test handling of API data errors."""
    error_df = pd.DataFrame({"Error": ["Invalid station ID"]})
    with patch("pandas.read_csv", return_value=error_df):
        with pytest.raises(NoaaDataError, match="Invalid station ID"):
            NoaaApi.Tides(station=9414290)


def test_invalid_temperature_dates() -> None:
    """Test validation of temperature date ranges."""
    with pytest.raises(ValueError, match="begin_date must be <= end_date"):
        NoaaApi.Temperature(
            station=9414290,
            product="water_temperature",
            begin_date=datetime.date(2025, 4, 20),
            end_date=datetime.date(2025, 4, 19),
        )


def test_invalid_temperature_product() -> None:
    """Test validation of temperature product type."""
    with pytest.raises(ValueError, match="Invalid product"):
        NoaaApi.Temperature(
            station=9414290,
            product="invalid_product",  # type: ignore[arg-type] # intentionally invalid for testing error case
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
        )


def test_current_interpolation(mock_current_data: pd.DataFrame) -> None:
    """Test current interpolation."""
    with patch("pandas.read_csv", return_value=mock_current_data):
        df = NoaaApi.Currents(station="SFB1201", interpolate=True)

    # Should have many more points due to 60s interpolation
    assert len(df) > len(mock_current_data)
    # First and last values should match original
    assert df["velocity"].iloc[0] == mock_current_data[" Velocity_Major"].iloc[0]
    assert df["velocity"].iloc[-1] == mock_current_data[" Velocity_Major"].iloc[-1]
