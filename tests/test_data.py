"""Tests for data.py functionality."""

# Standard library imports
import datetime
from unittest.mock import MagicMock

# Third-party imports
import pandas as pd
import pytest
from typing import Any

# Local imports
from shallweswim.data import Data
from shallweswim.types import CurrentInfo  # Import from types module where it's defined
from shallweswim import config as config_lib


@pytest.fixture
def mock_config() -> Any:
    """Mock location config fixture."""
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    config.code = "nyc"
    config.StationName = "New York City"
    return config


@pytest.fixture
def mock_current_data() -> pd.DataFrame:
    """Create mock current data with peaks and transitions."""
    # Create a datetime index spanning 24 hours with 1-hour increments
    index = pd.date_range(
        start=datetime.datetime(2025, 4, 22, 0, 0, 0),
        end=datetime.datetime(2025, 4, 22, 23, 0, 0),
        freq="h",
    )

    # Generate a sinusoidal pattern for current velocity
    # Positive values = flood, negative values = ebb
    velocity = [
        1.0,
        1.5,
        1.2,
        0.6,
        0.1,
        -0.3,
        -0.8,
        -1.3,
        -1.5,
        -1.3,
        -0.8,
        -0.3,
        0.2,
        0.7,
        1.2,
        1.5,
        1.3,
        0.7,
        0.1,
        -0.4,
        -0.9,
        -1.3,
        -1.2,
        -0.6,
    ]

    # Create the DataFrame with velocity data
    df = pd.DataFrame({"velocity": velocity}, index=index)
    return df


@pytest.fixture
def mock_data_with_currents(mock_config: Any, mock_current_data: pd.DataFrame) -> Data:
    """Create a Data instance with mock current data."""
    data = Data(mock_config)
    data.currents = mock_current_data
    return data


def test_current_prediction_at_flood_peak(mock_data_with_currents: Data) -> None:
    """Test current prediction at a flood peak."""
    # Test at 3:00 PM (15:00) which is a flood peak at 1.5 knots
    t = datetime.datetime(2025, 4, 22, 15, 0, 0)
    result = mock_data_with_currents.current_prediction(t)

    # Check the result
    assert result.direction == "flooding"
    assert pytest.approx(result.magnitude, 0.1) == 1.5
    assert "at its strongest" in result.state_description


def test_current_prediction_at_ebb_peak() -> None:
    """Test current prediction at an ebb peak."""
    # Create a custom test instance
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    data = Data(config)

    # Create a more comprehensive dataset for testing ebb currents
    # Create a single day with a complete cycle
    test_day = datetime.datetime(2025, 4, 22)
    hours = list(range(24))
    idx = pd.DatetimeIndex([test_day.replace(hour=h) for h in hours])

    # Create an ebb pattern with clear strengthening and weakening periods
    velocities = [
        0.1,  # 0:00 - Near slack
        0.5,  # 1:00 - Flooding
        0.9,  # 2:00 - Flooding
        1.2,  # 3:00 - Peak flood
        0.8,  # 4:00 - Weakening flood
        0.2,  # 5:00 - Near slack
        -0.3,  # 6:00 - Beginning to ebb
        -0.8,  # 7:00 - Strengthening ebb
        -1.3,  # 8:00 - Strengthening ebb
        -1.8,  # 9:00 - Peak ebb current (strongest)
        -1.5,  # 10:00 - Weakening
        -1.0,  # 11:00 - Weakening
        -0.6,  # 12:00 - Weakening
        -0.1,  # 13:00 - Near slack (clearly below 0.2 threshold)
        0.2,  # 14:00 - Beginning to flood
        0.7,  # 15:00 - Flooding
        1.1,  # 16:00 - Flooding
        1.5,  # 17:00 - Peak flood
        1.1,  # 18:00 - Weakening
        0.5,  # 19:00 - Weakening
        0.1,  # 20:00 - Near slack
        -0.4,  # 21:00 - Beginning to ebb
        -0.9,  # 22:00 - Ebbing
        -1.3,  # 23:00 - Ebbing
    ]

    # Create the DataFrame with the full tidal cycle
    current_df = pd.DataFrame({"velocity": velocities}, index=idx)
    data.currents = current_df

    # Test at peak ebb (9:00)
    peak_time = test_day.replace(hour=9)
    result = data.current_prediction(peak_time)

    # Verify the prediction at peak
    assert result.direction == "ebbing"
    assert pytest.approx(result.magnitude, abs=0.1) == 1.8  # Should be about 1.8 knots
    assert "at its strongest" in result.state_description

    # Test during strengthening ebb (8:00)
    strengthening_time = test_day.replace(hour=8)
    result = data.current_prediction(strengthening_time)
    assert result.direction == "ebbing"
    assert "getting stronger" in result.state_description

    # Test during weakening ebb (11:00)
    weakening_time = test_day.replace(hour=11)
    result = data.current_prediction(weakening_time)
    assert result.direction == "ebbing"
    assert "getting weaker" in result.state_description

    # Test near slack after ebb (13:00)
    slack_time = test_day.replace(hour=13)
    result = data.current_prediction(slack_time)
    assert "weakest" in result.state_description


def test_current_prediction_at_slack(mock_data_with_currents: Data) -> None:
    """Test current prediction at slack water."""
    # Test at 4:00 AM (4:00) which is near zero (0.1 knots)
    t = datetime.datetime(2025, 4, 22, 4, 0, 0)
    result = mock_data_with_currents.current_prediction(t)

    # Check the result
    assert result.magnitude < 0.2
    assert "at its weakest (slack)" in result.state_description


def test_current_prediction_strengthening() -> None:
    """Test current prediction when current is strengthening."""
    # Create a custom test instance with clear strengthening pattern
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    data = Data(config)

    # Create data with a clear strengthening pattern
    hours = [12, 13, 14]
    idx = pd.DatetimeIndex([datetime.datetime(2025, 4, 22, h, 0, 0) for h in hours])
    # Clearly increasing velocity
    velocities = [0.5, 0.8, 1.2]

    # Create the DataFrame with strengthening pattern
    df = pd.DataFrame({"velocity": velocities}, index=idx)
    data.currents = df

    # Test at the middle point where slope is positive
    t = datetime.datetime(2025, 4, 22, 13, 0, 0)
    result = data.current_prediction(t)

    # Check the result
    assert result.direction == "flooding"
    # Should be either strengthening or at strongest
    assert any(
        msg in result.state_description
        for msg in ["getting stronger", "at its strongest"]
    )


def test_current_prediction_weakening() -> None:
    """Test current prediction when current is weakening."""
    # Create a custom test instance with clear weakening pattern
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)
    data = Data(config)

    # Create data with a clear weakening pattern
    hours = [15, 16, 17]
    idx = pd.DatetimeIndex([datetime.datetime(2025, 4, 22, h, 0, 0) for h in hours])
    # Clearly decreasing velocity
    velocities = [1.2, 0.8, 0.5]

    # Create the DataFrame with weakening pattern
    df = pd.DataFrame({"velocity": velocities}, index=idx)
    data.currents = df

    # Test at the middle point where slope is negative
    t = datetime.datetime(2025, 4, 22, 16, 0, 0)
    result = data.current_prediction(t)

    # Check the result
    assert result.direction == "flooding"
    assert "getting weaker" in result.state_description


def test_process_peaks_function() -> None:
    """Test that peaks are identified properly in the CurrentPrediction method."""
    # Create a custom data instance with a clear flood peak pattern
    config = MagicMock(spec=config_lib.LocationConfig)
    config.local_now.return_value = datetime.datetime(2025, 4, 22, 12, 0, 0)

    data = Data(config)

    # Create a very distinct peak pattern
    index = pd.date_range(start="2025-04-22", periods=5, freq="h")
    currents_df = pd.DataFrame(
        {
            "velocity": [0.5, 1.5, 0.8, 0.3, 0.1],  # Clear peak at index 1
        },
        index=index,
    )

    # We'll add a slope column to help with trend detection
    currents_df["slope"] = currents_df["velocity"].diff().fillna(0)
    data.currents = currents_df

    # Use a time point that's exactly at the peak
    peak_time = index[1]
    result = data.current_prediction(peak_time)

    # The peak should be detected and described correctly
    assert result.direction == "flooding"
    assert pytest.approx(result.magnitude, abs=0.1) == 1.5

    # Check if it's marked as a strong current
    # We're looking for either "at its strongest" or "getting stronger/weaker"
    assert (
        "at its strongest" in result.state_description
        or "getting stronger" in result.state_description
        or "getting weaker" in result.state_description
    )


def test_current_info_representation() -> None:
    """Test the string representation of CurrentInfo."""
    info = CurrentInfo(
        direction="flooding",
        magnitude=1.5,
        magnitude_pct=0.8,
        state_description="at its strongest",
    )

    # Test the string representation
    representation = str(info)
    assert "flooding" in representation
    assert "1.5" in representation
    assert "at its strongest" in representation
