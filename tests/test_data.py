"""Tests for data.py functionality."""

# Standard library imports
import asyncio
import datetime
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import pytest_asyncio
import pytz

# Third-party imports
from freezegun import freeze_time
from pytest_mock import MockerFixture

# Local imports
from shallweswim import config as config_lib
from shallweswim import feeds
from shallweswim.api_types import FeedStatus

# Import API client classes
from shallweswim.clients.base import BaseApiClient
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.ndbc import NdbcApi
from shallweswim.clients.nwis import NwisApi
from shallweswim.core import queries
from shallweswim.core.queries import DataUnavailableError, get_current_tide_info
from shallweswim.data import LocationDataManager
from shallweswim.feeds import Feed
from shallweswim.types import (
    CurrentDirection,
    CurrentInfo,
    CurrentPhase,
    CurrentStrength,
    CurrentTrend,
    DataSourceType,
    TideTrend,
)

# Local helpers
from tests.helpers import assert_json_serializable


# Helper function to create a LocationConfig for tests
def create_test_location_config(
    code: str = "tst",
    live_temps_enabled: bool = True,
    historic_temps_enabled: bool = True,
) -> config_lib.LocationConfig:
    """Create a LocationConfig object with all necessary attributes for testing.

    Args:
        code: The location code (must be exactly 3 characters)
        live_temps_enabled: Whether live temperature feeds are enabled
        historic_temps_enabled: Whether historic temperature feeds are enabled
    """
    return config_lib.LocationConfig(
        code=code,
        name="Test Location",
        description="Test location for tests",
        latitude=40.7128,
        longitude=-74.0060,
        timezone=pytz.timezone("America/New_York"),
        swim_location="Test Beach",
        swim_location_link="https://example.com/test-beach",
        # Add empty config objects for the data sources
        tide_source=config_lib.CoopsTideFeedConfig(station=8518750, name="The Battery"),
        currents_source=config_lib.CoopsCurrentsFeedConfig(
            stations=["NYH1905"],
        ),
        temp_source=config_lib.CoopsTempFeedConfig(
            station=8518750,
            name="The Battery",
            live_enabled=live_temps_enabled,
            historic_enabled=historic_temps_enabled,
        ),
    )


@pytest.fixture
def mock_config() -> config_lib.LocationConfig:
    """Location config fixture with all required attributes."""
    # Use the helper function with a specific code for NYC
    return create_test_location_config(code="nyc")


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
def mock_tide_data() -> pd.DataFrame:
    """Create mock high/low tide prediction data."""
    index = pd.DatetimeIndex(
        [
            datetime.datetime(2025, 4, 22, 0, 0, 0),
            datetime.datetime(2025, 4, 22, 6, 0, 0),
            datetime.datetime(2025, 4, 22, 12, 0, 0),
            datetime.datetime(2025, 4, 22, 18, 0, 0),
        ]
    )
    return pd.DataFrame(
        {
            "prediction": [0.0, 4.0, 0.0, 4.0],
            "type": pd.Categorical(
                ["low", "high", "low", "high"],
                categories=["low", "high"],
            ),
        },
        index=index,
    )


@pytest_asyncio.fixture
async def process_pool() -> AsyncGenerator[ProcessPoolExecutor]:
    """Fixture to provide a ProcessPoolExecutor and ensure it's shut down."""
    pool = ProcessPoolExecutor()
    yield pool
    pool.shutdown(wait=False)  # Ensure pool is shut down after test


@pytest.fixture
def mock_clients() -> Mapping[str, BaseApiClient]:
    """Provide mock API clients for testing."""
    return {
        "coops": MagicMock(spec=CoopsApi),
        "ndbc": MagicMock(spec=NdbcApi),
        "nwis": MagicMock(spec=NwisApi),
    }  # type: ignore[return-value]


@pytest_asyncio.fixture
async def mock_data_with_currents(
    mock_config: config_lib.LocationConfig,
    mock_current_data: pd.DataFrame,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
) -> AsyncGenerator[LocationDataManager]:
    """Create a LocationDataManager instance with mock current data."""
    # Use freezegun to set a fixed time for testing
    with freeze_time("2025-04-22 12:00:00"):
        # Use the provided process_pool fixture with mock clients
        data = LocationDataManager(
            mock_config,
            clients=mock_clients,  # type: ignore[reportArgumentType]
            process_pool=process_pool,
        )

        # Create a mock currents feed
        mock_currents_feed = MagicMock()
        mock_currents_feed.values = mock_current_data
        mock_currents_feed.is_expired = False

        # Set the mock feed in the _feeds dictionary
        data._feeds["currents"] = mock_currents_feed

        yield data

    # Pool shutdown is handled by the process_pool fixture


@pytest.mark.asyncio
async def test_current_prediction_at_flood_peak(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Test current prediction at a flood peak."""
    # Test at 3:00 PM (15:00) which is a flood peak at 1.5 knots
    t = datetime.datetime(2025, 4, 22, 15, 0, 0)
    result = mock_data_with_currents.predict_flow_at_time(t)

    # Check the result
    assert result.direction == CurrentDirection.FLOODING  # Use Enum member
    assert result.phase == CurrentPhase.FLOOD
    assert result.strength == CurrentStrength.STRONG
    assert result.trend == CurrentTrend.BUILDING
    assert pytest.approx(result.magnitude, 0.1) == 1.5
    assert result.state_description == "strong flood and building"


@pytest.mark.asyncio
async def test_current_prediction_reuses_precomputed_frame(
    mock_data_with_currents: LocationDataManager,
    mock_current_data: pd.DataFrame,
    mocker: MockerFixture,
) -> None:
    """Prediction source data is precomputed once per feed update."""
    current_feed = mock_data_with_currents._feeds["currents"]
    assert current_feed is not None
    current_feed_mock = cast(Any, current_feed)
    current_feed_mock._data = mock_current_data
    current_feed_mock.values = mock_current_data
    current_feed_mock._fetch_timestamp = datetime.datetime(2025, 4, 22, 0, 0, 0)

    prepare_spy = mocker.patch(
        "shallweswim.core.manager.queries.prepare_current_prediction_frame",
        wraps=queries.prepare_current_prediction_frame,
    )

    t = datetime.datetime(2025, 4, 22, 15, 0, 0)
    first = mock_data_with_currents.predict_flow_at_time(t)
    second = mock_data_with_currents.predict_flow_at_time(t)

    assert first == second
    assert prepare_spy.call_count == 1

    updated_data = mock_current_data.copy()
    current_feed_mock._data = updated_data
    current_feed_mock.values = updated_data
    current_feed_mock._fetch_timestamp = datetime.datetime(2025, 4, 22, 1, 0, 0)

    mock_data_with_currents.predict_flow_at_time(t)
    assert prepare_spy.call_count == 2


def test_current_prediction_range_uses_segment_start_slack_when_building(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Building current range runs from prior slack to the segment peak."""
    result = mock_data_with_currents.predict_flow_at_time(
        datetime.datetime(2025, 4, 22, 13, 0, 0)
    )

    assert result.range is not None
    assert result.trend == CurrentTrend.BUILDING
    assert result.range.slack.timestamp == datetime.datetime(2025, 4, 22, 11, 36, 0)
    assert result.range.slack.magnitude == pytest.approx(0.0)
    assert result.range.slack.units == "kt"
    assert result.range.slack.phase is None
    assert result.range.peak.timestamp == datetime.datetime(2025, 4, 22, 15, 0, 0)
    assert result.range.peak.magnitude == pytest.approx(1.5)
    assert result.range.peak.units == "kt"
    assert result.range.peak.phase == CurrentPhase.FLOOD


def test_current_prediction_range_uses_segment_end_slack_when_easing(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Easing current range uses the upcoming slack boundary."""
    result = mock_data_with_currents.predict_flow_at_time(
        datetime.datetime(2025, 4, 22, 16, 0, 0)
    )

    assert result.range is not None
    assert result.trend == CurrentTrend.EASING
    assert result.range.slack.timestamp == datetime.datetime(2025, 4, 22, 18, 0, 0)
    assert result.range.slack.magnitude == pytest.approx(0.1)
    assert result.range.peak.timestamp == datetime.datetime(2025, 4, 22, 15, 0, 0)
    assert result.range.peak.magnitude == pytest.approx(1.5)
    assert result.range.peak.phase == CurrentPhase.FLOOD


def test_current_prediction_range_supports_ebb_segments(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Ebb ranges expose ebb peak context."""
    result = mock_data_with_currents.predict_flow_at_time(
        datetime.datetime(2025, 4, 22, 7, 0, 0)
    )

    assert result.range is not None
    assert result.phase == CurrentPhase.EBB
    assert result.trend == CurrentTrend.BUILDING
    assert result.range.slack.timestamp == datetime.datetime(2025, 4, 22, 4, 0, 0)
    assert result.range.slack.magnitude == pytest.approx(0.1)
    assert result.range.peak.timestamp == datetime.datetime(2025, 4, 22, 8, 0, 0)
    assert result.range.peak.magnitude == pytest.approx(1.5)
    assert result.range.peak.phase == CurrentPhase.EBB


def test_prepare_current_prediction_frame_derives_segment_context(
    mock_current_data: pd.DataFrame,
) -> None:
    """Current preprocessing derives per-segment peak and slack context."""
    frame = queries.prepare_current_prediction_frame(mock_current_data)

    required_columns = {
        "local_mag_pct",
        "segment_id",
        "segment_phase",
        "segment_peak_time",
        "segment_peak_magnitude",
        "segment_start_slack_time",
        "segment_start_slack_magnitude",
        "segment_end_slack_time",
        "segment_end_slack_magnitude",
    }
    assert required_columns.issubset(frame.columns)

    flood_row = frame.loc[datetime.datetime(2025, 4, 22, 13, 0, 0)]
    assert flood_row["segment_phase"] == "flood"
    assert flood_row["segment_peak_time"] == datetime.datetime(2025, 4, 22, 15, 0, 0)
    assert flood_row["segment_peak_magnitude"] == pytest.approx(1.5)
    assert flood_row["local_mag_pct"] == pytest.approx(0.7 / 1.5)
    assert flood_row["segment_start_slack_time"] == datetime.datetime(
        2025, 4, 22, 11, 36, 0
    )
    assert flood_row["segment_start_slack_magnitude"] == pytest.approx(0.0)
    assert flood_row["segment_end_slack_time"] == datetime.datetime(
        2025, 4, 22, 18, 0, 0
    )
    assert flood_row["segment_end_slack_magnitude"] == pytest.approx(0.1)

    peak_row = frame.loc[datetime.datetime(2025, 4, 22, 15, 0, 0)]
    assert peak_row["local_mag_pct"] == pytest.approx(1.0)


def test_prepare_current_prediction_frame_normalizes_within_ebb_segment() -> None:
    """Ebb percentages are normalized against their own segment peak."""
    index = pd.DatetimeIndex(
        [
            datetime.datetime(2025, 4, 22, 18, 0, 0),
            datetime.datetime(2025, 4, 22, 19, 0, 0),
            datetime.datetime(2025, 4, 22, 20, 0, 0),
            datetime.datetime(2025, 4, 22, 21, 0, 0),
            datetime.datetime(2025, 4, 22, 22, 0, 0),
        ]
    )
    currents = pd.DataFrame(
        {"velocity": [0.0, -0.4, -1.2, -0.8, 0.0]},
        index=index,
    )

    frame = queries.prepare_current_prediction_frame(currents)

    ebb_row = frame.loc[datetime.datetime(2025, 4, 22, 20, 0, 0)]
    assert ebb_row["segment_phase"] == "ebb"
    assert ebb_row["segment_peak_time"] == datetime.datetime(2025, 4, 22, 20, 0, 0)
    assert ebb_row["segment_peak_magnitude"] == pytest.approx(1.2)
    assert ebb_row["local_mag_pct"] == pytest.approx(1.0)

    easing_row = frame.loc[datetime.datetime(2025, 4, 22, 21, 0, 0)]
    assert easing_row["local_mag_pct"] == pytest.approx(0.8 / 1.2)
    assert easing_row["segment_start_slack_time"] == datetime.datetime(
        2025, 4, 22, 18, 0, 0
    )
    assert easing_row["segment_end_slack_time"] == datetime.datetime(
        2025, 4, 22, 22, 0, 0
    )


def test_prepare_current_prediction_frame_handles_all_slack_data() -> None:
    """A current frame with no non-slack segments remains queryable."""
    index = pd.DatetimeIndex(
        [
            datetime.datetime(2025, 4, 22, 18, 0, 0),
            datetime.datetime(2025, 4, 22, 19, 0, 0),
            datetime.datetime(2025, 4, 22, 20, 0, 0),
        ]
    )
    currents = pd.DataFrame({"velocity": [0.0, 0.1, -0.1]}, index=index)

    frame = queries.prepare_current_prediction_frame(currents)

    assert frame["local_mag_pct"].tolist() == [0.0, 0.0, 0.0]
    assert frame["segment_id"].isna().all()
    assert frame["segment_peak_time"].isna().all()
    assert frame["segment_peak_magnitude"].isna().all()

    result = queries.predict_flow_from_precomputed_frame(
        frame,
        create_test_location_config(),
        datetime.datetime(2025, 4, 22, 19, 0, 0),
    )
    assert result.phase == CurrentPhase.SLACK
    assert result.range is None


def test_prepare_tide_prediction_frame_derives_minute_curve(
    mock_tide_data: pd.DataFrame,
) -> None:
    """High/low tide events are precomputed into a minute-resolution curve."""
    result = queries.prepare_tide_prediction_frame(mock_tide_data)

    assert result.index[0] == datetime.datetime(2025, 4, 22, 0, 0, 0)
    assert result.index[1] == datetime.datetime(2025, 4, 22, 0, 1, 0)
    assert result.index[-1] == datetime.datetime(2025, 4, 22, 18, 0, 0)
    assert set(result.columns) == {"prediction", "height_pct", "trend"}
    assert result.loc[datetime.datetime(2025, 4, 22, 3, 0, 0), "trend"] == "rising"
    assert result.loc[datetime.datetime(2025, 4, 22, 9, 0, 0), "trend"] == "falling"
    assert 0.0 <= result["height_pct"].min() <= result["height_pct"].max() <= 1.0


def test_prepare_tide_prediction_frame_supports_two_event_linear_curve() -> None:
    """Two high/low events are enough for a linear fallback tide curve."""
    tides = pd.DataFrame(
        {
            "prediction": [0.0, 4.0],
            "type": pd.Categorical(["low", "high"], categories=["low", "high"]),
        },
        index=pd.DatetimeIndex(
            [
                datetime.datetime(2025, 4, 22, 0, 0, 0),
                datetime.datetime(2025, 4, 22, 6, 0, 0),
            ]
        ),
    )

    result = queries.prepare_tide_prediction_frame(tides)

    midpoint = result.loc[datetime.datetime(2025, 4, 22, 3, 0, 0)]
    assert midpoint["prediction"] == pytest.approx(2.0)
    assert midpoint["height_pct"] == pytest.approx(0.5)
    assert midpoint["trend"] == "rising"


def test_prepare_tide_prediction_frame_rejects_insufficient_events() -> None:
    """A tide curve needs at least two high/low events."""
    tides = pd.DataFrame(
        {
            "prediction": [4.0],
            "type": pd.Categorical(["high"], categories=["low", "high"]),
        },
        index=pd.DatetimeIndex([datetime.datetime(2025, 4, 22, 0, 0, 0)]),
    )

    with pytest.raises(
        DataUnavailableError,
        match="At least two tide predictions are required",
    ):
        queries.prepare_tide_prediction_frame(tides)


def test_prepare_tide_prediction_frame_fails_fast_on_timezone_aware_feed_data() -> None:
    """Precomputed tide curves fail loudly for invalid feed frame contracts."""
    tides = pd.DataFrame(
        {
            "prediction": [0.0, 4.0],
            "type": pd.Categorical(["low", "high"], categories=["low", "high"]),
        },
        index=pd.DatetimeIndex(
            [
                datetime.datetime(2025, 4, 22, 0, 0, 0),
                datetime.datetime(2025, 4, 22, 6, 0, 0),
            ],
            tz=datetime.UTC,
        ),
    )

    with pytest.raises(ValueError, match="should use naive datetimes"):
        queries.prepare_tide_prediction_frame(tides)


def test_predict_tide_from_precomputed_frame_returns_estimated_state(
    mock_config: config_lib.LocationConfig,
    mock_tide_data: pd.DataFrame,
) -> None:
    """A precomputed tide frame supports cheap point-in-time tide state lookup."""
    frame = queries.prepare_tide_prediction_frame(mock_tide_data)

    result = queries.predict_tide_from_precomputed_frame(
        frame,
        mock_config,
        datetime.datetime(2025, 4, 22, 3, 0, 30),
    )

    assert result.timestamp == datetime.datetime(2025, 4, 22, 3, 0, 0)
    assert result.estimated_height == pytest.approx(
        frame.loc[datetime.datetime(2025, 4, 22, 3, 0, 0), "prediction"]
    )
    assert result.units == "ft"
    assert result.trend == TideTrend.RISING
    assert 0.0 <= cast(float, result.height_pct) <= 1.0


def test_predict_tide_from_precomputed_frame_rejects_timezone_aware_input(
    mock_config: config_lib.LocationConfig,
    mock_tide_data: pd.DataFrame,
) -> None:
    """Tide state lookup requires local-naive input datetimes."""
    frame = queries.prepare_tide_prediction_frame(mock_tide_data)

    with pytest.raises(ValueError, match="Input datetime must be naive"):
        queries.predict_tide_from_precomputed_frame(
            frame,
            mock_config,
            datetime.datetime(2025, 4, 22, 3, 0, 0, tzinfo=datetime.UTC),
        )


def test_predict_tide_from_precomputed_frame_converts_nan_height_pct_to_none(
    mock_config: config_lib.LocationConfig,
    mock_tide_data: pd.DataFrame,
) -> None:
    """Unavailable normalized tide height serializes as null, not NaN."""
    frame = queries.prepare_tide_prediction_frame(mock_tide_data)
    frame.loc[datetime.datetime(2025, 4, 22, 3, 0, 0), "height_pct"] = float("nan")

    result = queries.predict_tide_from_precomputed_frame(
        frame,
        mock_config,
        datetime.datetime(2025, 4, 22, 3, 0, 0),
    )

    assert result.height_pct is None


@pytest.mark.asyncio
async def test_tide_prediction_reuses_precomputed_frame(
    mock_config: config_lib.LocationConfig,
    mock_tide_data: pd.DataFrame,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
    mocker: MockerFixture,
) -> None:
    """Tide prediction source data is precomputed once per feed update."""
    data = LocationDataManager(
        mock_config,
        clients=mock_clients,  # type: ignore[reportArgumentType]
        process_pool=process_pool,
    )
    tide_feed = MagicMock()
    tide_feed._data = mock_tide_data
    tide_feed.values = mock_tide_data
    tide_feed._fetch_timestamp = datetime.datetime(2025, 4, 22, 0, 0, 0)
    data._feeds[feeds.FEED_TIDES] = tide_feed

    prepare_spy = mocker.patch(
        "shallweswim.core.manager.queries.prepare_tide_prediction_frame",
        wraps=queries.prepare_tide_prediction_frame,
    )

    data._precompute_tide_predictions()
    data._precompute_tide_predictions()

    assert prepare_spy.call_count == 1
    assert data._tide_prediction_frame is not None

    updated_data = mock_tide_data.copy()
    tide_feed._data = updated_data
    tide_feed.values = updated_data
    tide_feed._fetch_timestamp = datetime.datetime(2025, 4, 22, 1, 0, 0)

    data._precompute_tide_predictions()
    assert prepare_spy.call_count == 2


@pytest.mark.asyncio
async def test_predict_tide_at_time_reuses_precomputed_frame(
    mock_config: config_lib.LocationConfig,
    mock_tide_data: pd.DataFrame,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
    mocker: MockerFixture,
) -> None:
    """Manager tide state lookup reuses the derived tide frame."""
    data = LocationDataManager(
        mock_config,
        clients=mock_clients,  # type: ignore[reportArgumentType]
        process_pool=process_pool,
    )
    tide_feed = MagicMock()
    tide_feed._data = mock_tide_data
    tide_feed.values = mock_tide_data
    tide_feed._fetch_timestamp = datetime.datetime(2025, 4, 22, 0, 0, 0)
    data._feeds[feeds.FEED_TIDES] = tide_feed

    prepare_spy = mocker.patch(
        "shallweswim.core.manager.queries.prepare_tide_prediction_frame",
        wraps=queries.prepare_tide_prediction_frame,
    )

    t = datetime.datetime(2025, 4, 22, 3, 0, 0)
    first = data.predict_tide_at_time(t)
    second = data.predict_tide_at_time(t)

    assert first == second
    assert first is not None
    assert first.trend == TideTrend.RISING
    assert prepare_spy.call_count == 1


@pytest.mark.asyncio
async def test_tide_prediction_precompute_skips_missing_tide_source(
    mock_config: config_lib.LocationConfig,
    mock_tide_data: pd.DataFrame,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
    mocker: MockerFixture,
) -> None:
    """Locations without tide support do not try to build a tide curve."""
    config = mock_config.model_copy(update={"tide_source": None})
    data = LocationDataManager(
        config,
        clients=mock_clients,  # type: ignore[reportArgumentType]
        process_pool=process_pool,
    )
    tide_feed = MagicMock()
    tide_feed._data = mock_tide_data
    tide_feed.values = mock_tide_data
    data._feeds[feeds.FEED_TIDES] = tide_feed

    prepare_spy = mocker.patch(
        "shallweswim.core.manager.queries.prepare_tide_prediction_frame",
        wraps=queries.prepare_tide_prediction_frame,
    )

    data._precompute_tide_predictions()

    assert prepare_spy.call_count == 0
    assert data._tide_prediction_frame is None


@pytest.mark.asyncio
async def test_predict_tide_at_time_returns_none_without_tide_source(
    mock_config: config_lib.LocationConfig,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
) -> None:
    """Locations without tide support do not expose tide state."""
    config = mock_config.model_copy(update={"tide_source": None})
    data = LocationDataManager(
        config,
        clients=mock_clients,  # type: ignore[reportArgumentType]
        process_pool=process_pool,
    )

    assert data.predict_tide_at_time(datetime.datetime(2025, 4, 22, 3, 0, 0)) is None


@pytest.mark.asyncio
async def test_tide_prediction_precompute_skips_missing_feed_data(
    mock_config: config_lib.LocationConfig,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
    mocker: MockerFixture,
) -> None:
    """Configured tide feeds without loaded data do not fail precompute."""
    data = LocationDataManager(
        mock_config,
        clients=mock_clients,  # type: ignore[reportArgumentType]
        process_pool=process_pool,
    )
    tide_feed = MagicMock()
    tide_feed._data = None
    data._feeds[feeds.FEED_TIDES] = tide_feed

    prepare_spy = mocker.patch(
        "shallweswim.core.manager.queries.prepare_tide_prediction_frame",
        wraps=queries.prepare_tide_prediction_frame,
    )

    data._precompute_tide_predictions()

    assert prepare_spy.call_count == 0
    assert data._tide_prediction_frame is None


@pytest.mark.asyncio
async def test_predict_tide_at_time_returns_none_without_feed_data(
    mock_config: config_lib.LocationConfig,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
) -> None:
    """Configured tide locations without loaded tide data do not expose tide state."""
    data = LocationDataManager(
        mock_config,
        clients=mock_clients,  # type: ignore[reportArgumentType]
        process_pool=process_pool,
    )
    tide_feed = MagicMock()
    tide_feed._data = None
    data._feeds[feeds.FEED_TIDES] = tide_feed

    assert data.predict_tide_at_time(datetime.datetime(2025, 4, 22, 3, 0, 0)) is None


@pytest.mark.asyncio
async def test_tide_prediction_precompute_keeps_unavailable_frame_optional(
    mock_config: config_lib.LocationConfig,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
    mocker: MockerFixture,
) -> None:
    """Sparse high/low data leaves the derived tide frame absent until data changes."""
    data = LocationDataManager(
        mock_config,
        clients=mock_clients,  # type: ignore[reportArgumentType]
        process_pool=process_pool,
    )
    sparse_tides = pd.DataFrame(
        {
            "prediction": [4.0],
            "type": pd.Categorical(["high"], categories=["low", "high"]),
        },
        index=pd.DatetimeIndex([datetime.datetime(2025, 4, 22, 0, 0, 0)]),
    )
    tide_feed = MagicMock()
    tide_feed._data = sparse_tides
    tide_feed.values = sparse_tides
    tide_feed._fetch_timestamp = datetime.datetime(2025, 4, 22, 0, 0, 0)
    data._feeds[feeds.FEED_TIDES] = tide_feed

    prepare_spy = mocker.patch(
        "shallweswim.core.manager.queries.prepare_tide_prediction_frame",
        wraps=queries.prepare_tide_prediction_frame,
    )

    data._precompute_tide_predictions()
    data._precompute_tide_predictions()

    assert prepare_spy.call_count == 1
    assert data._tide_prediction_frame is None


@pytest.mark.asyncio
async def test_predict_tide_at_time_returns_none_when_derived_frame_unavailable(
    mock_config: config_lib.LocationConfig,
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
) -> None:
    """Sparse high/low data leaves tide state unavailable without crashing."""
    data = LocationDataManager(
        mock_config,
        clients=mock_clients,  # type: ignore[reportArgumentType]
        process_pool=process_pool,
    )
    sparse_tides = pd.DataFrame(
        {
            "prediction": [4.0],
            "type": pd.Categorical(["high"], categories=["low", "high"]),
        },
        index=pd.DatetimeIndex([datetime.datetime(2025, 4, 22, 0, 0, 0)]),
    )
    tide_feed = MagicMock()
    tide_feed._data = sparse_tides
    tide_feed.values = sparse_tides
    tide_feed._fetch_timestamp = datetime.datetime(2025, 4, 22, 0, 0, 0)
    data._feeds[feeds.FEED_TIDES] = tide_feed

    assert data.predict_tide_at_time(datetime.datetime(2025, 4, 22, 3, 0, 0)) is None


@pytest.mark.asyncio
async def test_current_prediction_at_ebb_peak(
    process_pool: ProcessPoolExecutor,
) -> None:
    """Test current prediction at an ebb peak."""
    # Create a proper LocationConfig object
    config = create_test_location_config()

    # Create mock clients
    mock_clients = {
        "coops": MagicMock(spec=CoopsApi),
        "ndbc": MagicMock(spec=NdbcApi),
        "nwis": MagicMock(spec=NwisApi),
    }

    # Use freezegun to set a fixed time for testing
    with freeze_time("2025-04-22 12:00:00"):
        data = LocationDataManager(
            config,
            clients=mock_clients,  # type: ignore[reportArgumentType]
            process_pool=process_pool,
        )

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
        1.2,  # 3:00 - Flood peak
        0.8,  # 4:00 - Weakening flood
        0.2,  # 5:00 - Near slack
        -0.4,  # 6:00 - Strengthening ebb
        -0.9,  # 7:00 - Strengthening ebb
        -1.3,  # 8:00 - Ebb peak
        -1.0,  # 9:00 - Weakening ebb
        -0.5,  # 10:00 - Weakening ebb
        -0.1,  # 11:00 - Near slack
        0.3,  # 12:00 - Strengthening flood
        0.7,  # 13:00 - Strengthening flood
        1.1,  # 14:00 - Strengthening flood
        1.4,  # 15:00 - Flood peak
        1.0,  # 16:00 - Weakening flood
        0.4,  # 17:00 - Weakening flood
        -0.2,  # 18:00 - Near slack / turning ebb
        -0.7,  # 19:00 - Strengthening ebb
        -1.2,  # 20:00 - Strengthening ebb
        -1.5,  # 21:00 - Ebb peak
        -1.1,  # 22:00 - Weakening ebb
        -0.6,  # 23:00 - Weakening ebb
    ]
    mock_df = pd.DataFrame({"velocity": velocities}, index=idx)

    # Create a mock currents feed
    mock_currents_feed = MagicMock()
    mock_currents_feed.values = mock_df
    mock_currents_feed.is_expired = False

    # Set the mock feed in the _feeds dictionary
    data._feeds["currents"] = mock_currents_feed

    # Test at 9:00 PM (21:00) which is an ebb peak at -1.5 knots
    t = datetime.datetime(2025, 4, 22, 21, 0, 0)
    result = data.predict_flow_at_time(t)

    # Check the result
    assert result.direction == CurrentDirection.EBBING  # Use Enum member
    assert result.phase == CurrentPhase.EBB
    assert result.strength == CurrentStrength.STRONG
    assert result.trend == CurrentTrend.BUILDING
    assert pytest.approx(result.magnitude, 0.1) == 1.5  # Magnitude is positive
    assert result.state_description == "strong ebb and building"
    # Pool shutdown is handled by the process_pool fixture


@pytest.mark.asyncio
async def test_current_prediction_at_slack(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Test current prediction at slack water."""
    # Test at 4:00 AM (4:00) which is near zero (0.1 knots)
    t = datetime.datetime(2025, 4, 22, 4, 0, 0)
    result = mock_data_with_currents.predict_flow_at_time(t)

    # Check the result
    assert result.phase == CurrentPhase.SLACK_BEFORE_EBB
    assert result.strength is None
    assert result.trend is None
    assert result.magnitude < 0.2
    assert result.state_description == "slack before ebb"


@pytest.mark.asyncio
async def test_current_prediction_at_slack_before_flood(
    process_pool: ProcessPoolExecutor,
) -> None:
    """Slack phase indicates the next non-slack flood direction."""
    config = create_test_location_config()
    mock_clients = {
        "coops": MagicMock(spec=CoopsApi),
        "ndbc": MagicMock(spec=NdbcApi),
        "nwis": MagicMock(spec=NwisApi),
    }

    with freeze_time("2025-04-22 12:00:00"):
        data = LocationDataManager(
            config,
            clients=mock_clients,  # type: ignore[reportArgumentType]
            process_pool=process_pool,
        )

    idx = pd.DatetimeIndex(
        [
            datetime.datetime(2025, 4, 22, 10, 0, 0),
            datetime.datetime(2025, 4, 22, 11, 0, 0),
            datetime.datetime(2025, 4, 22, 12, 0, 0),
        ]
    )
    mock_df = pd.DataFrame({"velocity": [-0.4, 0.0, 0.5]}, index=idx)

    mock_currents_feed = MagicMock()
    mock_currents_feed.values = mock_df
    mock_currents_feed.is_expired = False
    data._feeds["currents"] = mock_currents_feed

    result = data.predict_flow_at_time(datetime.datetime(2025, 4, 22, 11, 0, 0))

    assert result.phase == CurrentPhase.SLACK_BEFORE_FLOOD
    assert result.direction == CurrentDirection.FLOODING
    assert result.strength is None
    assert result.trend is None
    assert result.magnitude == 0.0
    assert result.state_description == "slack before flood"


@pytest.mark.asyncio
async def test_current_prediction_rejects_timezone_aware_input(
    mock_data_with_currents: LocationDataManager,
) -> None:
    """Current predictions require naive local datetimes."""
    t = datetime.datetime(2025, 4, 22, 11, 0, 0, tzinfo=datetime.UTC)

    with pytest.raises(ValueError, match="Input datetime must be naive"):
        mock_data_with_currents.predict_flow_at_time(t)


@pytest.mark.asyncio
async def test_current_prediction_rejects_timezone_aware_feed_data(
    mock_data_with_currents: LocationDataManager,
    mock_current_data: pd.DataFrame,
) -> None:
    """Current prediction data must use naive local datetimes."""
    mock_currents_feed = MagicMock()
    mock_currents_feed._data = mock_current_data
    mock_currents_feed.values = mock_current_data.tz_localize("UTC")
    mock_currents_feed.is_expired = False
    mock_data_with_currents._feeds["currents"] = mock_currents_feed

    with pytest.raises(
        DataUnavailableError,
        match="Current prediction DataFrame should use naive datetimes",
    ):
        mock_data_with_currents.predict_flow_at_time(
            datetime.datetime(2025, 4, 22, 11, 0, 0)
        )


def test_tide_info_rejects_timezone_aware_feed_data(
    mock_config: config_lib.LocationConfig,
) -> None:
    """Tide data must use naive local datetimes."""
    tides_df = pd.DataFrame(
        {
            "time": [datetime.datetime(2025, 4, 22, 10, 0, 0)],
            "type": ["high"],
            "prediction": [4.2],
        },
        index=pd.DatetimeIndex(
            [datetime.datetime(2025, 4, 22, 10, 0, 0)], tz=datetime.UTC
        ),
    )
    mock_tides_feed = MagicMock()
    mock_tides_feed._data = tides_df
    mock_tides_feed.values = tides_df

    with pytest.raises(
        DataUnavailableError,
        match="Tide DataFrame should use naive datetimes",
    ):
        get_current_tide_info({feeds.FEED_TIDES: mock_tides_feed}, mock_config)


@pytest.mark.asyncio
async def test_current_prediction_strengthening() -> None:
    """Test current prediction when current is strengthening."""
    # Create a proper LocationConfig object
    config = create_test_location_config()

    # Create mock clients
    mock_clients = {
        "coops": MagicMock(spec=CoopsApi),
        "ndbc": MagicMock(spec=NdbcApi),
        "nwis": MagicMock(spec=NwisApi),
    }  # type: ignore[assignment]

    # Provide a process pool for the test
    pool = ProcessPoolExecutor()

    # Use freezegun to set a fixed time for testing
    with freeze_time("2025-04-22 12:00:00"):
        data = LocationDataManager(config, clients=mock_clients, process_pool=pool)  # type: ignore[arg-type]

    try:
        # Create data with a clear strengthening pattern
        hours = [12, 13, 14]
        idx = pd.DatetimeIndex([datetime.datetime(2025, 4, 22, h, 0, 0) for h in hours])
        # Clearly increasing velocity
        velocities = [0.5, 0.8, 1.2]

        # Create the DataFrame with strengthening pattern
        df = pd.DataFrame({"velocity": velocities}, index=idx)

        # Create a mock currents feed
        mock_currents_feed = MagicMock()
        mock_currents_feed.values = df
        mock_currents_feed.is_expired = False

        # Set the mock feed in the _feeds dictionary
        data._feeds["currents"] = mock_currents_feed

        # Test at the middle point where slope is positive
        t = datetime.datetime(2025, 4, 22, 13, 0, 0)
        result = data.predict_flow_at_time(t)

        # Check the result
        assert result.direction == CurrentDirection.FLOODING  # Use Enum member
        assert result.trend == CurrentTrend.BUILDING
        assert (
            result.state_description is not None
            and "flood and building" in result.state_description
        )
    finally:
        # Clean up the process pool
        pool.shutdown(wait=False)  # Don't wait in tests


@pytest.mark.asyncio
async def test_current_prediction_weakening() -> None:
    """Test current prediction when current is weakening."""
    # Create a proper LocationConfig object
    config = create_test_location_config()

    # Create mock clients
    mock_clients = {
        "coops": MagicMock(spec=CoopsApi),
        "ndbc": MagicMock(spec=NdbcApi),
        "nwis": MagicMock(spec=NwisApi),
    }  # type: ignore[assignment]

    # Provide a process pool for the test
    pool = ProcessPoolExecutor()

    # Use freezegun to set a fixed time for testing
    with freeze_time("2025-04-22 12:00:00"):
        data = LocationDataManager(config, clients=mock_clients, process_pool=pool)  # type: ignore[arg-type]

    try:
        # Create data with a clear weakening pattern
        hours = [15, 16, 17]
        idx = pd.DatetimeIndex([datetime.datetime(2025, 4, 22, h, 0, 0) for h in hours])
        # Clearly decreasing velocity
        velocities = [1.2, 0.8, 0.5]

        # Create the DataFrame with weakening pattern
        df = pd.DataFrame({"velocity": velocities}, index=idx)

        # Create a mock currents feed
        mock_currents_feed = MagicMock()
        mock_currents_feed.values = df
        mock_currents_feed.is_expired = False

        # Set the mock feed in the _feeds dictionary
        data._feeds["currents"] = mock_currents_feed

        # Test at the middle point where slope is negative
        t = datetime.datetime(2025, 4, 22, 16, 0, 0)
        result = data.predict_flow_at_time(t)

        # Check the result
        assert result.direction == CurrentDirection.FLOODING  # Use Enum member
        assert result.trend == CurrentTrend.EASING
        assert (
            result.state_description is not None
            and "flood and easing" in result.state_description
        )
    finally:
        # Clean up the process pool
        pool.shutdown(wait=False)  # Don't wait in tests


@pytest.mark.asyncio
async def test_process_peaks_function() -> None:
    """Test that peaks are identified properly in the CurrentPrediction method."""
    # Create a proper LocationConfig object
    config = create_test_location_config()

    # Create mock clients
    mock_clients = {
        "coops": MagicMock(spec=CoopsApi),
        "ndbc": MagicMock(spec=NdbcApi),
        "nwis": MagicMock(spec=NwisApi),
    }  # type: ignore[assignment]

    # Provide a process pool for the test
    pool = ProcessPoolExecutor()

    # Use freezegun to set a fixed time for testing
    with freeze_time("2025-04-22 12:00:00"):
        data = LocationDataManager(config, clients=mock_clients, process_pool=pool)  # type: ignore[arg-type]

    try:
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

        # Create a mock currents feed
        mock_currents_feed = MagicMock()
        mock_currents_feed.values = currents_df
        mock_currents_feed.is_expired = False

        # Set the mock feed in the _feeds dictionary
        data._feeds["currents"] = mock_currents_feed

        # Use a time point that's exactly at the peak
        # Convert pandas Timestamp to Python datetime.datetime using simple casting
        peak_time = datetime.datetime(
            2025, 4, 22, 1, 0
        )  # This matches index[1] (2025-04-22 01:00:00)
        result = data.predict_flow_at_time(peak_time)

        # The peak should be detected and described correctly
        assert result.direction == CurrentDirection.FLOODING  # Use Enum member
        assert pytest.approx(result.magnitude, abs=0.1) == 1.5

        assert result.strength == CurrentStrength.STRONG
        assert result.trend == CurrentTrend.BUILDING
        assert result.state_description == "strong flood and building"
    finally:
        # Clean up the process pool
        pool.shutdown(wait=False)  # Don't wait in tests


@pytest.mark.asyncio
@freeze_time("2025-04-22 15:00:00")
async def test_current_info_representation() -> None:
    """Test the representation of CurrentInfo objects."""
    # Test with flooding current
    test_timestamp = datetime.datetime(2025, 4, 22, 15, 0, 0)
    flood_info = CurrentInfo(
        timestamp=test_timestamp,
        source_type=DataSourceType.PREDICTION,
        direction=CurrentDirection.FLOODING,  # Use Enum member
        phase=CurrentPhase.FLOOD,
        strength=CurrentStrength.STRONG,
        trend=CurrentTrend.BUILDING,
        magnitude=1.5,
        magnitude_pct=0.8,
        state_description="strong flood and building",
    )
    assert flood_info.direction == CurrentDirection.FLOODING  # Use Enum member
    assert flood_info.phase == CurrentPhase.FLOOD
    assert flood_info.strength == CurrentStrength.STRONG
    assert flood_info.trend == CurrentTrend.BUILDING
    assert flood_info.magnitude == 1.5
    assert flood_info.magnitude_pct == 0.8
    assert flood_info.state_description == "strong flood and building"
    # Check that the string representation contains the important information
    flood_str = str(flood_info)
    assert "flooding" in flood_str.lower()
    assert "1.5" in flood_str
    assert "strong flood and building" in flood_str

    # Test with ebbing current
    ebb_info = CurrentInfo(
        timestamp=test_timestamp,
        source_type=DataSourceType.PREDICTION,
        direction=CurrentDirection.EBBING,  # Use Enum member
        phase=CurrentPhase.EBB,
        strength=CurrentStrength.MODERATE,
        trend=CurrentTrend.EASING,
        magnitude=1.2,
        magnitude_pct=0.6,
        state_description="moderate ebb and easing",
    )
    assert ebb_info.direction == CurrentDirection.EBBING  # Use Enum member
    assert ebb_info.phase == CurrentPhase.EBB
    assert ebb_info.strength == CurrentStrength.MODERATE
    assert ebb_info.trend == CurrentTrend.EASING
    assert ebb_info.magnitude == 1.2
    assert ebb_info.magnitude_pct == 0.6
    assert ebb_info.state_description == "moderate ebb and easing"
    # Check that the string representation contains the important information
    ebb_str = str(ebb_info)
    assert "ebbing" in ebb_str.lower()
    assert "1.2" in ebb_str
    assert "moderate ebb and easing" in ebb_str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tides_timestamp, live_temps_timestamp, historic_temps_timestamp, expected_ready, configured_feeds",
    [
        # --- All feeds configured ---
        # Case 1: All feeds ready (timestamps are recent)
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            True,  # Expected ready
            True,  # All feeds configured
        ),
        # Case 2: Tides expired
        (
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            False,  # Expected not ready
            True,
        ),
        # Case 3: Live temps expired
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            False,  # Expected not ready
            True,
        ),
        # Case 4: Historic temps expired
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            datetime.datetime(2025, 4, 19, 12, 0, 0),
            False,  # Expected not ready
            True,
        ),
        # Case 5: Multiple feeds expired
        (
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            False,  # Expected not ready
            True,
        ),
        # --- Only tides and historic configured ---
        # Case 6: Configured feeds ready
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            None,  # Live temps not configured
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            True,  # Expected ready
            False,  # Live temps not configured
        ),
        # Case 7: Configured tides expired
        (
            datetime.datetime(2025, 4, 27, 10, 0, 0),
            None,
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            False,  # Expected not ready
            False,
        ),
        # Case 8: Configured historic temps expired
        (
            datetime.datetime(2025, 4, 27, 12, 0, 0),
            None,
            datetime.datetime(2025, 4, 19, 12, 0, 0),
            False,  # Expected not ready
            False,
        ),
    ],
)
@pytest.mark.asyncio
@freeze_time("2025-04-27 12:00:00")  # Freeze time for consistent expiration checks
# pylint: disable=unused-argument
async def test_data_ready_property(
    tides_timestamp: datetime.datetime | None,
    live_temps_timestamp: datetime.datetime | None,
    historic_temps_timestamp: datetime.datetime | None,
    expected_ready: bool,
    configured_feeds: bool,  # True if live temps feed is configured
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
) -> None:
    """Test the ready property of the Data class.

    Tests different combinations of dataset states to verify the ready property
    accurately represents if all data has been loaded and is not expired.
    """
    # Use the helper function to create a LocationConfig with the appropriate configuration
    config = create_test_location_config(
        code="tst",
        live_temps_enabled=configured_feeds,  # Set based on the test parameter
        historic_temps_enabled=True,  # Always enable historic temps
    )

    # Use the provided process_pool fixture and mock clients
    data = LocationDataManager(config, clients=mock_clients, process_pool=process_pool)  # type: ignore[arg-type]  # type: ignore[arg-type]

    # --- Mock Feed Setup ---
    # Define expiration times for feeds
    expiration_map = {
        "tides": datetime.timedelta(hours=1),  # Example: 1 hour expiry
        "live_temps": datetime.timedelta(hours=1),  # Example: 1 hour expiry
        "historic_temps": datetime.timedelta(days=7),  # Example: 7 day expiry
    }

    # Create mock feeds for each data type
    tides_feed = None
    live_temps_feed = None
    historic_temps_feed = None

    # Create the tides feed if timestamp is provided
    if tides_timestamp is not None:
        tides_feed = MagicMock(spec=feeds.CoopsTidesFeed)
        tides_feed._fetch_timestamp = tides_timestamp
        tides_feed.expiration_interval = expiration_map["tides"]
        # Set is_expired based on the timestamp
        frozen_time = datetime.datetime(
            2025, 4, 27, 12, 0, 0
        )  # From the freeze_time decorator
        tides_feed.is_expired = (frozen_time - tides_timestamp) > expiration_map[
            "tides"
        ]

    # Create the live temps feed if configured and timestamp is provided
    if configured_feeds and live_temps_timestamp is not None:
        live_temps_feed = MagicMock(spec=feeds.CoopsTempFeed)
        live_temps_feed._fetch_timestamp = live_temps_timestamp
        live_temps_feed.expiration_interval = expiration_map["live_temps"]
        # Set is_expired based on the timestamp
        frozen_time = datetime.datetime(
            2025, 4, 27, 12, 0, 0
        )  # From the freeze_time decorator
        live_temps_feed.is_expired = (
            frozen_time - live_temps_timestamp
        ) > expiration_map["live_temps"]

    # Create the historic temps feed if timestamp is provided
    if historic_temps_timestamp is not None:
        historic_temps_feed = MagicMock(spec=feeds.HistoricalTempsFeed)
        historic_temps_feed._fetch_timestamp = historic_temps_timestamp
        historic_temps_feed.expiration_interval = expiration_map["historic_temps"]
        # Set is_expired based on the timestamp
        frozen_time = datetime.datetime(
            2025, 4, 27, 12, 0, 0
        )  # From the freeze_time decorator
        historic_temps_feed.is_expired = (
            frozen_time - historic_temps_timestamp
        ) > expiration_map["historic_temps"]

    # Directly set the feeds in the data manager
    data._feeds = {
        feeds.FeedName.TIDES: tides_feed,
        feeds.FeedName.LIVE_TEMPS: live_temps_feed,
        feeds.FeedName.HISTORIC_TEMPS: historic_temps_feed,
        feeds.FeedName.CURRENTS: None,  # Not used in this test
    }

    # --- Test the ready property ---
    assert data.ready == expected_ready


@pytest.mark.asyncio
@freeze_time("2025-04-27 12:00:00")
async def test_data_status_property(process_pool: ProcessPoolExecutor) -> None:
    """Test the status property of the LocationDataManager class.

    Verifies that the status property returns a dictionary mapping feed names to their status dictionaries,
    and that the dictionary is JSON serializable.
    """
    # Create a proper LocationConfig object
    config = create_test_location_config(code="nyc")

    # Create mock clients
    mock_clients = {
        "coops": MagicMock(spec=CoopsApi),
        "ndbc": MagicMock(spec=NdbcApi),
        "nwis": MagicMock(spec=NwisApi),
    }  # type: ignore[assignment]

    # Use the provided process_pool fixture with proper config and mock clients
    data = LocationDataManager(config, clients=mock_clients, process_pool=process_pool)  # type: ignore[arg-type]

    # Create mock feeds with status dictionaries
    mock_feeds = []
    for i in range(3):
        mock_feed = MagicMock(spec=Feed)
        # Use FeedStatus model instead of dict
        mock_feed.status = FeedStatus(
            name=f"MockFeed{i}",
            location="nyc",
            fetch_timestamp=datetime.datetime.fromisoformat("2025-04-27T12:00:00"),
            age_seconds=3600,
            is_expired=False,
            expiration_seconds=3600,
            data_summary=None,  # Explicitly set optional fields if needed
            error=None,  # Explicitly set optional fields if needed
            is_healthy=True,  # Update field name and value
        )
        mock_feeds.append(mock_feed)

    # Set the mock feeds in the _feeds dictionary
    data._feeds = {
        feeds.FeedName.TIDES: mock_feeds[0],
        feeds.FeedName.CURRENTS: mock_feeds[1],
        feeds.FeedName.LIVE_TEMPS: mock_feeds[2],
        feeds.FeedName.HISTORIC_TEMPS: None,  # Test with a None feed
    }

    # Get the status dictionary
    status = data.status

    # Check that the status dictionary contains the expected keys
    assert set(status.feeds.keys()) == {"tides", "currents", "live_temps"}

    # Check that the status dictionaries for each feed match the mock status
    assert status.feeds["tides"] == mock_feeds[0].status
    assert status.feeds["currents"] == mock_feeds[1].status
    assert status.feeds["live_temps"] == mock_feeds[2].status

    # Check that the status dictionary derived from the model is JSON serializable
    assert_json_serializable(status.model_dump(mode="json"))
    # Pool shutdown is handled by the process_pool fixture


# --- Tests for LocationDataManager.wait_until_ready ---


@pytest.mark.asyncio
async def test_start_rejects_duplicate_running_task(
    process_pool: ProcessPoolExecutor,
    mock_clients: Mapping[str, BaseApiClient],
) -> None:
    """Starting the same manager twice reports a lifecycle error."""
    config = create_test_location_config(code="tst")
    data = LocationDataManager(config, clients=mock_clients, process_pool=process_pool)  # type: ignore[arg-type]

    data.start()
    try:
        with pytest.raises(RuntimeError, match="Data update task already running"):
            data.start()
    finally:
        await data.stop(timeout=0.1)


@pytest.fixture
def mock_data_manager(
    process_pool: ProcessPoolExecutor, mock_clients: Mapping[str, BaseApiClient]
) -> LocationDataManager:
    """Fixture to create a LocationDataManager with mocked internals."""
    # Use the helper function to create a LocationConfig
    config = create_test_location_config(code="tst")

    # Create the data manager with real config and mock clients
    data = LocationDataManager(config, clients=mock_clients, process_pool=process_pool)  # type: ignore[arg-type]

    # Mock the event and task for testing
    data._ready_event = MagicMock(spec=asyncio.Event)
    data._update_task = MagicMock(spec=asyncio.Task)
    return data


@pytest.mark.asyncio
async def test_wait_until_ready_already_ready(
    mock_data_manager: LocationDataManager,
) -> None:
    """Test case 1: Manager is already ready."""
    mock_data_manager._ready_event.is_set.return_value = True  # type: ignore[attr-defined]
    result = await mock_data_manager.wait_until_ready(timeout=0.1)
    assert result is True
    mock_data_manager._ready_event.wait.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_wait_until_ready_task_not_started(
    mock_data_manager: LocationDataManager,
) -> None:
    """Test case 2: Update task hasn't been started (should raise RuntimeError)."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task = None
    with pytest.raises(RuntimeError, match="Update task has not been started"):
        await mock_data_manager.wait_until_ready(timeout=0.1)


@pytest.mark.asyncio
async def test_wait_until_ready_task_already_failed(
    mock_data_manager: LocationDataManager,
) -> None:
    """Test case 3: Update task already finished with an exception."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = True  # type: ignore[union-attr]
    mock_data_manager._update_task.exception.return_value = ValueError("Task failed")  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="Task failed"):
        await mock_data_manager.wait_until_ready(timeout=0.1)


@pytest.mark.asyncio
async def test_wait_until_ready_task_already_done_unexpectedly(
    mock_data_manager: LocationDataManager,
) -> None:
    """Test case 4: Update task already finished ok, but ready event not set."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = True  # type: ignore[union-attr]
    mock_data_manager._update_task.exception.return_value = None  # type: ignore[union-attr]

    with pytest.raises(RuntimeError, match="finished unexpectedly"):
        await mock_data_manager.wait_until_ready(timeout=0.1)


@pytest.mark.asyncio
async def test_wait_until_ready_event_set_during_wait(
    mock_data_manager: LocationDataManager,
    mocker: MockerFixture,
) -> None:
    """Test case 5: Ready event is set during the wait.

    Simulates the scenario where the ready event gets set *while* the
    wait_until_ready method is waiting.
    """
    # Initial setup: event not set, task not done
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = False  # type: ignore[union-attr]

    # Define the side effect for the event's wait() method
    async def wait_side_effect(*_args: Any, **_kwargs: Any) -> bool:
        # Simulate some delay during which the event might be set
        await asyncio.sleep(0)
        # Simulate the event being set externally
        mock_data_manager._ready_event.is_set.return_value = True  # type: ignore[attr-defined]
        # wait() should return True when the event is set
        return True

    # Mock event's wait() method with the async side effect
    mock_wait_method = mocker.patch.object(
        mock_data_manager._ready_event,
        "wait",
        new_callable=AsyncMock,
        side_effect=wait_side_effect,
    )

    # Call the method under test - use the real asyncio.wait
    result = await mock_data_manager.wait_until_ready(timeout=0.1)  # Short timeout

    # Assertions
    assert result is True
    mock_data_manager._update_task.exception.assert_not_called()  # type: ignore[union-attr]
    mock_wait_method.assert_awaited_once()  # Check the wait method was awaited


@pytest.mark.asyncio
async def test_wait_until_ready_task_fails_during_wait(
    mock_data_manager: LocationDataManager,
    mocker: MockerFixture,
) -> None:
    """Test case 6: Update task fails during the wait."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = False  # type: ignore[union-attr]

    async def mock_wait(
        waiters: set[asyncio.Future[Any]], **_kwargs: Any
    ) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
        update_task = mock_data_manager._update_task
        mock_data_manager._update_task.exception.return_value = ValueError(  # type: ignore[union-attr]
            "Task failed during wait"
        )
        done_set: set[asyncio.Future[Any]] = {cast(asyncio.Future[Any], update_task)}
        pending_set = waiters - done_set
        return done_set, pending_set

    mocker.patch("asyncio.wait", new=mock_wait)

    with pytest.raises(ValueError, match="Task failed during wait"):
        await mock_data_manager.wait_until_ready(timeout=1.0)


@pytest.mark.asyncio
async def test_wait_until_ready_task_done_unexpectedly_during_wait(
    mock_data_manager: LocationDataManager,
    mocker: MockerFixture,
) -> None:
    """Test case 7: Update task finishes ok during wait, but ready event not set."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = False  # type: ignore[union-attr]

    async def mock_wait(
        waiters: set[asyncio.Future[Any]], **_kwargs: Any
    ) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
        update_task = mock_data_manager._update_task
        mock_data_manager._update_task.exception.return_value = None  # type: ignore[union-attr]

        done_set: set[asyncio.Future[Any]] = {cast(asyncio.Future[Any], update_task)}
        pending_set = waiters - done_set
        return done_set, pending_set

    mocker.patch("asyncio.wait", new=mock_wait)

    with pytest.raises(RuntimeError, match="finished unexpectedly"):
        await mock_data_manager.wait_until_ready(timeout=1.0)


@pytest.mark.asyncio
async def test_wait_until_ready_timeout(
    mock_data_manager: LocationDataManager,
    mocker: MockerFixture,
) -> None:
    """Test case 8: Timeout occurs before ready or task completion."""
    mock_data_manager._ready_event.is_set.return_value = False  # type: ignore[attr-defined]
    mock_data_manager._update_task.done.return_value = False  # type: ignore[union-attr]

    async def mock_wait(
        waiters: set[asyncio.Future[Any]], **_kwargs: Any
    ) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
        return set(), waiters  # Return empty done, all pending

    mocker.patch("asyncio.wait", new=mock_wait)
    mocker.patch.object(
        mock_data_manager._ready_event, "wait", side_effect=asyncio.TimeoutError
    )  # type: ignore[attr-defined, unused-ignore]

    result = await mock_data_manager.wait_until_ready(timeout=0.1)
    assert result is False


@freeze_time("2025-05-04 10:15:00")
def test_current_info_retrieval(mock_data_manager: LocationDataManager) -> None:
    """Test retrieving the latest current observation using the fixture."""
    # mock_data_manager fixture handles config and basic setup.
    # Ensure currents_source is configured in the fixture or mock it if needed.
    # Assuming the fixture sets up a config with currents_source.

    # 1. Create Mock Feed Data
    timestamps = [
        datetime.datetime(2025, 5, 4, 10, 0, 0),
        datetime.datetime(2025, 5, 4, 10, 10, 0),
    ]
    velocities = [1.1, 1.3]  # Latest velocity
    mock_df = pd.DataFrame({"velocity": velocities}, index=pd.DatetimeIndex(timestamps))

    mock_currents_feed = MagicMock()
    mock_currents_feed.values = mock_df
    mock_currents_feed.is_expired = False  # Ensure feed is considered valid

    # 2. Manually inject the mock feed into the fixture's manager
    # The fixture should have already handled the _configure_currents_feed part.
    mock_data_manager._feeds["currents"] = mock_currents_feed

    # 3. Call the method on the fixture's manager
    result = mock_data_manager.get_current_flow_info()

    # 4. Assert results (same assertions as before)
    assert isinstance(result, CurrentInfo)
    assert result.timestamp == timestamps[-1]
    assert result.magnitude == velocities[-1]
    assert result.source_type == DataSourceType.OBSERVATION

    # 5. Assert optional fields are None for observations
    assert result.direction is None
    assert result.magnitude_pct is None
    assert result.state_description is None
