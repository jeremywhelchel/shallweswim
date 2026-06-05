"""Tests for USGS NWIS API client."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pandas as pd
import pytest

from shallweswim.clients.base import (
    ClientConnectionError,
    RetryableClientError,
    StationUnavailableError,
)
from shallweswim.clients.nwis import (
    NWIS_PAGE_LIMIT,
    NwisApi,
    NwisApiError,
    NwisDataError,
)


def create_mock_nwis_data(parameter_cd: str = "00010") -> pd.DataFrame:
    """Create raw NWIS data in the legacy-compatible shape."""
    timestamps = pd.DatetimeIndex(
        [
            "2025-04-19 14:00:00",
            "2025-04-19 15:00:00",
        ],
        tz="UTC",
    )
    temp_column = f"{parameter_cd}_00011"
    values = [15.5, 16.2] if parameter_cd == "00010" else [59.9, 61.2]
    return pd.DataFrame(
        {
            "site_no": "03292494",
            temp_column: values,
        },
        index=timestamps,
    )


def create_continuous_payload(
    *,
    parameter_cd: str = "00010",
    values: list[str] | None = None,
    next_href: str | None = None,
) -> dict[str, object]:
    """Create a minimal USGS continuous-values FeatureCollection."""
    values = values or ["15.5", "16.2"]
    features = []
    for idx, value in enumerate(values):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": f"obs-{idx}",
                    "time_series_id": "series-id",
                    "monitoring_location_id": "USGS-03292494",
                    "parameter_code": parameter_cd,
                    "statistic_id": "00011",
                    "time": f"2025-04-19T{14 + idx:02d}:00:00+00:00",
                    "value": value,
                    "unit_of_measure": "degC",
                    "approval_status": "Provisional",
                    "qualifier": None,
                    "last_modified": "2025-04-19T14:05:00+00:00",
                },
                "id": f"obs-{idx}",
                "geometry": None,
            }
        )

    links = []
    if next_href is not None:
        links.append({"rel": "next", "href": next_href})
    return {
        "type": "FeatureCollection",
        "features": features,
        "numberReturned": len(features),
        "links": links,
    }


@pytest.fixture
def mock_session() -> MagicMock:
    """Provides a mock aiohttp ClientSession."""
    return MagicMock(spec=aiohttp.ClientSession)


@pytest.fixture
def nwis_client(mock_session: MagicMock) -> NwisApi:
    """Provides an instance of NwisApi with a mock session."""
    return NwisApi(session=mock_session)


@pytest.mark.asyncio
async def test_temperature_success(nwis_client: NwisApi) -> None:
    """Test successful temperature fetch."""
    with patch.object(
        nwis_client, "_fetch_json_pages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = [create_continuous_payload(parameter_cd="00010")]

        df = await nwis_client.temperature(
            site_no="03292494",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            timezone="America/New_York",
            location_code="sdf",
        )

    assert len(df) == 2
    assert "water_temp" in df.columns
    assert round(df["water_temp"].iloc[0], 1) == 59.9
    assert round(df["water_temp"].iloc[1], 1) == 61.2


@pytest.mark.asyncio
async def test_temperature_fetch_uses_local_day_request_bounds(
    nwis_client: NwisApi,
) -> None:
    """temperature() forwards timezone so API requests cover local days."""
    with patch.object(
        nwis_client, "_fetch_json_pages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = [create_continuous_payload(parameter_cd="00011")]

        await nwis_client.temperature(
            site_no="03292494",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 20),
            timezone="America/New_York",
            location_code="sdf",
            parameter_cd="00011",
        )

    assert (
        mock_fetch.await_args.kwargs["params"]["time"]
        == "2025-04-19T04:00:00Z/2025-04-21T03:59:59Z"
    )


@pytest.mark.asyncio
async def test_temperature_celsius_param(nwis_client: NwisApi) -> None:
    """Test temperature fetch with parameter 00010 (Celsius)."""
    with patch.object(
        nwis_client, "_fetch_json_pages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = [create_continuous_payload(parameter_cd="00010")]

        df = await nwis_client.temperature(
            site_no="03292494",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            timezone="America/New_York",
            location_code="sdf",
            parameter_cd="00010",
        )

    assert len(df) == 2
    assert "water_temp" in df.columns
    assert round(df["water_temp"].iloc[0], 1) == 59.9
    assert round(df["water_temp"].iloc[1], 1) == 61.2


@pytest.mark.asyncio
async def test_temperature_fahrenheit_param(nwis_client: NwisApi) -> None:
    """Test temperature fetch with parameter 00011 (Fahrenheit)."""
    with patch.object(
        nwis_client, "_fetch_json_pages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = [
            create_continuous_payload(parameter_cd="00011", values=["59.9", "61.2"])
        ]

        df = await nwis_client.temperature(
            site_no="03292494",
            begin_date=datetime.date(2025, 4, 19),
            end_date=datetime.date(2025, 4, 19),
            timezone="America/New_York",
            location_code="sdf",
            parameter_cd="00011",
        )

    assert len(df) == 2
    assert "water_temp" in df.columns
    assert round(df["water_temp"].iloc[0], 1) == 59.9
    assert round(df["water_temp"].iloc[1], 1) == 61.2


@pytest.mark.asyncio
async def test_currents_success(nwis_client: NwisApi) -> None:
    """Test successful current velocity fetch."""
    with patch.object(
        nwis_client, "_fetch_json_pages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = [
            create_continuous_payload(parameter_cd="72255", values=["3.44", "3.43"])
        ]

        df = await nwis_client.currents(
            site_no="03292494",
            parameter_cd="72255",
            timezone="America/New_York",
            location_code="sdf",
        )

    assert len(df) == 2
    assert "velocity_fps" in df.columns
    assert df["velocity_fps"].tolist() == [3.44, 3.43]


@pytest.mark.asyncio
async def test_api_error(nwis_client: NwisApi) -> None:
    """Test handling of connection errors after retries fail."""
    with patch.object(
        nwis_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request_with_retry:
        mock_request_with_retry.side_effect = ClientConnectionError(
            "Simulated connection error after retries"
        )

        with pytest.raises(
            ClientConnectionError, match="Simulated connection error after retries"
        ):
            await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
            )


@pytest.mark.asyncio
async def test_execute_request_retries_aiohttp_errors(
    nwis_client: NwisApi,
) -> None:
    """Protocol/decode failures from aiohttp requests are transient."""
    with patch.object(
        nwis_client, "_fetch_json_pages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.side_effect = aiohttp.ClientPayloadError("bad gzip")

        with pytest.raises(RetryableClientError, match="ClientPayloadError"):
            await nwis_client._execute_request(
                sites="01463500",
                service="iv",
                parameterCd=["00010"],
                start="2025-04-19",
                end="2025-04-20",
                location_code="test",
            )


@pytest.mark.asyncio
async def test_fetch_json_page_retries_service_unavailable(
    nwis_client: NwisApi,
) -> None:
    """Retryable USGS HTTP statuses are mapped to RetryableClientError."""
    response = AsyncMock()
    response.status = 503
    response.__aenter__.return_value = response
    nwis_client._session.get.return_value = response

    with pytest.raises(RetryableClientError, match="HTTP 503"):
        await nwis_client._fetch_json_page(
            url="https://example.test",
            params={},
            site_no="01463500",
            location_code="test",
        )


@pytest.mark.asyncio
async def test_execute_request_maps_empty_features_to_station_unavailable(
    nwis_client: NwisApi,
) -> None:
    """Empty USGS FeatureCollections are expected no-data conditions."""
    with patch.object(
        nwis_client, "_fetch_json_pages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = [
            {
                "type": "FeatureCollection",
                "features": [],
                "numberReturned": 0,
                "links": [],
            }
        ]

        with pytest.raises(StationUnavailableError, match="returned no data"):
            await nwis_client._execute_request(
                sites="01463500",
                service="iv",
                parameterCd=["00010"],
                start="2025-04-19",
                end="2025-04-20",
                location_code="test",
            )


@pytest.mark.asyncio
async def test_execute_request_keeps_unexpected_schema_terminal(
    nwis_client: NwisApi,
) -> None:
    """Malformed USGS JSON stays a terminal data error."""
    with patch.object(
        nwis_client, "_fetch_json_pages", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = [{"type": "FeatureCollection"}]

        with pytest.raises(NwisDataError, match="missing features list"):
            await nwis_client._execute_request(
                sites="01463500",
                service="iv",
                parameterCd=["00010"],
                start="2025-04-19",
                end="2025-04-20",
                location_code="test",
            )


@pytest.mark.asyncio
async def test_fetch_json_pages_follows_next_links(nwis_client: NwisApi) -> None:
    """USGS pagination links are followed until no next page remains."""
    page_one = create_continuous_payload(next_href="https://example.test/next")
    page_two = create_continuous_payload(values=["17.1"], next_href=None)

    with patch.object(
        nwis_client, "_fetch_json_page", new_callable=AsyncMock
    ) as mock_fetch_page:
        mock_fetch_page.side_effect = [page_one, page_two]

        payloads = await nwis_client._fetch_json_pages(
            url="https://example.test/first",
            params={"limit": NWIS_PAGE_LIMIT},
            site_no="03292494",
            location_code="sdf",
        )

    assert payloads == [page_one, page_two]
    assert mock_fetch_page.await_args_list[0].kwargs["params"] == {
        "limit": NWIS_PAGE_LIMIT
    }
    assert (
        mock_fetch_page.await_args_list[1].kwargs["url"] == "https://example.test/next"
    )
    assert mock_fetch_page.await_args_list[1].kwargs["params"] is None


@pytest.mark.asyncio
async def test_fetch_json_pages_rejects_repeated_next_link(
    nwis_client: NwisApi,
) -> None:
    """Repeated USGS next links should not loop forever."""
    repeated_page = create_continuous_payload(next_href="https://example.test/repeat")

    with patch.object(
        nwis_client, "_fetch_json_page", new_callable=AsyncMock
    ) as mock_fetch_page:
        mock_fetch_page.side_effect = [repeated_page, repeated_page]

        with pytest.raises(NwisDataError, match="pagination loop"):
            await nwis_client._fetch_json_pages(
                url="https://example.test/repeat",
                params={"limit": NWIS_PAGE_LIMIT},
                site_no="03292494",
                location_code="sdf",
            )


def test_continuous_request_params() -> None:
    """The direct NWIS request targets the modern continuous-values endpoint."""
    params = NwisApi._continuous_request_params(
        site_no="03292494",
        parameter_cd="00011",
        start="2025-04-19",
        end="2025-04-20",
        timezone="UTC",
    )

    assert params["monitoring_location_id"] == "USGS-03292494"
    assert params["parameter_code"] == "00011"
    assert params["time"] == "2025-04-19T00:00:00Z/2025-04-20T23:59:59Z"
    assert params["limit"] == NWIS_PAGE_LIMIT


def test_continuous_request_params_use_local_day_boundaries() -> None:
    """Requested date ranges are local station days converted to UTC."""
    params = NwisApi._continuous_request_params(
        site_no="03292494",
        parameter_cd="00011",
        start="2025-04-19",
        end="2025-04-20",
        timezone="America/New_York",
    )

    assert params["time"] == "2025-04-19T04:00:00Z/2025-04-21T03:59:59Z"


def test_continuous_request_params_use_local_historical_year_boundaries() -> None:
    """Full historical years are requested as full local calendar years."""
    params = NwisApi._continuous_request_params(
        site_no="03292494",
        parameter_cd="00011",
        start="2025-01-01",
        end="2025-12-31",
        timezone="America/New_York",
    )

    assert params["time"] == "2025-01-01T05:00:00Z/2026-01-01T04:59:59Z"


def test_parse_continuous_payloads_filters_parameter_and_statistic() -> None:
    """Only matching instantaneous observations become raw NWIS data."""
    payload = create_continuous_payload(parameter_cd="00010", values=["15.5"])
    payload["features"].extend(
        [
            {
                "type": "Feature",
                "properties": {
                    "parameter_code": "00010",
                    "statistic_id": "00003",
                    "time": "2025-04-19T16:00:00+00:00",
                    "value": "99.9",
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "parameter_code": "00060",
                    "statistic_id": "00011",
                    "time": "2025-04-19T17:00:00+00:00",
                    "value": "1000",
                },
            },
        ]
    )

    df = NwisApi._parse_continuous_payloads(
        payloads=[payload],
        site_no="03292494",
        parameter_cd="00010",
    )

    assert len(df) == 1
    assert "00010_00011" in df.columns
    assert df["00010_00011"].iloc[0] == 15.5


def test_parse_continuous_payloads_rejects_missing_timestamps() -> None:
    """Matching observations must include parseable timestamp strings."""
    payload = create_continuous_payload(parameter_cd="00010", values=["15.5"])
    payload["features"][0]["properties"].pop("time")

    with pytest.raises(NwisDataError, match="missing timestamp"):
        NwisApi._parse_continuous_payloads(
            payloads=[payload],
            site_no="03292494",
            parameter_cd="00010",
        )


def test_parse_continuous_payloads_rejects_all_invalid_values() -> None:
    """Matching observations with no numeric values indicate bad data."""
    payload = create_continuous_payload(parameter_cd="00010", values=["not-a-number"])

    with pytest.raises(NwisDataError, match="no parseable values"):
        NwisApi._parse_continuous_payloads(
            payloads=[payload],
            site_no="03292494",
            parameter_cd="00010",
        )


@pytest.mark.asyncio
async def test_data_error(nwis_client: NwisApi) -> None:
    """Test handling of terminal NWIS data errors."""
    with patch.object(
        nwis_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request_with_retry:
        mock_request_with_retry.side_effect = NwisDataError(
            "NWIS response for site 03292494 missing features list"
        )

        with pytest.raises(NwisDataError, match="NWIS response for site 03292494"):
            await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
            )


@pytest.mark.asyncio
async def test_missing_temp_column(nwis_client: NwisApi) -> None:
    df = pd.DataFrame(
        {
            "00060_00011": [
                20.5,
                21.2,
            ],
        },
        index=pd.DatetimeIndex(
            [
                "2025-04-19 10:00",
                "2025-04-19 16:00",
            ]
        ),
    )

    with patch.object(
        nwis_client, "request_with_retry", new_callable=AsyncMock
    ) as mock_request_with_retry:
        mock_request_with_retry.return_value = df

        expected_message = r"No water temperature data \(parameter 00010\) available for NWIS site 03292494"
        with pytest.raises(NwisDataError, match=expected_message):
            await nwis_client.temperature(
                site_no="03292494",
                begin_date=datetime.date(2025, 4, 19),
                end_date=datetime.date(2025, 4, 19),
                timezone="America/New_York",
                location_code="sdf",
            )


@pytest.mark.asyncio
async def test_fix_time(nwis_client: NwisApi) -> None:
    """Test the _fix_time method."""
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-04-19 14:00:00", tz="UTC"),
            pd.Timestamp("2025-04-19 20:00:00", tz="UTC"),
        ]
    )
    df = pd.DataFrame({"water_temp": [59.9, 61.2]}, index=index)

    result_df = nwis_client._fix_time(df, "America/New_York")

    expected_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-04-19 10:00:00"),
            pd.Timestamp("2025-04-19 16:00:00"),
        ],
        name="time",
    )
    pd.testing.assert_index_equal(result_df.index, expected_index)


@pytest.mark.asyncio
async def test_fix_time_rejects_naive_timestamps(nwis_client: NwisApi) -> None:
    """_fix_time requires timestamps that still carry source timezone info."""
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-04-19 14:00:00"),
            pd.Timestamp("2025-04-19 20:00:00"),
        ]
    )
    df = pd.DataFrame({"water_temp": [59.9, 61.2]}, index=index)

    with pytest.raises(NwisApiError, match="NWIS timestamps must be timezone-aware"):
        nwis_client._fix_time(df, "America/New_York")
