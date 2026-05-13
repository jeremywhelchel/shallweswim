"""Performance guardrails for request-path data lookups.

These tests use deterministic in-memory feed data. They are intentionally separate
from the unit suite because wall-clock thresholds can be noisy on shared runners.
Run with:

    uv run pytest tests/performance -v --run-performance
"""

import datetime
import json
import math
import os
import statistics
import time
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from shallweswim import config as config_lib
from shallweswim.api import register_routes
from shallweswim.core import feeds
from shallweswim.core.manager import LocationDataManager
from shallweswim.core.queries import prepare_current_prediction_frame
from shallweswim.types import TIDE_TYPE_CATEGORIES
from shallweswim.util import utc_now
from tests.conftest import TEST_CONFIG_FULL
from tests.helpers import create_test_app

pytestmark = pytest.mark.performance

PERFORMANCE_CONFIG = TEST_CONFIG_FULL.model_copy(
    update={
        "temp_source": TEST_CONFIG_FULL.temp_source.model_copy(
            update={"historic_enabled": False}
        )
        if TEST_CONFIG_FULL.temp_source is not None
        else None
    }
)
ENDPOINT_P95_LIMIT_MS = float(os.getenv("PERF_MAX_ENDPOINT_P95_MS", "25"))
MANAGER_P95_LIMIT_MS = float(os.getenv("PERF_MAX_MANAGER_P95_MS", "5"))
REPORT_PATH = Path(os.getenv("PERFORMANCE_RESULTS_PATH", "performance-results.json"))


def _now() -> datetime.datetime:
    """Return current local test time rounded to the minute."""
    return PERFORMANCE_CONFIG.local_now().replace(second=0, microsecond=0)


def _make_tides(now: datetime.datetime) -> pd.DataFrame:
    """Create enough tide predictions for current and shifted lookups."""
    tide_times = pd.date_range(
        now - datetime.timedelta(hours=36),
        now + datetime.timedelta(hours=36),
        freq="6h",
        name="time",
    )
    tide_types = ["low", "high"] * math.ceil(len(tide_times) / 2)
    predictions = np.sin(np.linspace(0, 2 * math.pi, len(tide_times)))
    return pd.DataFrame(
        {
            "prediction": predictions,
            "type": pd.Categorical(
                tide_types[: len(tide_times)], categories=TIDE_TYPE_CATEGORIES
            ),
        },
        index=tide_times,
    )


def _make_currents(now: datetime.datetime) -> pd.DataFrame:
    """Create minute-granularity current predictions around the current time."""
    current_times = pd.date_range(
        now - datetime.timedelta(hours=36),
        now + datetime.timedelta(hours=36),
        freq="1min",
        name="time",
    )
    samples = np.arange(len(current_times), dtype=float)
    velocities = 1.5 * np.sin(samples * math.pi / 360)
    return pd.DataFrame({"velocity": velocities}, index=current_times)


def _make_live_temps(now: datetime.datetime) -> pd.DataFrame:
    """Create recent live temperature readings."""
    temp_times = pd.date_range(
        now - datetime.timedelta(days=2),
        now,
        freq="1h",
        name="time",
    )
    return pd.DataFrame({"water_temp": [68.5] * len(temp_times)}, index=temp_times)


def _loaded_feed(data: pd.DataFrame) -> MagicMock:
    """Create a minimal loaded feed object for LocationDataManager tests."""
    feed = MagicMock()
    feed._data = data
    feed.values = data
    feed._fetch_timestamp = utc_now()
    feed.is_expired = False
    return feed


def _build_manager(process_pool: ProcessPoolExecutor) -> LocationDataManager:
    """Build a data manager with deterministic, already-loaded feed data."""
    now = _now()
    manager = LocationDataManager(
        PERFORMANCE_CONFIG,
        clients={"coops": MagicMock()},
        process_pool=process_pool,
    )
    manager._feeds[feeds.FEED_TIDES] = _loaded_feed(_make_tides(now))
    manager._feeds[feeds.FEED_CURRENTS] = _loaded_feed(_make_currents(now))
    manager._feeds[feeds.FEED_LIVE_TEMPS] = _loaded_feed(_make_live_temps(now))
    manager._feeds[feeds.FEED_HISTORIC_TEMPS] = None
    manager._precompute_current_predictions()
    return manager


@pytest.fixture
def performance_manager() -> Generator[LocationDataManager]:
    """Provide a loaded manager without starting background tasks."""
    process_pool = ProcessPoolExecutor(max_workers=1)
    try:
        yield _build_manager(process_pool)
    finally:
        process_pool.shutdown(wait=False)


@pytest.fixture
def performance_client(
    performance_manager: LocationDataManager,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient]:
    """Provide a TestClient backed by the loaded performance manager."""
    monkeypatch.setattr(
        config_lib,
        "get",
        lambda code: PERFORMANCE_CONFIG if code == PERFORMANCE_CONFIG.code else None,
    )

    app = create_test_app()
    app.state.data_managers = {PERFORMANCE_CONFIG.code: performance_manager}
    register_routes(app)

    with TestClient(app) as client:
        yield client


def _measure(
    action: Callable[[], Any],
    *,
    iterations: int = 40,
    warmups: int = 5,
) -> list[float]:
    """Measure action runtime in milliseconds after warmup calls."""
    for _ in range(warmups):
        action()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        action()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def _p95(samples: list[float]) -> float:
    """Return the 95th percentile from a non-empty sample set."""
    return sorted(samples)[math.ceil(len(samples) * 0.95) - 1]


def _record_result(name: str, samples: list[float], limit_ms: float) -> None:
    """Append a timing result to the local JSON performance report."""
    existing: list[dict[str, Any]] = (
        json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else []
    )

    p95_ms = _p95(samples)
    existing.append(
        {
            "name": name,
            "iterations": len(samples),
            "mean_ms": statistics.fmean(samples),
            "p95_ms": p95_ms,
            "max_ms": max(samples),
            "limit_ms": limit_ms,
        }
    )
    REPORT_PATH.write_text(json.dumps(existing, indent=2) + "\n")


def test_manager_current_prediction_lookup_stays_precomputed(
    performance_manager: LocationDataManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated current prediction lookups stay on the precomputed fast path."""
    calls = 0

    def counting_prepare(currents_data: pd.DataFrame) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return prepare_current_prediction_frame(currents_data)

    monkeypatch.setattr(
        "shallweswim.core.manager.queries.prepare_current_prediction_frame",
        counting_prepare,
    )

    # Clear the eager fixture precompute so this test proves one recomputation
    # followed by repeated cache reuse.
    performance_manager._current_prediction_frame = None
    performance_manager._current_prediction_source_timestamp = None
    performance_manager._current_prediction_source_data_id = None

    samples = _measure(performance_manager.predict_flow_at_time)
    _record_result("manager.predict_flow_at_time", samples, MANAGER_P95_LIMIT_MS)

    assert calls == 1
    assert _p95(samples) < MANAGER_P95_LIMIT_MS


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("/api/nyc/conditions", "api.conditions"),
        ("/api/nyc/currents", "api.currents"),
        ("/api/nyc/currents?shift=60", "api.currents.shift_60"),
    ],
)
def test_fast_user_facing_api_endpoints_stay_under_latency_guardrail(
    performance_client: TestClient,
    path: str,
    name: str,
) -> None:
    """Mocked fast-path endpoints should not grow request-time DataFrame work."""

    def fetch() -> None:
        response = performance_client.get(path)
        assert response.status_code == 200

    samples = _measure(fetch)
    _record_result(name, samples, ENDPOINT_P95_LIMIT_MS)

    assert _p95(samples) < ENDPOINT_P95_LIMIT_MS
