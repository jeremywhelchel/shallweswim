"""Utility functions for tests."""

import contextlib
import datetime
import json
import uuid
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, cast

import pytest
from fastapi import FastAPI

from shallweswim.archive.store import MemoryObjectStore
from shallweswim.clients.base import RetryableClientError
from shallweswim.core.manager import LocationDataManager
from shallweswim.core.serving import LocationServing
from shallweswim.snapshot.build import build_location_snapshot
from shallweswim.snapshot.manager import SnapshotLocationManager
from shallweswim.snapshot.model import Snapshot
from shallweswim.snapshot.publish import publish
from shallweswim.snapshot.refresh import SnapshotState
from shallweswim.snapshot.store import SnapshotStore


@contextlib.contextmanager
def skip_on_provider_outage(provider: str) -> Iterator[None]:
    """Skip the test when `provider` is down, and fail on anything else.

    A request that exhausts its retries on a timeout, a connection failure, or
    a `5xx` is the provider's outage: the client is not at fault and production
    monitoring already reports it, so the test skips with the error as its
    reason. A `429` is a retryable status too, but it is an answer about this
    caller, so it still fails, as does every non-retryable error (a `4xx`, a
    changed format, a validation error).

    Only tests of providers that serve no `test_required` location use this.

    Args:
        provider: The provider's name, for the skip reason.
    """
    try:
        yield
    except RetryableClientError as e:
        if e.status is not None and e.status < 500:
            raise
        pytest.skip(f"{provider} outage: {e}")


def install_managers(
    app: FastAPI, managers: Mapping[str, LocationServing]
) -> SnapshotState:
    """Serve `managers` as this app's loaded generation.

    The state is the real `SnapshotState`, over an empty store it never reads:
    only the managers are supplied, so a test can serve doubles without
    building and publishing frames. `generation_id`, `published_at`, and
    `loaded_at` stay None, as they are before any generation is loaded.

    Args:
        app: The application whose `state.snapshot` is replaced.
        managers: The per-location serving state the routes will resolve.

    Returns:
        The installed state.
    """
    state = SnapshotState(SnapshotStore(MemoryObjectStore()))
    # The state loads `SnapshotLocationManager`s; a test installs any serving
    # state the routes can call, including doubles.
    state.managers = cast(
        Mapping[str, SnapshotLocationManager], MappingProxyType(dict(managers))
    )
    app.state.snapshot = state
    return state


async def load_generation(
    app: FastAPI, snapshot: Snapshot, *, now: datetime.datetime | None = None
) -> SnapshotState:
    """Publish `snapshot` into the app's own store and load it for serving.

    This is the real path the deployed service takes: the job publishes a
    generation and the web instance loads it. The first call creates the store
    and the state; a later call publishes another generation into the same
    store and reloads, which is what an elected request's refresh does.

    Args:
        app: The application whose `state.snapshot` serves the generation.
        snapshot: The serving state to publish. A snapshot holding no data
            publishes nothing, leaving the previous generation in place.
        now: The publication instant; defaults to now.

    Returns:
        The app's serving state, having loaded whatever is current.
    """
    store = getattr(app.state, "snapshot_store", None)
    if store is None:
        store = SnapshotStore(MemoryObjectStore())
        app.state.snapshot_store = store
        app.state.snapshot = SnapshotState(store)
    await publish(
        store,
        snapshot,
        run_id=uuid.uuid4().hex,
        now=now or datetime.datetime.now(datetime.UTC),
    )
    state: SnapshotState = app.state.snapshot
    await state.initial_load()
    return state


async def serve_manager(
    app: FastAPI, manager: LocationDataManager, *, now: datetime.datetime | None = None
) -> SnapshotState:
    """Publish one fetching manager's current state and serve it.

    Frames a mocked stack fetched become a published generation, exactly as the
    job would publish them, so the routes under test read what they read in
    production.

    Args:
        app: The application whose `state.snapshot` serves the generation.
        manager: The manager whose feeds and plots are published.
        now: The publication instant; defaults to now.

    Returns:
        The app's serving state, having loaded whatever is current.
    """
    return await load_generation(
        app,
        Snapshot(locations={manager.config.code: build_location_snapshot(manager)}),
        now=now,
    )


def create_test_app(**kwargs: Any) -> FastAPI:
    """Create a FastAPI app configured for testing.

    FastAPI 0.130+ uses Pydantic's native JSON serialization which handles
    NaN -> null conversion automatically when response_model is set.

    Args:
        **kwargs: Additional arguments passed to FastAPI constructor.

    Returns:
        Configured FastAPI application.
    """
    return FastAPI(**kwargs)


def assert_json_serializable(obj: Any) -> None:
    """Assert that an object is JSON serializable.

    Args:
        obj: The object to check for JSON serializability.

    Raises:
        AssertionError: If the object is not JSON serializable.
    """
    try:
        json.dumps(obj)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"Object is not JSON serializable: {e}") from e
