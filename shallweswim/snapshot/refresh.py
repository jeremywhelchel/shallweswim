"""Web-side shadow state: the loaded generation and its request-elected refresh.

Shadow mode changes nothing the web serves. `SnapshotState` holds one loaded
generation and the read-only per-location managers built from it, and an HTTP
middleware elects one request per check interval to bring it up to date. The
state is replaced, never mutated: a check that finds a new generation builds
every manager first and then publishes them with one assignment of an
immutable mapping, so a request reading `managers` sees one whole generation.
"""

import asyncio
import datetime
import logging
import time
from collections.abc import Mapping
from types import MappingProxyType

from shallweswim import config as config_lib
from shallweswim.snapshot.load import LoadedSnapshot, load_current
from shallweswim.snapshot.manager import SnapshotLocationManager
from shallweswim.snapshot.store import SnapshotStore

# How long a loaded generation is served before an arriving request is elected
# to check the pointer again.
CHECK_INTERVAL_SECONDS = 60.0

# The startup load is bounded so a slow or unreachable bucket cannot delay the
# instance. In shadow mode the instance then starts with no generation.
INITIAL_LOAD_TIMEOUT_SECONDS = 20.0

_NO_MANAGERS: Mapping[str, SnapshotLocationManager] = MappingProxyType({})


class SnapshotState:
    """One instance's loaded generation, refreshed on elected requests."""

    def __init__(self, store: SnapshotStore) -> None:
        """Hold an empty state over `store` until the first load succeeds."""
        self._store = store
        self._lock = asyncio.Lock()
        self._loaded: LoadedSnapshot | None = None
        self._last_check = time.monotonic()
        self.managers: Mapping[str, SnapshotLocationManager] = _NO_MANAGERS
        self.loaded_at: datetime.datetime | None = None

    @property
    def generation_id(self) -> str | None:
        """The loaded generation's id, or None while none is loaded."""
        return None if self._loaded is None else self._loaded.manifest.generation_id

    async def initial_load(self, timeout: float = INITIAL_LOAD_TIMEOUT_SECONDS) -> None:
        """Load the current generation once at startup, bounded by `timeout`.

        A failure or timeout leaves the instance with no generation and does
        not raise: in shadow mode nothing is served from it.
        """
        async with self._lock:
            await self._check(timeout)

    async def check_and_refresh(self) -> None:
        """Check the pointer if the interval elapsed and no check is running.

        Returns immediately when another request is already checking, so a
        burst of concurrent requests performs one check between them.
        """
        if time.monotonic() - self._last_check < CHECK_INTERVAL_SECONDS:
            return
        # The lock is acquired without suspending when it is free, so a caller
        # that sees it unheld is the one that takes it.
        if self._lock.locked():
            return
        async with self._lock:
            if time.monotonic() - self._last_check < CHECK_INTERVAL_SECONDS:
                return
            await self._check(None)

    async def _check(self, timeout: float | None) -> None:
        """Read the pointer and adopt a new generation, logging what it did.

        Every attempt, successful or not, schedules the next check a full
        interval later, so a failing bucket is not retried on every request.
        """
        started = time.monotonic()
        previous = self._loaded
        try:
            loaded = await self._load(previous, timeout)
            if loaded is None or loaded is previous:
                logging.debug(
                    f"[snapshot] generation unchanged: {self.generation_id or 'none'}"
                )
                return
            loaded_at = datetime.datetime.now(datetime.UTC)
            managers = self._build_managers(loaded, loaded_at)
        except Exception as error:
            # WARNING throughout shadow mode. At cutover the readiness-blocking
            # startup load becomes ERROR, because an instance that loaded no
            # generation then has nothing to serve; a failed refresh over an
            # already loaded generation stays WARNING and pages through the
            # bundle-age alert instead.
            logging.warning(
                f"[snapshot] load failed: {error}",
                extra=self._event("failed", started, age_seconds=None, record_count=0),
            )
            return
        finally:
            self._last_check = time.monotonic()

        self._loaded = loaded
        self.loaded_at = loaded_at
        self.managers = managers
        age_seconds = int((loaded_at - loaded.manifest.published_at).total_seconds())
        logging.info(
            f"[snapshot] loaded generation {loaded.manifest.generation_id}"
            f" ({loaded.objects_read} objects, {age_seconds}s behind)",
            extra=self._event(
                "success",
                started,
                age_seconds=age_seconds,
                record_count=loaded.objects_read,
            ),
        )

    async def _load(
        self, previous: LoadedSnapshot | None, timeout: float | None
    ) -> LoadedSnapshot | None:
        """Load the current generation, bounded by `timeout` when one is given."""
        if timeout is None:
            return await load_current(self._store, previous)
        return await asyncio.wait_for(load_current(self._store, previous), timeout)

    def _build_managers(
        self, loaded: LoadedSnapshot, loaded_at: datetime.datetime
    ) -> Mapping[str, SnapshotLocationManager]:
        """Build one read-only manager per published, configured location.

        A configured location the generation does not carry simply has no
        manager, as does a published location this build no longer configures.
        """
        return MappingProxyType(
            {
                code: SnapshotLocationManager(
                    config_lib.CONFIGS[code],
                    location,
                    loaded.frames[code],
                    loaded.plots[code],
                    loaded_at,
                )
                for code, location in loaded.manifest.locations.items()
                if code in config_lib.CONFIGS
            }
        )

    def _event(
        self,
        outcome: str,
        started: float,
        *,
        age_seconds: int | None,
        record_count: int,
    ) -> dict[str, object]:
        """The structured fields of one `snapshot` `load` event.

        `age_seconds` is present exactly when a generation was loaded, so a
        metric extractor reading it never has to distinguish a missing lag from
        a zero one. A failed load omits it, as the freshness event does for a
        feed that is absent.
        """
        fields: dict[str, object] = {
            "component": "snapshot",
            "operation": "load",
            "outcome": outcome,
            "generation_id": self.generation_id,
            "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
            "record_count": record_count,
        }
        if age_seconds is not None:
            fields["age_seconds"] = age_seconds
        return fields
