"""Content-addressed snapshot layout over the conditional object store.

Objects are immutable and named by their content, so writing one is
create-only and an identical key already present is reuse rather than a
conflict. The current pointer is the only object ever replaced, and only
conditionally, so promotion can never move it backward past a newer publisher.
Memory, filesystem, and GCS backends all come from the existing `ObjectStore`
implementations.
"""

import hashlib
from typing import Literal

from shallweswim.archive.store import ObjectStore, VersionConflictError
from shallweswim.snapshot.model import CurrentPointer, Manifest

# The bucket whose `published/` prefix a reader loads generations from. The web
# service sets it to enable shadow mode; the comparison command requires it. It
# is read-only and distinct from the archive write and hydration variables.
SNAPSHOT_READ_BUCKET_ENV_VAR = "SHALLWESWIM_SNAPSHOT_READ_BUCKET"

OBJECTS_PREFIX = "published/objects"
MANIFESTS_PREFIX = "published/manifests"
CURRENT_KEY = "published/current.json"


class PromotionConflictError(Exception):
    """The current pointer changed since the publisher observed it."""


def object_key(data: bytes, suffix: Literal["parquet", "svg"]) -> str:
    """Return the content-addressed key for one object's bytes."""
    return f"{OBJECTS_PREFIX}/sha256-{hashlib.sha256(data).hexdigest()}.{suffix}"


def manifest_key(generation_id: str) -> str:
    """Return the key holding one generation's manifest."""
    return f"{MANIFESTS_PREFIX}/{generation_id}.json"


class SnapshotStore:
    """Read and write snapshot objects, manifests, and the current pointer."""

    def __init__(self, store: ObjectStore) -> None:
        self._store = store

    async def read_current(self) -> tuple[CurrentPointer, str] | None:
        """Return the current pointer and its store version, or None if unset."""
        stored = await self._store.read(CURRENT_KEY)
        if stored is None:
            return None
        return CurrentPointer.model_validate_json(stored.data), stored.version

    async def read_manifest(self, key: str) -> Manifest | None:
        """Return the manifest stored at `key`, or None if it is absent."""
        stored = await self._store.read(key)
        if stored is None:
            return None
        return Manifest.model_validate_json(stored.data)

    async def read_object(self, key: str) -> bytes | None:
        """Return one object's bytes, or None if it is absent."""
        stored = await self._store.read(key)
        return None if stored is None else stored.data

    async def write_object(self, key: str, data: bytes) -> bool:
        """Create a content-addressed object; return False if it already existed."""
        try:
            await self._store.compare_and_swap(key, expected_version=None, data=data)
        except VersionConflictError:
            return False
        return True

    async def write_manifest(self, manifest: Manifest) -> tuple[str, bytes]:
        """Write a generation's manifest and return its key and bytes.

        Raises:
            VersionConflictError: If this generation id was already published,
                which two runs with the same run id would cause.
        """
        key = manifest_key(manifest.generation_id)
        data = manifest.model_dump_json().encode()
        await self._store.compare_and_swap(key, expected_version=None, data=data)
        return key, data

    async def promote(
        self, pointer: CurrentPointer, expected_version: str | None
    ) -> str:
        """Replace the current pointer only if it still has the observed version.

        Args:
            pointer: The pointer naming the newly written manifest.
            expected_version: The version observed by `read_current`, or None
                when no pointer existed.

        Returns:
            The pointer's new store version.

        Raises:
            PromotionConflictError: If another publisher promoted first.
        """
        try:
            return await self._store.compare_and_swap(
                CURRENT_KEY,
                expected_version=expected_version,
                data=pointer.model_dump_json().encode(),
            )
        except VersionConflictError as error:
            raise PromotionConflictError(
                f"current pointer changed while publishing {pointer.generation_id}"
            ) from error
