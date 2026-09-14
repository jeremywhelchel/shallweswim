"""Provider-neutral conditional byte-object storage.

Every store the application reads or writes is named by one locator string,
resolved by `object_store`:

- a bare name, such as `shallweswim-archive`, is a GCS bucket
- a locator containing `/`, such as `./local-store` or `/tmp/swim`, is a
  `FilesystemObjectStore` rooted at that path
- the literal `memory` is one process-wide `MemoryObjectStore`

The environment variables that select a store (`SHALLWESWIM_ARCHIVE_BUCKET`,
`SHALLWESWIM_ARCHIVE_READ_BUCKET`, `SHALLWESWIM_SNAPSHOT_READ_BUCKET`) carry a
locator, so the local entry point can point every one of them at a directory or
at memory without any code knowing it is not a bucket.
"""

from __future__ import annotations

import asyncio
import datetime

# The first-class filesystem target is Unix; fcntl provides inter-process locks.
import fcntl
import hashlib
import os
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from google.cloud import storage


# The locator naming the process-wide in-memory store.
MEMORY_LOCATOR = "memory"


@dataclass(frozen=True)
class StoredObject:
    """One coherent object value and its opaque adapter-owned CAS version."""

    data: bytes
    version: str


@dataclass(frozen=True)
class ListedObject:
    """One key a listing found, and when the store says it was created."""

    key: str
    created_at: datetime.datetime


class VersionConflictError(Exception):
    """The object no longer has the version expected by the caller."""


class ObjectStore(Protocol):
    """Read and conditionally replace named byte objects.

    A create-only write whose conflict is an identical content-addressed object
    may be treated as success by the caller.
    """

    async def read(self, key: str) -> StoredObject | None:
        """Return one coherent object and version, or None when absent."""
        ...

    async def compare_and_swap(
        self,
        key: str,
        *,
        expected_version: str | None,
        data: bytes,
    ) -> str:
        """Create if absent or replace only the expected current version."""
        ...

    async def list(self, prefix: str) -> list[ListedObject]:
        """Return every object under `prefix` with its creation time."""
        ...

    async def delete(self, key: str) -> None:
        """Remove one object; an absent key is not an error."""
        ...


def _validate_key(key: str) -> PurePosixPath:
    path = PurePosixPath(key)
    if (
        not key
        or path.is_absolute()
        or str(path) != key
        or "\\" in key
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Invalid object key: {key!r}")
    return path


def _content_version(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MemoryObjectStore:
    """Concurrent in-memory object store for tests and local composition.

    `_created` records when each key was last written, which is this store's
    answer to a listing's creation time, as a replaced GCS object's
    `time_created` and a rewritten file's modification time both are. A test
    that needs an aged object overwrites the entry for its key.
    """

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._created: dict[str, datetime.datetime] = {}
        self._lock = asyncio.Lock()

    async def read(self, key: str) -> StoredObject | None:
        _validate_key(key)
        async with self._lock:
            data = self._objects.get(key)
            if data is None:
                return None
            return StoredObject(data=data, version=_content_version(data))

    async def compare_and_swap(
        self,
        key: str,
        *,
        expected_version: str | None,
        data: bytes,
    ) -> str:
        _validate_key(key)
        async with self._lock:
            current = self._objects.get(key)
            current_version = None if current is None else _content_version(current)
            if current_version != expected_version:
                raise VersionConflictError(key)
            self._objects[key] = data
            self._created[key] = datetime.datetime.now(datetime.UTC)
            return _content_version(data)

    async def list(self, prefix: str) -> list[ListedObject]:
        _validate_key(prefix)
        async with self._lock:
            return [
                ListedObject(key=key, created_at=self._created[key])
                for key in sorted(self._objects)
                if key.startswith(prefix)
            ]

    async def delete(self, key: str) -> None:
        _validate_key(key)
        async with self._lock:
            self._objects.pop(key, None)
            self._created.pop(key, None)


class FilesystemObjectStore:
    """Atomic filesystem object store safe for concurrent local processes.

    Versions are content hashes. This deliberately permits benign ABA: CAS may
    succeed if an object changed and then returned to byte-identical content.
    Archive merges are state-based and idempotent, so identical current content
    is the same valid merge base. Do not replace this with mtime tokens.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock_root = self._root / ".locks"

    def _path(self, key: str) -> Path:
        return self._root.joinpath(*_validate_key(key).parts)

    def _lock_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._lock_root / f"{digest}.lock"

    async def read(self, key: str) -> StoredObject | None:
        return await asyncio.to_thread(self._read_sync, key)

    def _read_sync(self, key: str) -> StoredObject | None:
        path = self._path(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        return StoredObject(data=data, version=_content_version(data))

    async def compare_and_swap(
        self,
        key: str,
        *,
        expected_version: str | None,
        data: bytes,
    ) -> str:
        return await asyncio.to_thread(
            self._compare_and_swap_sync, key, expected_version, data
        )

    def _compare_and_swap_sync(
        self, key: str, expected_version: str | None, data: bytes
    ) -> str:
        path = self._path(key)
        lock_path = self._lock_path(key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            current = self._read_sync(key)
            current_version = None if current is None else current.version
            if current_version != expected_version:
                raise VersionConflictError(key)

            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=f".{path.name}.", delete=False
                ) as temporary:
                    temporary.write(data)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_path = Path(temporary.name)
                os.replace(temporary_path, path)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            return _content_version(data)

    async def list(self, prefix: str) -> list[ListedObject]:
        return await asyncio.to_thread(self._list_sync, prefix)

    def _list_sync(self, prefix: str) -> list[ListedObject]:
        """List one directory tree, dot-files excluded.

        The lock directory and the partially written temporaries CAS creates
        are both dot-named, and neither is an object, so a listing skips every
        name that starts with a dot. A prefix naming no directory lists
        nothing, as an empty bucket prefix does.
        """
        base = self._root.joinpath(*_validate_key(prefix).parts)
        if not base.is_dir():
            return []
        listed = []
        for path in sorted(base.rglob("*")):
            if path.name.startswith(".") or not path.is_file():
                continue
            listed.append(
                ListedObject(
                    key=str(PurePosixPath(*path.relative_to(self._root).parts)),
                    created_at=datetime.datetime.fromtimestamp(
                        path.stat().st_mtime, datetime.UTC
                    ),
                )
            )
        return listed

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._delete_sync, key)

    def _delete_sync(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class GcsObjectStore:
    """GCS adapter using generation preconditions without event-loop blocking."""

    _MAX_COHERENT_READ_ATTEMPTS = 3

    def __init__(
        self,
        bucket_name: str,
        *,
        client: storage.Client | None = None,
    ) -> None:
        if client is None:
            from google.cloud import storage

            client = storage.Client()
        self._client = client
        self._bucket = self._client.bucket(bucket_name)

    async def read(self, key: str) -> StoredObject | None:
        _validate_key(key)
        return await asyncio.to_thread(self._read_sync, key)

    def _read_sync(self, key: str) -> StoredObject | None:
        from google.api_core import exceptions as google_exceptions

        for _attempt in range(self._MAX_COHERENT_READ_ATTEMPTS):
            blob = self._bucket.get_blob(key)
            if blob is None:
                return None
            generation = blob.generation
            if generation is None:
                raise RuntimeError(f"GCS object has no generation: {key}")
            try:
                data = blob.download_as_bytes(if_generation_match=generation)
            except google_exceptions.NotFound:
                continue
            except google_exceptions.PreconditionFailed:
                continue
            return StoredObject(data=data, version=str(generation))
        raise RuntimeError(f"GCS object changed during repeated reads: {key}")

    async def compare_and_swap(
        self,
        key: str,
        *,
        expected_version: str | None,
        data: bytes,
    ) -> str:
        _validate_key(key)
        return await asyncio.to_thread(
            self._compare_and_swap_sync, key, expected_version, data
        )

    def _compare_and_swap_sync(
        self, key: str, expected_version: str | None, data: bytes
    ) -> str:
        from google.api_core import exceptions as google_exceptions

        blob = self._bucket.blob(key)
        generation_match = 0 if expected_version is None else int(expected_version)
        try:
            blob.upload_from_string(data, if_generation_match=generation_match)
        except google_exceptions.PreconditionFailed as error:
            raise VersionConflictError(key) from error
        if blob.generation is None:
            raise RuntimeError(f"GCS write returned no generation: {key}")
        return str(blob.generation)

    async def list(self, prefix: str) -> list[ListedObject]:
        _validate_key(prefix)
        return await asyncio.to_thread(self._list_sync, prefix)

    def _list_sync(self, prefix: str) -> list[ListedObject]:
        listed = []
        for blob in self._bucket.list_blobs(prefix=prefix):
            if blob.time_created is None:
                raise RuntimeError(f"GCS object has no creation time: {blob.name}")
            listed.append(
                ListedObject(
                    key=blob.name,
                    created_at=blob.time_created.astimezone(datetime.UTC),
                )
            )
        return listed

    async def delete(self, key: str) -> None:
        _validate_key(key)
        await asyncio.to_thread(self._delete_sync, key)

    def _delete_sync(self, key: str) -> None:
        from google.api_core import exceptions as google_exceptions

        try:
            self._bucket.blob(key).delete()
        except google_exceptions.NotFound:
            # Another sweep, or a retry of this one, already removed it.
            pass


@cache
def gcs_store(bucket: str) -> GcsObjectStore:
    """Reuse the GCS client and connection pool for each bucket in this process.

    Capture writes through this store and local hydration reads through it, so
    one process holds at most one client per bucket.
    """
    return GcsObjectStore(bucket)


@cache
def memory_store() -> MemoryObjectStore:
    """Return the one in-memory store this process shares.

    The local entry point writes and reads the same store from one process, so
    the `memory` locator must resolve to a single instance rather than a new
    empty store per call.
    """
    return MemoryObjectStore()


def object_store(locator: str) -> ObjectStore:
    """Resolve one store locator to the store it names.

    Args:
        locator: `memory` for the process-wide in-memory store, a path
            containing `/` for a filesystem store rooted there, or a bare name
            for a GCS bucket.

    Returns:
        The store the locator names. GCS stores and the memory store are cached
        per process; a filesystem store holds no connection to reuse.
    """
    if locator == MEMORY_LOCATOR:
        return memory_store()
    if "/" in locator:
        return FilesystemObjectStore(locator)
    return gcs_store(locator)
