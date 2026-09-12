"""Provider-neutral conditional byte-object storage."""

from __future__ import annotations

import asyncio

# The first-class filesystem target is Unix; fcntl provides inter-process locks.
import fcntl
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from google.cloud import storage


@dataclass(frozen=True)
class StoredObject:
    """One coherent object value and its opaque adapter-owned CAS version."""

    data: bytes
    version: str


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
    """Concurrent in-memory object store for tests and local composition."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
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
            return _content_version(data)


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
