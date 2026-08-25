"""Persistent storage backend.

Provides a unified interface for reading/writing JSON and binary files.
When the ``GCS_BUCKET`` environment variable is set, files are stored
in Google Cloud Storage (for Cloud Run deployments). Otherwise, falls
back to local filesystem storage.

Environment:

* ``GCS_BUCKET`` — bucket name. Empty / unset means use LocalStorage.
* ``REQUIRE_SHARED_STORAGE`` — when truthy (``1``/``true``/``yes``/``on``),
  refuse to start unless GCS is actually usable: ``GCS_BUCKET`` must be
  set AND GCS initialization must succeed. Set this on any deploy with
  >1 instance (Cloud Run autoscaling, blue/green, multi-region) — local
  storage diverges per instance and corrupts the index silently.

Usage:
    from data_concierge.data_layer.storage import storage

    storage.write_json("verified_notebooks/index.json", data)
    data = storage.read_json("verified_notebooks/index.json")
    storage.write_bytes("verified_notebooks/verified/abc.ipynb", content)
    exists = storage.exists("verified_notebooks/index.json")
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

# Project root (4 levels up from this file: data_layer/storage.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# NOTE: GCS_BUCKET and REQUIRE_SHARED_STORAGE are read inside
# _create_storage() at call time, not captured here, so the live env
# (and pytest's ``monkeypatch.setenv``) is what actually decides the
# backend.


class LocalStorage:
    """Local filesystem storage backend."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write ``data`` to ``path`` atomically.

        A crash mid-write must never truncate or corrupt an existing
        state file (``roles.json``, ``github_settings.json``, chats,
        query logs — issue #94). We write to a temp file in the SAME
        directory (so ``os.replace`` is a same-filesystem rename, which
        is atomic), fsync it to durably flush the bytes, then atomically
        replace the destination. ``os.replace`` overwrites an existing
        target cross-platform (unlike ``Path.rename`` on Windows).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            # Don't leave a partial temp file behind on failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def read_json(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def write_json(self, key: str, data: dict[str, Any]) -> None:
        path = self._path(key)
        payload = json.dumps(data, indent=2).encode("utf-8")
        self._atomic_write(path, payload)

    def read_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.exists():
            return None
        return path.read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        self._atomic_write(path, data)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_keys(self, prefix: str, suffix: str = "") -> list[str]:
        """List keys under a prefix, optionally filtered by suffix."""
        base = self._path(prefix)
        if not base.exists():
            return []
        keys = []
        for p in base.iterdir():
            if p.is_file() and (not suffix or p.name.endswith(suffix)):
                keys.append(f"{prefix}/{p.name}")
        return keys

    def full_path(self, key: str) -> str:
        """Return the absolute local file path (for FileResponse etc.)."""
        return str(self._path(key))


class GCSStorage:
    """Google Cloud Storage backend."""

    def __init__(self, bucket_name: str) -> None:
        from google.cloud import storage as gcs

        self._client = gcs.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._bucket_name = bucket_name
        logger.info("GCS storage initialized", bucket=bucket_name)

    def read_json(self, key: str) -> dict[str, Any] | None:
        blob = self._bucket.blob(key)
        if not blob.exists():
            return None
        data = blob.download_as_text()
        return json.loads(data)

    def write_json(self, key: str, data: dict[str, Any]) -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")

    def read_bytes(self, key: str) -> bytes | None:
        blob = self._bucket.blob(key)
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        blob = self._bucket.blob(key)
        blob.upload_from_string(data)

    def exists(self, key: str) -> bool:
        return self._bucket.blob(key).exists()

    def delete(self, key: str) -> bool:
        blob = self._bucket.blob(key)
        if blob.exists():
            blob.delete()
            return True
        return False

    def list_keys(self, prefix: str, suffix: str = "") -> list[str]:
        """List keys under a prefix, optionally filtered by suffix."""
        blobs = self._client.list_blobs(self._bucket_name, prefix=prefix + "/")
        keys = []
        for blob in blobs:
            if not suffix or blob.name.endswith(suffix):
                keys.append(blob.name)
        return keys

    def full_path(self, key: str) -> str:
        """Return GCS URI (not a local path). For file serving, use read_bytes instead."""
        return f"gs://{self._bucket_name}/{key}"

    def download_to_tmp(self, key: str) -> str | None:
        """Download a file to a temporary local path for FileResponse."""
        import tempfile

        blob = self._bucket.blob(key)
        if not blob.exists():
            return None
        # Use key's filename as suffix
        suffix = Path(key).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        blob.download_to_filename(tmp.name)
        return tmp.name


def _probe_gcs_writable(backend: GCSStorage) -> None:
    """Verify the configured GCS bucket is actually usable.

    ``GCSStorage.__init__`` is lazy — it constructs a ``gcs.Client`` and
    a bucket handle, but performs NO network I/O. So a wrong bucket
    name, a service account without ``storage.objects.create`` /
    ``.get`` / ``.delete`` / ``.list``, or a network block all PASS
    the constructor and only blow up on the first real call. That
    defeats the ``REQUIRE_SHARED_STORAGE`` fail-fast promise
    (an earlier adversarial review, #2547).

    Probe = write a small temp object, read it back, list to confirm
    it's visible, delete it. This exercises ALL the IAM verbs we use
    at runtime — including ``storage.objects.list`` which the
    ``/notebooks`` endpoint depends on (without the list check, a
    deploy with create/get/delete but no list would pass startup and
    fail on first listing call).

    The probe key embeds both a millisecond timestamp and a UUID4
    suffix so concurrent multi-instance startups (autoscale-out,
    blue/green cutover) can never collide on the same object — without
    the UUID, two instances picking the same millisecond would race
    on the same key (instance A's delete could fire while instance B
    is mid-read, producing a false-positive startup failure on a
    healthy bucket).

    Best-effort cleanup if any step fails. Probe is opt-in (only the
    ``require_shared`` branch calls it) so single-instance deploys
    don't pay the startup latency.
    """
    import time
    import uuid

    probe_key = (
        f"_probe/require_shared_storage_"
        f"{int(time.time() * 1000)}_{uuid.uuid4().hex}.tmp"
    )
    probe_data = b"REQUIRE_SHARED_STORAGE startup probe"
    try:
        backend.write_bytes(probe_key, probe_data)
        read_back = backend.read_bytes(probe_key)
        if read_back != probe_data:
            raise RuntimeError(
                f"GCS probe wrote {len(probe_data)} bytes but read back "
                f"{0 if read_back is None else len(read_back)} bytes — "
                "bucket may be returning stale or corrupted data."
            )
        # Verify list permission. The /notebooks endpoint calls
        # list_keys, so a deploy that can write/read/delete but not
        # list would pass startup and fail on first user listing call.
        listed = backend.list_keys("_probe", suffix=".tmp")
        if probe_key not in listed:
            raise RuntimeError(
                f"GCS probe wrote {probe_key!r} but it did not appear "
                "in list_keys() results — the service account may lack "
                "storage.objects.list permission, or the bucket is "
                "returning stale list pages."
            )
        backend.delete(probe_key)
    except Exception:
        # Best-effort cleanup so a failed probe doesn't leave litter.
        try:
            backend.delete(probe_key)
        except Exception:
            pass
        raise


def _create_storage() -> LocalStorage | GCSStorage:
    """Create the appropriate storage backend.

    Behavior matrix:

    +-------------------+-----------+----------------+--------------------------+
    | REQUIRE_SHARED    | GCS_BUCKET| GCS init/probe | Result                   |
    +===================+===========+================+==========================+
    | unset / falsy     | unset     | n/a            | LocalStorage             |
    | unset / falsy     | set       | init OK        | GCSStorage (no probe)    |
    | unset / falsy     | set       | init fails     | LocalStorage (warn-only) |
    | truthy            | unset     | n/a            | RuntimeError at startup  |
    | truthy            | set       | init fails     | RuntimeError at startup  |
    | truthy            | set       | probe fails    | RuntimeError at startup  |
    | truthy            | set       | probe OK       | GCSStorage               |
    +-------------------+-----------+----------------+--------------------------+

    The ``REQUIRE_SHARED_STORAGE=true`` opt-in is for multi-instance
    deploys (Cloud Run with >1 instance, autoscaling, blue/green, etc.)
    where ``LocalStorage`` is FATAL: each instance gets its own
    ephemeral disk, the verified-notebook index diverges per instance,
    submissions land on one instance and approvals come from another,
    and reads return whatever instance answers. The previous silent
    fallback to LocalStorage on GCS init failure made this failure
    mode invisible — operators only noticed after the data was already
    inconsistent.

    When ``require_shared`` is on, we ALSO probe the bucket with a
    write/read/delete round-trip before returning — ``GCSStorage``
    construction is lazy and would otherwise pass startup for a
    bucket that doesn't exist or that the service account can't
    write to (an earlier adversarial review).

    Env vars are read at call time (not at module import) so tests
    can vary the behavior with ``monkeypatch.setenv``.
    """
    bucket = os.environ.get("GCS_BUCKET", "")
    require_shared = os.environ.get("REQUIRE_SHARED_STORAGE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if require_shared and not bucket:
        raise RuntimeError(
            "REQUIRE_SHARED_STORAGE=true but GCS_BUCKET is not set. "
            "Multi-instance deployments require shared storage; "
            "LocalStorage would diverge per instance. "
            "Either set GCS_BUCKET to a writable bucket, or unset "
            "REQUIRE_SHARED_STORAGE for a single-instance / local-dev "
            "deploy."
        )

    if bucket:
        # Step 1: construct the backend (lazy — no I/O).
        try:
            backend = GCSStorage(bucket)
        except Exception as e:
            if require_shared:
                raise RuntimeError(
                    f"REQUIRE_SHARED_STORAGE=true and GCS init failed "
                    f"(bucket={bucket!r}, error={e}). Refusing to fall "
                    "back to LocalStorage on a multi-instance deploy — "
                    "instances would silently diverge. Fix GCS access "
                    "(IAM, bucket name, network) and retry."
                ) from e
            logger.warning(
                "Failed to initialize GCS storage, falling back to local",
                error=str(e),
                bucket=bucket,
            )
            return LocalStorage(_PROJECT_ROOT)

        # Step 2: when require_shared, exercise the bucket end-to-end
        # so a wrong name / missing IAM doesn't slip through (the
        # original guard was constructor-only — see an earlier adversarial review).
        if require_shared:
            try:
                _probe_gcs_writable(backend)
            except Exception as e:
                raise RuntimeError(
                    f"REQUIRE_SHARED_STORAGE=true and GCS bucket probe "
                    f"failed (bucket={bucket!r}, error={e}). The bucket "
                    "may not exist, the service account may lack "
                    "storage.objects.create / .get / .list / .delete "
                    "permission, or networking may be blocked. Refusing "
                    "to fall back to LocalStorage on a multi-instance "
                    "deploy. Fix the GCS configuration and retry."
                ) from e

        return backend

    return LocalStorage(_PROJECT_ROOT)


# Singleton storage instance
storage = _create_storage()
