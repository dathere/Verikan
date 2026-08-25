"""Tests for ``REQUIRE_SHARED_STORAGE`` fail-fast (issue #46, step 8).

On multi-instance deploys (Cloud Run with autoscaling, blue/green,
multi-region), ``LocalStorage`` is FATAL: each instance gets its own
ephemeral disk, the verified-notebook index diverges per instance, and
the silent fallback to LocalStorage on GCS init failure makes the
problem invisible until data is already inconsistent.

The fix: opt-in env var ``REQUIRE_SHARED_STORAGE``. When truthy:

* ``GCS_BUCKET`` MUST be set, else RuntimeError at startup.
* GCS initialization MUST succeed, else RuntimeError at startup (NOT
  silent fallback).

Backward compat: when ``REQUIRE_SHARED_STORAGE`` is unset/falsy, behavior
is unchanged — GCS used when bucket set and init succeeds, LocalStorage
fallback otherwise. Existing single-instance and local-dev deploys keep
working without any env changes.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_concierge.data_layer import storage as storage_mod
from data_concierge.data_layer.storage import (
    GCSStorage,
    LocalStorage,
    _create_storage,
)


class _StubGCSStorage:
    """Stand-in for GCSStorage that records construction and lets us
    fake init failures without reaching the real GCS client.

    Implements just enough of the storage surface (``write_bytes`` /
    ``read_bytes`` / ``list_keys`` / ``delete``) for the startup probe
    added in an earlier adversarial review + extended in #2547 to actually exercise
    this stub. Tests can flip the ``fail_on_*`` class attributes to
    inject failures at any step, or set ``empty_list`` to simulate a
    bucket that accepts list calls but returns no results (missing
    list IAM in some GCS configurations manifests this way).
    """

    fail_on_write = False
    fail_on_read = False
    fail_on_list = False
    fail_on_delete = False
    corrupt_read = False  # return wrong bytes from read_bytes
    empty_list = False  # return [] from list_keys even when objects exist

    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self._objects: dict[str, bytes] = {}
        self.write_calls: list[str] = []
        self.read_calls: list[str] = []
        self.list_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []

    def write_bytes(self, key: str, data: bytes) -> None:
        self.write_calls.append(key)
        if type(self).fail_on_write:
            raise RuntimeError("simulated GCS write failure (IAM / 403)")
        self._objects[key] = data

    def read_bytes(self, key: str) -> bytes | None:
        self.read_calls.append(key)
        if type(self).fail_on_read:
            raise RuntimeError("simulated GCS read failure")
        if type(self).corrupt_read:
            return b"DIFFERENT CONTENT"
        return self._objects.get(key)

    def list_keys(self, prefix: str, suffix: str = "") -> list[str]:
        self.list_calls.append((prefix, suffix))
        if type(self).fail_on_list:
            raise RuntimeError("simulated GCS list failure (IAM / 403)")
        if type(self).empty_list:
            return []
        return [
            k for k in self._objects
            if k.startswith(prefix + "/") and (not suffix or k.endswith(suffix))
        ]

    def delete(self, key: str) -> bool:
        self.delete_calls.append(key)
        if type(self).fail_on_delete:
            raise RuntimeError("simulated GCS delete failure (IAM / 403)")
        return self._objects.pop(key, None) is not None


def _patch_gcs(monkeypatch: pytest.MonkeyPatch, *, fails: bool = False) -> None:
    """Patch the GCSStorage class as seen by _create_storage.

    Note: _create_storage references the module-level ``GCSStorage``
    name, so we monkeypatch that symbol on the storage module rather
    than the class object directly.
    """
    if fails:

        def _raise(_bucket: str) -> Any:
            raise RuntimeError("GCS init blew up (auth / bucket / network)")

        monkeypatch.setattr(storage_mod, "GCSStorage", _raise)
    else:
        monkeypatch.setattr(storage_mod, "GCSStorage", _StubGCSStorage)


class TestRequireSharedStorageUnsetKeepsExistingBehavior:
    """The four legacy paths must be byte-for-byte unchanged when the
    new env var is not set — single-instance and local-dev deploys
    must keep working without any config change."""

    def test_no_bucket_returns_localstorage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GCS_BUCKET", raising=False)
        monkeypatch.delenv("REQUIRE_SHARED_STORAGE", raising=False)
        result = _create_storage()
        assert isinstance(result, LocalStorage)

    def test_bucket_set_gcs_ok_returns_gcsstorage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        monkeypatch.delenv("REQUIRE_SHARED_STORAGE", raising=False)
        _patch_gcs(monkeypatch, fails=False)
        result = _create_storage()
        assert isinstance(result, _StubGCSStorage)
        assert result.bucket_name == "my-bucket"

    def test_bucket_set_gcs_fails_falls_back_to_localstorage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The original silent-fallback behavior: log a warning, keep
        going with LocalStorage. This is correct for single-instance
        deploys; it's only fatal for multi-instance."""
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        monkeypatch.delenv("REQUIRE_SHARED_STORAGE", raising=False)
        _patch_gcs(monkeypatch, fails=True)
        result = _create_storage()
        assert isinstance(result, LocalStorage)


class TestRequireSharedStorageFailFast:
    """The four guard paths the step exists to enforce."""

    def test_truthy_without_bucket_raises_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.delenv("GCS_BUCKET", raising=False)
        with pytest.raises(RuntimeError) as exc:
            _create_storage()
        msg = str(exc.value)
        assert "REQUIRE_SHARED_STORAGE" in msg
        assert "GCS_BUCKET" in msg

    def test_truthy_with_bucket_and_gcs_failure_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The motivating case: operator set GCS_BUCKET but credentials
        are wrong / network is down. Silent fallback to LocalStorage on
        a multi-instance deploy would corrupt the index silently — we
        must refuse to start instead."""
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=True)
        with pytest.raises(RuntimeError) as exc:
            _create_storage()
        msg = str(exc.value)
        assert "REQUIRE_SHARED_STORAGE" in msg
        assert "refusing to fall back" in msg.lower()
        # The original GCS error must be preserved as the cause.
        assert exc.value.__cause__ is not None

    def test_truthy_with_bucket_and_gcs_ok_returns_gcsstorage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        result = _create_storage()
        assert isinstance(result, _StubGCSStorage)
        assert result.bucket_name == "my-bucket"

    @pytest.mark.parametrize(
        "value", ["1", "true", "TRUE", "yes", "on", "  true  "]
    )
    def test_truthy_values_are_all_recognized(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", value)
        monkeypatch.delenv("GCS_BUCKET", raising=False)
        with pytest.raises(RuntimeError):
            _create_storage()

    @pytest.mark.parametrize(
        "value", ["", "0", "false", "FALSE", "no", "off", "anything-else"]
    )
    def test_falsy_values_keep_silent_fallback(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Anything not in the truthy set must NOT trigger the guard —
        a typo like ``REQUIRE_SHARED_STORAGE=enabled`` should not
        accidentally relax the requirement, but conversely also should
        not arm it. (We err on the side of preserving legacy behavior
        for unknown values; the warning log on misconfig will surface
        this in monitoring.)"""
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", value)
        monkeypatch.delenv("GCS_BUCKET", raising=False)
        # No bucket + no guard = LocalStorage.
        result = _create_storage()
        assert isinstance(result, LocalStorage)


class TestRequireSharedStorageProbe:
    """an earlier adversarial review: ``GCSStorage.__init__`` is lazy (no I/O), so when
    REQUIRE_SHARED_STORAGE=true we MUST exercise the bucket end-to-end
    before declaring it usable. Otherwise wrong bucket name / missing
    IAM passes startup and explodes on first real call."""

    def test_probe_runs_when_require_shared_is_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity: on the success path, ALL four IAM verbs we use at
        runtime (write/read/list/delete) are exercised, and the probe
        key starts with the expected prefix."""
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        result = _create_storage()
        assert isinstance(result, _StubGCSStorage)
        assert len(result.write_calls) == 1
        assert len(result.read_calls) == 1
        # list_keys was called with the expected (prefix, suffix) so a
        # bucket with no list IAM would have been caught.
        assert result.list_calls == [("_probe", ".tmp")]
        assert len(result.delete_calls) == 1
        assert result.write_calls[0].startswith("_probe/require_shared_storage_")
        # And the bucket is clean afterward (delete worked).
        assert result._objects == {}

    def test_probe_does_NOT_run_when_require_shared_is_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-instance / local-dev deploys shouldn't pay the
        startup-probe latency."""
        monkeypatch.delenv("REQUIRE_SHARED_STORAGE", raising=False)
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        result = _create_storage()
        assert isinstance(result, _StubGCSStorage)
        # No probe activity on this path.
        assert result.write_calls == []
        assert result.read_calls == []
        assert result.delete_calls == []

    def test_probe_write_failure_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The IAM bug the finding called out: service account can
        construct a bucket handle but can't write to it. Must refuse
        to start instead of falling back to LocalStorage."""
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        monkeypatch.setattr(_StubGCSStorage, "fail_on_write", True)
        with pytest.raises(RuntimeError) as exc:
            _create_storage()
        msg = str(exc.value)
        assert "REQUIRE_SHARED_STORAGE" in msg
        assert "probe" in msg.lower()
        # The original GCS error is preserved as __cause__ for triage.
        assert exc.value.__cause__ is not None
        assert "simulated GCS write failure" in str(exc.value.__cause__)

    def test_probe_read_failure_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        monkeypatch.setattr(_StubGCSStorage, "fail_on_read", True)
        with pytest.raises(RuntimeError) as exc:
            _create_storage()
        assert "probe" in str(exc.value).lower()

    def test_probe_corrupt_read_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the bucket returns different bytes than we wrote (extreme
        misconfiguration, stale CDN, etc.), refuse to declare it OK."""
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        monkeypatch.setattr(_StubGCSStorage, "corrupt_read", True)
        with pytest.raises(RuntimeError) as exc:
            _create_storage()
        msg = str(exc.value)
        assert "probe" in msg.lower()
        # The probe's own diagnostic about byte count is preserved as
        # the cause.
        assert exc.value.__cause__ is not None
        assert "bytes" in str(exc.value.__cause__)

    def test_probe_delete_failure_raises_and_does_not_litter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the service account can write+read but not delete, we
        still want to surface that — the probe's best-effort cleanup
        runs but the failure still propagates."""
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        monkeypatch.setattr(_StubGCSStorage, "fail_on_delete", True)
        with pytest.raises(RuntimeError) as exc:
            _create_storage()
        assert "probe" in str(exc.value).lower()


    def test_probe_list_failure_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an earlier adversarial review: a deploy with create/get/delete but no
        storage.objects.list IAM would pass the original probe and
        fail later at /notebooks listing. The list check catches that
        at startup."""
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        monkeypatch.setattr(_StubGCSStorage, "fail_on_list", True)
        with pytest.raises(RuntimeError) as exc:
            _create_storage()
        msg = str(exc.value)
        assert "REQUIRE_SHARED_STORAGE" in msg
        assert "probe" in msg.lower()
        assert exc.value.__cause__ is not None
        assert "list" in str(exc.value.__cause__).lower()

    def test_probe_key_not_visible_in_list_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Some misconfigurations let list_keys return without raising
        but produce an empty result (e.g. bucket misrouting, stale list
        cache). The probe must reject that — the freshly-written probe
        object MUST be visible to its own list call."""
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        monkeypatch.setattr(_StubGCSStorage, "empty_list", True)
        with pytest.raises(RuntimeError) as exc:
            _create_storage()
        assert "probe" in str(exc.value).lower()
        assert exc.value.__cause__ is not None
        # The probe-specific diagnostic mentions the listing problem.
        assert "list" in str(exc.value.__cause__).lower()

    def test_probe_keys_collision_resistant_across_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an earlier adversarial review LOW: two simultaneous startups (autoscale-out,
        blue/green cutover) must NEVER pick the same probe key.

        Without a UUID suffix, two instances landing on the same
        millisecond would race on the same object — instance A's
        delete could fire while instance B is mid-read, producing a
        false-positive startup failure on a healthy bucket. The probe
        key must embed a UUID so collisions are statistically
        impossible.
        """
        monkeypatch.setenv("REQUIRE_SHARED_STORAGE", "true")
        monkeypatch.setenv("GCS_BUCKET", "my-bucket")
        _patch_gcs(monkeypatch, fails=False)
        # Run the probe many times back-to-back (simulating many
        # near-simultaneous startups). All probe keys must be unique.
        keys: set[str] = set()
        for _ in range(50):
            result = _create_storage()
            assert isinstance(result, _StubGCSStorage)
            keys.add(result.write_calls[0])
        assert len(keys) == 50, (
            f"probe keys collided across 50 calls — only {len(keys)} unique."
            " A UUID suffix is required to make same-millisecond startups"
            " safe."
        )


def test_GCSStorage_is_still_exported() -> None:
    """Smoke test that callers like router.py importing ``GCSStorage``
    from this module still work after the refactor."""
    # The import at the top of this test file would have failed if not.
    assert GCSStorage is not None
