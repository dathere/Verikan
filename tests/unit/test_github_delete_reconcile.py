"""Tests for the non-atomic delete fix + reconciliation (issue #46 step 3).

Covers:

* ``_delete_file`` uses the correct httpx call signature
  (``client.request("DELETE", ..., json=body)`` instead of the broken
  ``client.delete(..., json=body)``).
* ``_delete_file`` retries once on 409 (SHA race), mirroring the PUT
  retry from step 2.
* ``publish_verified`` returns ``draft_cleanup_pending`` truthfully and
  logs a warning when the draft delete failed, but still returns success
  for the verified create.
* ``reconcile_drafts_verified`` finds filenames present in both folders
  and tries to delete the draft copies, reporting success/failure per
  path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from data_concierge.gateway import github_publisher
from data_concierge.gateway.github_publisher import (
    _delete_file,
    publish_verified,
    reconcile_drafts_verified,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self, status_code: int, text: str = "", json_body: Any | None = None
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json = json_body

    def json(self) -> Any:
        return self._json


class _FakeAsyncClient:
    """Captures GET/PUT/DELETE/request calls and returns scripted responses."""

    def __init__(
        self,
        sha_sequence: list[str | None] | None = None,
        put_sequence: list[_FakeResponse] | None = None,
        delete_sequence: list[_FakeResponse] | None = None,
        list_sequence: list[_FakeResponse] | None = None,
        get_response_sequence: list[_FakeResponse] | None = None,
    ) -> None:
        self._sha_sequence = list(sha_sequence or [])
        self._put_sequence = list(put_sequence or [])
        self._delete_sequence = list(delete_sequence or [])
        self._list_sequence = list(list_sequence or [])
        # When set, completely overrides the sha_sequence path for GET
        # responses — used to script raw status codes (e.g. 500 on refetch)
        # without going through the sha-encoding helper.
        self._get_response_sequence = list(get_response_sequence or [])
        self.get_calls: list[str] = []
        self.put_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    async def get(self, url: str, **kw: Any) -> _FakeResponse:
        self.get_calls.append(url)
        # Listing endpoints (folder URLs) come from _list_sequence; per-file
        # SHA checks come from _sha_sequence. Differentiate by trailing /.
        # Real GitHub: GET on a folder returns a list; GET on a file returns
        # an object with "sha". We approximate by emptying _list_sequence
        # first for any URL that ends in a known folder name.
        if self._list_sequence and ("/contents/drafts" in url and not url.rstrip("/").endswith(".ipynb") or
                                    "/contents/verified" in url and not url.rstrip("/").endswith(".ipynb")):
            return self._list_sequence.pop(0)
        # Initial SHA-encoded responses come from sha_sequence; once that is
        # exhausted, fall through to get_response_sequence for raw responses
        # (used by tests that exercise the refetch-after-409 path).
        if self._sha_sequence:
            sha = self._sha_sequence.pop(0)
            if sha is None:
                return _FakeResponse(404)
            return _FakeResponse(200, json_body={"sha": sha})
        if self._get_response_sequence:
            return self._get_response_sequence.pop(0)
        return _FakeResponse(404)

    async def put(self, url: str, **kw: Any) -> _FakeResponse:
        self.put_calls.append({"url": url, "body": kw.get("json", {})})
        return self._put_sequence.pop(0)

    async def request(self, method: str, url: str, **kw: Any) -> _FakeResponse:
        if method.upper() != "DELETE":
            raise AssertionError(f"unexpected request method {method!r}")
        # Copy the body so a caller mutating it after the call (e.g. on a
        # retry that overwrites body["sha"]) doesn't retroactively change
        # what we recorded — real httpx serializes the body at send time.
        self.delete_calls.append({"url": url, "body": dict(kw.get("json", {}))})
        return self._delete_sequence.pop(0)

    # respx-style helpers for async context-manager use
    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _delete_file — signature fix + 409 retry
# ---------------------------------------------------------------------------


class TestDeleteFile:
    def test_uses_request_delete_not_client_delete(self) -> None:
        """The fix is to route DELETE through client.request() because
        httpx's AsyncClient.delete() does NOT accept a json= kwarg. Before
        this fix, every DELETE raised TypeError. The fake client errors out
        on client.delete() being called, so this test fails if anyone
        regresses to the old broken signature."""
        client = _FakeAsyncClient(
            sha_sequence=["abc123"],
            delete_sequence=[_FakeResponse(200)],
        )
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/foo.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is True
        assert len(client.delete_calls) == 1
        assert client.delete_calls[0]["body"]["sha"] == "abc123"

    def test_missing_sha_returns_false_no_delete(self) -> None:
        """Pre-DELETE SHA fetch 404 — return False, no DELETE call."""
        client = _FakeAsyncClient(sha_sequence=[None])
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/nope.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is False
        assert client.delete_calls == []

    def test_retry_once_on_409_then_succeeds(self) -> None:
        """First DELETE 409 (SHA race), refetch SHA, second DELETE 200."""
        client = _FakeAsyncClient(
            sha_sequence=["sha-stale", "sha-fresh"],
            delete_sequence=[
                _FakeResponse(409, text="conflict"),
                _FakeResponse(200),
            ],
        )
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/foo.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is True
        assert len(client.delete_calls) == 2
        assert client.delete_calls[0]["body"]["sha"] == "sha-stale"
        assert client.delete_calls[1]["body"]["sha"] == "sha-fresh"

    def test_retry_409_then_404_returns_true(self) -> None:
        """First 409, then SHA refetch returns None (winner already deleted
        the file) — treat as 'gone' and return True."""
        client = _FakeAsyncClient(
            sha_sequence=["sha-stale", None],
            delete_sequence=[_FakeResponse(409)],
        )
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/foo.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is True
        assert len(client.delete_calls) == 1

    def test_retry_409_then_409_returns_false(self) -> None:
        """Two consecutive 409s — give up, return False (no infinite loop)."""
        client = _FakeAsyncClient(
            sha_sequence=["sha1", "sha2"],
            delete_sequence=[
                _FakeResponse(409),
                _FakeResponse(409),
            ],
        )
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/foo.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is False
        assert len(client.delete_calls) == 2

    def test_5xx_returns_false_no_retry(self) -> None:
        """500 isn't a SHA race; don't retry, just report failure."""
        client = _FakeAsyncClient(
            sha_sequence=["abc"],
            delete_sequence=[_FakeResponse(500, text="server error")],
        )
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/foo.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is False
        assert len(client.delete_calls) == 1


    def test_409_then_refetch_500_returns_false_not_true(self) -> None:
        """MEDIUM-severity guard (an earlier adversarial review): after a DELETE 409 the
        retry must distinguish a real 404 (file is gone, cleanup succeeded)
        from a transient 500/403/auth failure on the refetch (cleanup
        unknown, do NOT report success). Previously, _get_file_sha
        returning None for ANY non-200 caused 500-on-refetch to be reported
        as a successful cleanup."""
        client = _FakeAsyncClient(
            sha_sequence=["sha-initial"],  # for the first _get_file_sha
            delete_sequence=[_FakeResponse(409, text="conflict")],
            get_response_sequence=[
                # Refetch sees a transient 500 — we cannot conclude the
                # winner deleted the file. Must report cleanup failure.
                _FakeResponse(500, text="server error"),
            ],
        )
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/foo.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is False, (
            "500 on refetch must NOT be misreported as a successful cleanup"
        )
        # Only the initial DELETE happened — we don't issue a second DELETE
        # when we can't trust the SHA refetch.
        assert len(client.delete_calls) == 1

    def test_409_then_refetch_403_returns_false_not_true(self) -> None:
        """Same as above but with 403 (rate-limited or auth)."""
        client = _FakeAsyncClient(
            sha_sequence=["sha-initial"],
            delete_sequence=[_FakeResponse(409)],
            get_response_sequence=[_FakeResponse(403, text="rate limited")],
        )
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/foo.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is False

    def test_409_then_refetch_404_returns_true(self) -> None:
        """And the inverse: a *real* 404 on the refetch means the winner of
        the race already deleted the file — that IS a successful cleanup."""
        client = _FakeAsyncClient(
            sha_sequence=["sha-initial"],
            delete_sequence=[_FakeResponse(409)],
            get_response_sequence=[_FakeResponse(404)],
        )
        result = _run(
            _delete_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "drafts/foo.ipynb",
                "rm",
                "tok",
                "main",
            )
        )
        assert result is True
        # The retry path correctly skipped issuing a second DELETE since
        # there's nothing left to delete.
        assert len(client.delete_calls) == 1


# ---------------------------------------------------------------------------
# publish_verified — draft_cleanup_pending
# ---------------------------------------------------------------------------


class TestPublishVerifiedDraftCleanupPending:
    @pytest.fixture(autouse=True)
    def _settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings",
            lambda: {
                "enabled": True,
                "token": "tok",
                "repo": "owner/repo",
                "branch": "main",
                "drafts_folder": "drafts",
                "verified_folder": "verified",
            },
        )

    def test_returns_pending_false_when_delete_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _ok_create(*a: Any, **kw: Any) -> dict[str, Any]:
            return {"content": {"sha": "v1"}}

        async def _ok_delete(*a: Any, **kw: Any) -> bool:
            return True

        monkeypatch.setattr(github_publisher, "_create_or_update_file", _ok_create)
        monkeypatch.setattr(github_publisher, "_delete_file", _ok_delete)

        result = _run(
            publish_verified(
                submission_id="s1",
                notebook_id="n1",
                query="q",
                notebook_json={"cells": []},
            )
        )
        assert result is not None
        assert result["draft_cleanup_pending"] is False
        assert result["path"].startswith("verified/")
        assert result["sha"] == "v1"

    def test_returns_pending_true_when_delete_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verified create succeeded; draft delete returned False (e.g. 5xx
        or persistent 409). publish_verified must NOT raise — the verified
        side won. But the return dict flags the leak so admins can clean it."""

        async def _ok_create(*a: Any, **kw: Any) -> dict[str, Any]:
            return {"content": {"sha": "v1"}}

        async def _fail_delete(*a: Any, **kw: Any) -> bool:
            return False

        monkeypatch.setattr(github_publisher, "_create_or_update_file", _ok_create)
        monkeypatch.setattr(github_publisher, "_delete_file", _fail_delete)

        result = _run(
            publish_verified(
                submission_id="s1",
                notebook_id="n1",
                query="q for leaked draft",
                notebook_json={"cells": []},
            )
        )
        assert result is not None
        assert result["draft_cleanup_pending"] is True

        # A loud warning must be emitted so audit/log scrapers can flag the
        # leak. Capture once (capsys drains its buffers on read) and assert
        # the specific "may be orphaned" wording so a renamed log message
        # would have to opt back in.
        captured = capsys.readouterr()
        out_err = captured.out + captured.err
        assert "may be orphaned" in out_err.lower(), (
            "publish_verified must warn that the draft may be orphaned when "
            "the cleanup failed; got:\n" + out_err
        )


    def test_delete_file_raising_does_not_demote_publish(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """HIGH-severity guard (an earlier adversarial review): if _delete_file raises (e.g.
        network timeout, connection reset), publish_verified must NOT
        propagate that as a publish failure. The verified create already
        succeeded — the SSOT contract is to commit local state with
        draft_cleanup_pending=True, NOT return 502 to the admin after the
        GitHub side had already won."""

        async def _ok_create(*a: Any, **kw: Any) -> dict[str, Any]:
            return {"content": {"sha": "v1"}}

        async def _raises_delete(*a: Any, **kw: Any) -> bool:
            raise httpx.ConnectTimeout("draft delete timed out")

        monkeypatch.setattr(github_publisher, "_create_or_update_file", _ok_create)
        monkeypatch.setattr(github_publisher, "_delete_file", _raises_delete)

        result = _run(
            publish_verified(
                submission_id="s1",
                notebook_id="n1",
                query="q for raising delete",
                notebook_json={"cells": []},
            )
        )
        # The publish must succeed (write-through contract: verified is the
        # SSOT gate; cleanup is best-effort).
        assert result is not None
        assert result["draft_cleanup_pending"] is True
        assert result["path"].startswith("verified/")

        # The swallowed exception detail must show up in the log, not just
        # the generic "cleanup" handler text. Capture once (capsys drains
        # on read) and assert on the exception message itself so a change
        # that silently dropped the error= field would fail.
        captured = capsys.readouterr()
        out_err = captured.out + captured.err
        assert "draft delete timed out" in out_err, (
            "publish_verified must log the swallowed cleanup exception's "
            "message, not just a generic 'cleanup' note; got:\n" + out_err
        )


# ---------------------------------------------------------------------------
# reconcile_drafts_verified
# ---------------------------------------------------------------------------


class TestReconcileDraftsVerified:
    """Reconcile calls ``_list_folder_files(folder, strict=True)`` for both
    drafts and verified; the per-folder lookup is dispatched in the helper
    below so each test can declare the contents of each folder cleanly."""

    @staticmethod
    def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings",
            lambda: {
                "enabled": True,
                "token": "tok",
                "repo": "owner/repo",
                "branch": "main",
                "drafts_folder": "drafts",
                "verified_folder": "verified",
            },
        )

    @staticmethod
    def _patch_list_folders(
        monkeypatch: pytest.MonkeyPatch,
        *,
        drafts: list[dict[str, Any]] | type[Exception] | None = None,
        verified: list[dict[str, Any]] | type[Exception] | None = None,
        raises: type[Exception] | None = None,
        raises_message: str = "boom",
    ) -> None:
        """Dispatch _list_folder_files by folder argument.

        Pass a list for the per-folder contents, or set ``raises`` to make
        the helper raise that exception type for any folder (used to test
        the strict-listing surface).
        """

        async def _stub(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            if raises is not None:
                raise raises(raises_message)
            if folder == "drafts":
                return drafts or []
            if folder == "verified":
                return verified or []
            raise AssertionError(f"unexpected folder {folder!r}")

        monkeypatch.setattr(github_publisher, "_list_folder_files", _stub)

    def test_returns_disabled_report_when_github_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings",
            lambda: {"enabled": False, "token": ""},
        )
        result = _run(reconcile_drafts_verified())
        assert result == {
            "checked": 0,
            "duplicates_found": 0,
            "cleaned": [],
            "failed": [],
            "disabled": True,
        }

    def test_no_duplicates_is_a_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drafts and verified share no filenames — nothing to clean."""
        self._patch_settings(monkeypatch)
        self._patch_list_folders(
            monkeypatch,
            drafts=[
                {"name": "only-draft.ipynb", "path": "drafts/only-draft.ipynb",
                 "sha": "d1"},
            ],
            verified=[
                {"name": "only-verified.ipynb",
                 "path": "verified/only-verified.ipynb", "sha": "v1"},
            ],
        )

        result = _run(reconcile_drafts_verified())
        assert result["checked"] == 1
        assert result["duplicates_found"] == 0
        assert result["cleaned"] == []
        assert result["failed"] == []
        assert result["disabled"] is False

    def test_finds_and_cleans_duplicates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two drafts; one is also in verified. Reconcile cleans that one."""
        self._patch_settings(monkeypatch)
        self._patch_list_folders(
            monkeypatch,
            drafts=[
                {"name": "leaked.ipynb", "path": "drafts/leaked.ipynb", "sha": "d1"},
                {"name": "still-pending.ipynb",
                 "path": "drafts/still-pending.ipynb", "sha": "d2"},
            ],
            verified=[
                {"name": "leaked.ipynb", "path": "verified/leaked.ipynb", "sha": "v1"},
                {"name": "older.ipynb", "path": "verified/older.ipynb", "sha": "v2"},
            ],
        )

        delete_calls: list[str] = []

        async def _ok_delete(client: Any, repo: str, path: str, *a: Any,
                             **kw: Any) -> bool:
            delete_calls.append(path)
            return True

        monkeypatch.setattr(github_publisher, "_delete_file", _ok_delete)

        result = _run(reconcile_drafts_verified())
        assert result["checked"] == 2
        assert result["duplicates_found"] == 1
        assert result["cleaned"] == ["drafts/leaked.ipynb"]
        assert result["failed"] == []
        assert delete_calls == ["drafts/leaked.ipynb"]

    def test_records_failures_per_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mix of success and failure across multiple duplicates."""
        self._patch_settings(monkeypatch)
        self._patch_list_folders(
            monkeypatch,
            drafts=[
                {"name": "a.ipynb", "path": "drafts/a.ipynb", "sha": "1"},
                {"name": "b.ipynb", "path": "drafts/b.ipynb", "sha": "2"},
                {"name": "c.ipynb", "path": "drafts/c.ipynb", "sha": "3"},
            ],
            verified=[
                {"name": "a.ipynb", "path": "verified/a.ipynb", "sha": "1v"},
                {"name": "b.ipynb", "path": "verified/b.ipynb", "sha": "2v"},
                {"name": "c.ipynb", "path": "verified/c.ipynb", "sha": "3v"},
            ],
        )

        async def _patchy_delete(
            client: Any, repo: str, path: str, *a: Any, **kw: Any
        ) -> bool:
            # b fails, a and c succeed
            return "b.ipynb" not in path

        # The disambiguation check (_is_file_gone) runs only when delete
        # returns False; for "b" we want it to confirm the file is still
        # there so reconcile reports it as a real failure.
        async def _still_there(*a: Any, **kw: Any) -> bool:
            return False

        monkeypatch.setattr(github_publisher, "_delete_file", _patchy_delete)
        monkeypatch.setattr(github_publisher, "_is_file_gone", _still_there)

        result = _run(reconcile_drafts_verified())
        assert result["duplicates_found"] == 3
        assert set(result["cleaned"]) == {"drafts/a.ipynb", "drafts/c.ipynb"}
        assert len(result["failed"]) == 1
        assert result["failed"][0]["path"] == "drafts/b.ipynb"
        assert "delete returned False" in result["failed"][0]["reason"]

    def test_exception_during_delete_caught_and_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_delete_file raising shouldn't blow up the whole reconcile."""
        self._patch_settings(monkeypatch)
        self._patch_list_folders(
            monkeypatch,
            drafts=[{"name": "x.ipynb", "path": "drafts/x.ipynb", "sha": "1"}],
            verified=[{"name": "x.ipynb", "path": "verified/x.ipynb", "sha": "2"}],
        )

        async def _boom(*a: Any, **kw: Any) -> bool:
            raise httpx.ConnectError("network down")

        monkeypatch.setattr(github_publisher, "_delete_file", _boom)

        result = _run(reconcile_drafts_verified())
        assert result["duplicates_found"] == 1
        assert result["cleaned"] == []
        assert len(result["failed"]) == 1
        assert "network down" in result["failed"][0]["reason"]

    # -------------------- New for Copilot review on PR #78 --------------------

    def test_delete_false_but_file_gone_is_counted_as_cleaned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Idempotency guard (Copilot PR #78): a second concurrent reconcile
        (or any race where the file gets deleted between our list and our
        delete) should report the path as CLEANED, not FAILED. The endpoint
        documents itself as safe to call repeatedly; an admin who runs it
        twice in a row should not see spurious failures from the second run.
        """
        self._patch_settings(monkeypatch)
        self._patch_list_folders(
            monkeypatch,
            drafts=[{"name": "raced.ipynb", "path": "drafts/raced.ipynb",
                     "sha": "d"}],
            verified=[{"name": "raced.ipynb", "path": "verified/raced.ipynb",
                       "sha": "v"}],
        )

        async def _delete_returns_false(*a: Any, **kw: Any) -> bool:
            return False

        async def _file_is_gone(*a: Any, **kw: Any) -> bool:
            # Concurrent reconcile (or some other writer) deleted it first.
            return True

        monkeypatch.setattr(github_publisher, "_delete_file", _delete_returns_false)
        monkeypatch.setattr(github_publisher, "_is_file_gone", _file_is_gone)

        result = _run(reconcile_drafts_verified())
        assert result["duplicates_found"] == 1
        assert result["cleaned"] == ["drafts/raced.ipynb"], (
            "An already-gone draft must be reported as cleaned, not failed — "
            "otherwise the endpoint's 'safe to call repeatedly' contract breaks"
        )
        assert result["failed"] == []

    def test_strict_listing_failure_propagates_as_github_publish_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strictness guard (Copilot PR #78): when the listing call can't
        reach GitHub at all (auth, network, 5xx), the function must raise
        rather than silently report a clean 'checked: 0' no-op. The
        endpoint maps this to 502 so an admin sees the real failure."""
        self._patch_settings(monkeypatch)
        self._patch_list_folders(
            monkeypatch,
            raises=github_publisher.GitHubPublishError,
            raises_message="GitHub returned 401 listing drafts",
        )

        with pytest.raises(github_publisher.GitHubPublishError) as exc:
            _run(reconcile_drafts_verified())
        assert "401" in str(exc.value)


# ---------------------------------------------------------------------------
# _list_folder_files — strict vs lenient
# ---------------------------------------------------------------------------


class TestListFolderFilesStrict:
    """Direct coverage of the new strict-listing mode."""

    @staticmethod
    def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings",
            lambda: {
                "enabled": True,
                "token": "tok",
                "repo": "owner/repo",
                "branch": "main",
                "drafts_folder": "drafts",
                "verified_folder": "verified",
            },
        )

    @staticmethod
    def _patch_httpx_get_response(
        monkeypatch: pytest.MonkeyPatch, response_factory: Any
    ) -> None:
        """Replace httpx.AsyncClient so _list_folder_files' GET returns the
        scripted response without doing any real network work."""

        class _Client:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> _Client:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def get(self, url: str, **kw: Any) -> Any:
                return response_factory()

        monkeypatch.setattr(github_publisher.httpx, "AsyncClient", _Client)

    def test_disabled_raises_in_strict_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings",
            lambda: {"enabled": False, "token": ""},
        )
        with pytest.raises(github_publisher.GitHubPublishError):
            _run(github_publisher._list_folder_files("drafts", strict=True))

    def test_disabled_returns_empty_in_lenient_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings",
            lambda: {"enabled": False, "token": ""},
        )
        result = _run(github_publisher._list_folder_files("drafts", strict=False))
        assert result == []

    # --- New for an earlier adversarial review -------------------------------------------

    def test_invalid_json_raises_in_strict_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """200 with malformed JSON used to bubble up as an unhandled
        exception (becomes a 500 at the endpoint). Strict mode must
        catch this and raise GitHubPublishError so the endpoint can
        map it to 502."""
        self._patch_settings(monkeypatch)

        class _BadJson:
            status_code = 200
            text = "<html>not json</html>"

            def json(self) -> Any:
                raise ValueError("invalid json")

        self._patch_httpx_get_response(monkeypatch, _BadJson)

        with pytest.raises(github_publisher.GitHubPublishError) as exc:
            _run(github_publisher._list_folder_files("drafts", strict=True))
        assert "parse" in str(exc.value).lower()

    def test_invalid_json_returns_empty_in_lenient_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lenient callers (sync_all_from_github, etc.) keep their
        'empty on any failure' contract — malformed JSON is still
        treated as 'nothing to list'."""
        self._patch_settings(monkeypatch)

        class _BadJson:
            status_code = 200
            text = "<html>not json</html>"

            def json(self) -> Any:
                raise ValueError("invalid json")

        self._patch_httpx_get_response(monkeypatch, _BadJson)

        result = _run(github_publisher._list_folder_files("drafts", strict=False))
        assert result == []

    def test_non_list_response_raises_in_strict_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the folder path actually points at a file, GitHub returns
        the file's metadata object instead of a directory listing. In
        strict mode this is a misconfiguration — surface it instead of
        silently reporting an empty folder."""
        self._patch_settings(monkeypatch)

        class _FileMetadata:
            status_code = 200
            text = ""

            def json(self) -> Any:
                # Not a list — looks like single-file metadata.
                return {"type": "file", "name": "oops.ipynb", "sha": "abc"}

        self._patch_httpx_get_response(monkeypatch, _FileMetadata)

        with pytest.raises(github_publisher.GitHubPublishError) as exc:
            _run(github_publisher._list_folder_files("drafts", strict=True))
        assert (
            "non-list" in str(exc.value).lower()
            or "misconfigured" in str(exc.value).lower()
        )

    def test_non_list_response_returns_empty_in_lenient_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_settings(monkeypatch)

        class _FileMetadata:
            status_code = 200
            text = ""

            def json(self) -> Any:
                return {"type": "file", "name": "oops.ipynb"}

        self._patch_httpx_get_response(monkeypatch, _FileMetadata)

        result = _run(github_publisher._list_folder_files("drafts", strict=False))
        assert result == []
