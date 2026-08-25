"""GitHub Notebook Publisher.

Pushes notebooks to a configurable GitHub repository:
- On submission: pushed to the drafts folder
- On approval: moved to the verified folder (delete draft + create verified)
- On rejection: optionally removed from drafts

Uses the GitHub Contents API via httpx.
"""

import base64
import json
from typing import Any

import httpx

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

# Storage key for GitHub settings (persisted across restarts via storage backend)
_GITHUB_SETTINGS_KEY = "github_settings.json"


class GitHubPublishError(RuntimeError):
    """Raised when a GitHub publish operation fails (HTTP, network, or auth).

    Distinct from the "publishing not active" case, where ``publish_verified()``
    returns ``None``. Callers that need GitHub to be the source of truth
    should treat this exception as a hard failure and refuse to commit local
    state that assumes the GitHub side succeeded.
    """


def _build_default_settings() -> dict[str, Any]:
    """Build the default settings dict, seeded from env-var-driven config.

    Returns a fresh dict on every call (callers mutate it). Field model:

    * ``paused`` — explicit operator kill-switch; halts all GitHub I/O
      without erasing the token. See ``is_publishing_active``.
    * ``token`` / ``webhook_secret`` — secrets; ``SecretStr`` in config
      is unwrapped here so the publisher can pass plain strings to
      ``_headers`` and signature verification. An empty webhook secret
      disables the webhook endpoint (returns 503; see
      ``gateway/github_webhook.py``).
    * ``repo`` / ``branch`` / ``drafts_folder`` / ``verified_folder`` /
      ``verified_answers_folder`` — read from ``core.config.Settings``,
      so an operator can seed initial values via ``.env`` (``GITHUB_REPO``,
      ``GITHUB_BRANCH``, etc.) on a fresh deploy. The admin panel writes
      to ``github_settings.json``, which overlays these defaults in
      ``load_github_settings``.

    Resolution order at runtime: ``github_settings.json`` (admin-panel
    edits) > env vars (this function) > hardcoded fallbacks in
    ``Settings`` field definitions.
    """
    from data_concierge.core.config import get_settings

    cfg = get_settings()
    return {
        "paused": False,
        "token": cfg.github_token.get_secret_value(),
        "repo": cfg.github_repo,
        "branch": cfg.github_branch,
        "drafts_folder": cfg.github_drafts_folder,
        "verified_folder": cfg.github_verified_folder,
        "verified_answers_folder": cfg.github_verified_answers_folder,
        "webhook_secret": cfg.github_webhook_secret.get_secret_value(),
    }


def is_publishing_active(settings: dict[str, Any]) -> bool:
    """Whether GitHub I/O should proceed for the given settings dict.

    Returns ``True`` when the integration is fully configured (both
    ``repo`` and ``token`` are non-empty) AND the admin has not paused
    publishing. This replaces the legacy ``enabled`` flag: any
    deployment with valid creds is active by default, and operators
    pause explicitly via the admin UI.
    """
    return (
        bool(settings.get("token"))
        and bool(settings.get("repo"))
        and not settings.get("paused", False)
    )


def build_blob_url(
    path: str | None,
    settings: dict[str, Any] | None = None,
) -> str | None:
    """Build a clickable GitHub blob URL for a published artifact (#111).

    Returns ``https://github.com/<repo>/blob/<branch>/<path>`` so the admin
    UI can link straight to the notebook/answer file that backs a verified
    entry. Returns ``None`` when ``path`` is empty or no repo is configured
    (e.g. GitHub publishing is off), so callers simply omit the link.

    Note this does NOT require a token — the URL points at the public repo
    page, which works for any reader; only *writing* needs credentials.
    ``settings`` may be passed in to avoid re-reading the store in a loop.
    """
    if not path:
        return None
    if settings is None:
        settings = load_github_settings()
    repo = settings.get("repo")
    if not repo:
        return None
    branch = settings.get("branch") or "main"
    return f"https://github.com/{repo}/blob/{branch}/{path}"


def build_raw_url(
    path: str | None,
    settings: dict[str, Any] | None = None,
) -> str | None:
    """Build the raw (fetchable) GitHub URL for a published artifact.

    Returns ``https://raw.githubusercontent.com/<repo>/<branch>/<path>`` — the
    content URL a verifier fetches (vs. the HTML ``blob`` URL from
    :func:`build_blob_url`). Used as the evidence package's ``packageUrl`` so a
    reader can fetch the canonical-JSON package and recompute its hash.
    """
    if not path:
        return None
    if settings is None:
        settings = load_github_settings()
    repo = settings.get("repo")
    if not repo:
        return None
    branch = settings.get("branch") or "main"
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def load_github_settings_with_status() -> tuple[dict[str, Any], str]:
    """Variant of ``load_github_settings`` that also reports load status.

    Returns ``(settings, status)`` where status is one of:

    * ``"ok"`` — saved settings loaded normally, or no file existed
      (env-var seeded defaults apply).
    * ``"fail_closed"`` — ``storage.read_json`` errored; returned
      settings are env-derived with ``paused=True`` forced. This is
      a transient failure (corrupt file, storage outage), distinct
      from a deliberate admin pause.

    Callers that branch on this (notably ``gateway.github_webhook``)
    can return HTTP 503 on ``fail_closed`` so GitHub retries the
    delivery once storage recovers, while still returning HTTP 200
    for an intentional admin pause (an earlier adversarial review). Most callers
    should use the plain ``load_github_settings`` wrapper.
    """
    settings = _build_default_settings()
    migrated = False
    try:
        saved = storage.read_json(_GITHUB_SETTINGS_KEY)
        if saved:
            if "enabled" in saved:
                if "paused" not in saved:
                    saved["paused"] = not bool(saved.get("enabled", False))
                saved.pop("enabled", None)
                migrated = True
            settings.update(saved)
    except Exception as e:
        # Fail closed: if we can't read the saved settings, force
        # ``paused=True`` rather than returning env-derived defaults
        # with ``paused=False``. A corrupt/unreadable file MUST NOT
        # silently bypass a stored pause state we can't observe
        # (an earlier adversarial review). ``storage.read_json`` returns ``None`` for
        # missing files (no exception), so this branch only fires for
        # actual read errors — env-var seeding still works on a fresh
        # deploy where the file is genuinely absent.
        logger.warning(
            "Failed to load GitHub settings; failing closed (paused=True)",
            error=str(e),
        )
        settings["paused"] = True
        return settings, "fail_closed"
    if migrated:
        try:
            save_github_settings(settings)
            logger.info("Migrated legacy 'enabled' field to 'paused'")
        except Exception as e:
            logger.warning(
                "Failed to persist migrated GitHub settings; "
                "migration will retry on next load",
                error=str(e),
            )
    return settings, "ok"


def load_github_settings() -> dict[str, Any]:
    """Load GitHub settings from storage, falling back to defaults.

    Migrates the legacy ``enabled`` field to ``paused`` (inverted) on
    first read after upgrade: a saved ``{"enabled": false, ...}`` is
    rewritten as ``{"paused": true, ...}``, preserving the operator's
    "publishing is off" intent. ``enabled: true`` becomes
    ``paused: false``. The ``enabled`` key is dropped after migration,
    and the cleaned settings file is persisted so the migration runs
    at most once.

    If both keys somehow exist (mixed-version writes), ``paused`` wins
    and ``enabled`` is dropped — we don't want a stale legacy flag to
    silently re-pause a deliberately-resumed integration.

    Callers that need to distinguish "admin paused" from "settings
    unreadable" (e.g., to choose between HTTP 200 and 503 in the
    webhook handler) should use ``load_github_settings_with_status``.
    """
    settings, _ = load_github_settings_with_status()
    return settings


def save_github_settings(settings: dict[str, Any]) -> None:
    """Persist GitHub settings to storage."""
    storage.write_json(_GITHUB_SETTINGS_KEY, settings)
    logger.info("GitHub settings saved")


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _api_url(repo: str) -> str:
    return f"https://api.github.com/repos/{repo}/contents"


async def _get_file_sha(
    client: httpx.AsyncClient, repo: str, path: str, token: str, branch: str
) -> str | None:
    """Get the SHA of an existing file (needed for updates/deletes)."""
    url = f"{_api_url(repo)}/{path}"
    resp = await client.get(url, headers=_headers(token), params={"ref": branch})
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


async def _create_or_update_file(
    client: httpx.AsyncClient,
    repo: str,
    path: str,
    content_bytes: bytes,
    message: str,
    token: str,
    branch: str,
) -> dict[str, Any]:
    """Create or update a file in the GitHub repo.

    Two concurrent calls targeting the same path will race on the SHA: the
    first wins, the second comes back with HTTP 409 because the SHA we
    fetched before our PUT is no longer current. When that happens, refetch
    the SHA and retry once. If the retry also fails (still racing, or some
    other 4xx/5xx), surface the error normally via ``raise_for_status``.
    A single retry is enough in practice — the writer that lost is now
    serialized after the winner, so its second attempt sees the latest SHA.
    """
    url = f"{_api_url(repo)}/{path}"
    encoded = base64.b64encode(content_bytes).decode("ascii")

    for attempt in (0, 1):
        body: dict[str, Any] = {
            "message": message,
            "content": encoded,
            "branch": branch,
        }
        # Check if file already exists (need SHA for update)
        sha = await _get_file_sha(client, repo, path, token, branch)
        if sha:
            body["sha"] = sha

        resp = await client.put(url, headers=_headers(token), json=body)
        if resp.status_code == 409 and attempt == 0:
            logger.warning(
                "GitHub PUT 409 (SHA race); refetching SHA and retrying once",
                repo=repo,
                path=path,
                detail=resp.text[:200],
            )
            continue
        resp.raise_for_status()
        return resp.json()

    # Unreachable: the loop above either returns or raises.
    raise RuntimeError("_create_or_update_file retry loop fell through")


async def _delete_file(
    client: httpx.AsyncClient,
    repo: str,
    path: str,
    message: str,
    token: str,
    branch: str,
) -> bool:
    """Delete a file from the GitHub repo. Returns True if deleted.

    Three notes on the implementation:

    * httpx's ``AsyncClient.delete()`` does not accept a ``json=`` kwarg, so
      we go through ``client.request("DELETE", ...)`` to send the SHA-bearing
      body that the GitHub Contents API requires.
    * 409 on DELETE means another writer changed the SHA between our
      SHA lookup and our DELETE (mirrors the PUT SHA race fixed in
      ``_create_or_update_file``). Refetch the SHA and retry once.
    * Special care on the retry's SHA refetch: ``_get_file_sha()`` returns
      ``None`` for *any* non-200 response (404 OR transient 500 / 403 rate
      limit / auth failure). Treating all of those as "file is gone" would
      silently report cleanup success while the draft may still exist. On
      the retry we do an explicit status-aware GET and only treat a real
      404 as "already deleted".
    """
    url = f"{_api_url(repo)}/{path}"

    # Initial attempt.
    sha = await _get_file_sha(client, repo, path, token, branch)
    if not sha:
        # No SHA on the FIRST attempt means the file likely doesn't exist
        # (the typical "nothing to clean up" case). Existing callers
        # interpret this as "no draft to remove" and don't retry.
        return False

    body = {"message": message, "sha": sha, "branch": branch}
    resp = await client.request("DELETE", url, headers=_headers(token), json=body)
    if resp.status_code == 200:
        return True
    if resp.status_code != 409:
        logger.warning(
            "GitHub DELETE failed",
            repo=repo,
            path=path,
            status=resp.status_code,
            detail=resp.text[:200],
        )
        return False

    # 409 SHA race — refetch with a status-aware GET, only treat a real 404
    # as "winner already deleted it"; everything else means we can't trust
    # that cleanup succeeded.
    logger.warning(
        "GitHub DELETE 409 (SHA race); refetching SHA and retrying once",
        repo=repo,
        path=path,
        detail=resp.text[:200],
    )
    refetch = await client.get(url, headers=_headers(token), params={"ref": branch})
    if refetch.status_code == 404:
        # The winner of the SHA race already deleted it for us.
        return True
    if refetch.status_code != 200:
        logger.warning(
            "Refetch after DELETE 409 returned non-200/404; cannot confirm cleanup",
            repo=repo,
            path=path,
            status=refetch.status_code,
            detail=refetch.text[:200],
        )
        return False
    fresh_sha = refetch.json().get("sha")
    if not fresh_sha:
        # 200 but no sha field — defensive, shouldn't happen with GitHub.
        logger.warning(
            "Refetch returned 200 without a sha field; cannot confirm cleanup",
            repo=repo,
            path=path,
        )
        return False

    body["sha"] = fresh_sha
    resp = await client.request("DELETE", url, headers=_headers(token), json=body)
    if resp.status_code == 200:
        return True
    logger.warning(
        "GitHub DELETE retry failed",
        repo=repo,
        path=path,
        status=resp.status_code,
        detail=resp.text[:200],
    )
    return False


def _notebook_filename(submission_id: str, query: str) -> str:
    """Generate a readable filename from the query."""
    # Slugify: lowercase, replace spaces with hyphens, keep alphanumeric
    slug = query.lower().strip()[:60]
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = slug.strip().replace(" ", "-")
    slug = "-".join(part for part in slug.split("-") if part)  # collapse multiple hyphens
    if not slug:
        slug = "notebook"
    return f"{slug}_{submission_id[:8]}.ipynb"


def _build_commit_message(
    summary: str,
    reason: str | None = None,
    reviewer: str | None = None,
) -> str:
    """Compose a commit message that records the review reason and reviewer.

    The bot account is always the Git *committer* (the GitHub identity that
    owns the configured token). To preserve an audit trail (#112) we record
    *why* the action happened (``reason``) and *who* requested it
    (``reviewer`` — the reviewing admin's GitHub identity) in the commit
    message body:

        Verify notebook: <query>

        Reason: <reason>
        Reviewed-by: <reviewer>

    ``reason`` and ``reviewer`` are optional so internal callers (sync /
    reconcile flows that aren't tied to a human review action) can omit
    them and keep the plain one-line summary.
    """
    lines = [summary]
    trailer: list[str] = []
    if reason:
        trailer.append(f"Reason: {reason}")
    if reviewer:
        trailer.append(f"Reviewed-by: {reviewer}")
    if trailer:
        lines.append("")
        lines.extend(trailer)
    return "\n".join(lines)


# =============================================================================
# Public API
# =============================================================================


async def publish_draft(
    submission_id: str, query: str, notebook_json: dict[str, Any]
) -> dict[str, Any] | None:
    """Push a notebook to the drafts folder on GitHub.

    Returns the GitHub API response, or None if publishing is disabled/fails.
    """
    settings = load_github_settings()
    if not is_publishing_active(settings):
        return None

    repo = settings["repo"]
    branch = settings["branch"]
    token = settings["token"]
    folder = settings["drafts_folder"]
    filename = _notebook_filename(submission_id, query)
    path = f"{folder}/{filename}"

    content = json.dumps(notebook_json, indent=2).encode("utf-8")
    message = f"Add draft notebook: {query[:80]}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await _create_or_update_file(
                client, repo, path, content, message, token, branch
            )
        logger.info(
            "Draft notebook published to GitHub",
            repo=repo,
            path=path,
            submission_id=submission_id,
        )
        return {"path": path, "filename": filename, "sha": result.get("content", {}).get("sha")}
    except httpx.HTTPStatusError as e:
        logger.error(
            "GitHub API error publishing draft",
            status=e.response.status_code,
            detail=e.response.text[:200],
            submission_id=submission_id,
        )
        return None
    except Exception as e:
        logger.error("Failed to publish draft to GitHub", error=str(e))
        return None


async def publish_verified(
    submission_id: str,
    notebook_id: str,
    query: str,
    notebook_json: dict[str, Any],
    reason: str | None = None,
    reviewer: str | None = None,
    sibling_files: dict[str, bytes] | None = None,
) -> dict[str, Any] | None:
    """Move a notebook from drafts to verified folder on GitHub.

    1. Creates the file in the verified folder
    2. Deletes the file from the drafts folder

    Returns the GitHub API response on success, or ``None`` if GitHub
    publishing is disabled. Raises :class:`GitHubPublishError` on failure
    of the verified create (HTTP, network, or auth) so callers can refuse
    to commit local state that assumes the GitHub side succeeded.

    The verified create is the only operation that gates approval. If the
    draft delete fails (e.g. transient 409 retry exhausted, 5xx, network
    timeout, or the draft never existed because submission-time publishing
    was off), the notebook still ends up in the verified folder and we
    return success — but the returned dict carries
    ``draft_cleanup_pending=True`` so callers and audit logs can flag the
    leaked draft. The ``reconcile_drafts_verified()`` endpoint cleans
    these up out of band.

    Critically, the draft cleanup is wrapped in its own ``try/except``:
    if ``_delete_file`` itself raises (network timeout, connection reset,
    unexpected error), we must NOT convert that into a publish failure —
    the verified create already succeeded and the SSOT contract is to
    commit the local state. Raising here would return 502 to the admin
    after the GitHub side had already won, which is exactly the
    inconsistency this whole subsystem exists to avoid.
    """
    settings = load_github_settings()
    if not is_publishing_active(settings):
        return None

    repo = settings["repo"]
    branch = settings["branch"]
    token = settings["token"]
    drafts_folder = settings["drafts_folder"]
    verified_folder = settings["verified_folder"]
    filename = _notebook_filename(submission_id, query)
    draft_path = f"{drafts_folder}/{filename}"
    verified_path = f"{verified_folder}/{filename}"

    content = json.dumps(notebook_json, indent=2).encode("utf-8")
    commit_msg = _build_commit_message(
        f"Verify notebook: {query[:80]}", reason, reviewer
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create in verified folder — this is the SSOT gate. Any failure
            # here must propagate so the caller refuses to commit local state.
            result = await _create_or_update_file(
                client, repo, verified_path, content, commit_msg, token, branch
            )
            # Best-effort draft cleanup. Isolated from the verified create so
            # cleanup raises (timeouts, connection resets) can never demote
            # the publish to a failure.
            draft_deleted = False
            try:
                draft_deleted = await _delete_file(
                    client,
                    repo,
                    draft_path,
                    _build_commit_message(
                        f"Remove draft (verified): {query[:60]}", reason, reviewer
                    ),
                    token,
                    branch,
                )
            except Exception as cleanup_err:
                logger.warning(
                    "Draft cleanup raised after successful verified create; "
                    "treating as draft_cleanup_pending so reconcile can recover",
                    repo=repo,
                    draft_path=draft_path,
                    verified_path=verified_path,
                    submission_id=submission_id,
                    notebook_id=notebook_id,
                    error=str(cleanup_err),
                )

            # Best-effort sibling files (e.g. the canonical evidence package
            # JSON published next to the notebook so the embedded commitment
            # view's packageUrl resolves). Isolated like the draft cleanup: a
            # sibling failure must never demote the gating verified create.
            for sib_path, sib_content in (sibling_files or {}).items():
                try:
                    await _create_or_update_file(
                        client,
                        repo,
                        sib_path,
                        sib_content,
                        _build_commit_message(
                            f"Evidence package: {query[:60]}", reason, reviewer
                        ),
                        token,
                        branch,
                    )
                except Exception as sib_err:
                    logger.warning(
                        "Evidence sibling publish failed (non-blocking)",
                        repo=repo,
                        sibling_path=sib_path,
                        submission_id=submission_id,
                        error=str(sib_err),
                    )
        if not draft_deleted:
            logger.warning(
                "Verified create succeeded but draft delete did not; "
                "draft may be orphaned in GitHub — reconcile to clean up",
                repo=repo,
                draft_path=draft_path,
                verified_path=verified_path,
                submission_id=submission_id,
                notebook_id=notebook_id,
            )
        logger.info(
            "Notebook verified and published to GitHub",
            repo=repo,
            verified_path=verified_path,
            submission_id=submission_id,
            notebook_id=notebook_id,
            draft_cleanup_pending=not draft_deleted,
        )
        return {
            "path": verified_path,
            "filename": filename,
            "sha": result.get("content", {}).get("sha"),
            "draft_cleanup_pending": not draft_deleted,
        }
    except httpx.HTTPStatusError as e:
        logger.error(
            "GitHub API error publishing verified notebook",
            status=e.response.status_code,
            detail=e.response.text[:200],
            submission_id=submission_id,
            notebook_id=notebook_id,
        )
        raise GitHubPublishError(
            f"GitHub returned {e.response.status_code}: {e.response.text[:200]}"
        ) from e
    except Exception as e:
        logger.error(
            "Failed to publish verified notebook to GitHub",
            error=str(e),
            submission_id=submission_id,
            notebook_id=notebook_id,
        )
        raise GitHubPublishError(f"Unexpected error publishing to GitHub: {e}") from e



async def publish_verified_answer(
    answer_id: str,
    query: str,
    answer_json: dict[str, Any],
    reason: str | None = None,
    reviewer: str | None = None,
) -> dict[str, Any] | None:
    """Publish a verified quick answer to GitHub (issue #46 step 9).

    Counterpart to :func:`publish_verified` for notebooks. The file
    path is ``<verified_answers_folder>/<answer_id>.json`` — the
    ``answer_id`` IS the durable identity (filename === id, no
    metadata indirection needed because the file content IS the
    provenance).

    Returns the GitHub API response on success, or ``None`` if GitHub
    publishing is disabled. Raises :class:`GitHubPublishError` on
    failure (HTTP, network, or auth) so callers can refuse to commit
    local state that assumes the GitHub side succeeded — same
    write-through SSOT contract as the notebook publisher.

    Unlike :func:`publish_verified`, there is no drafts-to-verified
    move and therefore no ``draft_cleanup_pending`` flag. Answers have
    no draft state today — approve is the only publish moment.
    """
    settings = load_github_settings()
    if not is_publishing_active(settings):
        return None

    repo = settings["repo"]
    branch = settings["branch"]
    token = settings["token"]
    answers_folder = settings.get("verified_answers_folder", "verified-answers")
    path = f"{answers_folder}/{answer_id}.json"
    content = json.dumps(answer_json, indent=2).encode("utf-8")
    commit_msg = _build_commit_message(
        f"Verify answer: {query[:80]}", reason, reviewer
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await _create_or_update_file(
                client, repo, path, content, commit_msg, token, branch
            )
        logger.info(
            "Verified answer published to GitHub",
            repo=repo,
            path=path,
            answer_id=answer_id,
        )
        return {
            "path": path,
            "filename": f"{answer_id}.json",
            "sha": result.get("content", {}).get("sha"),
        }
    except httpx.HTTPStatusError as e:
        logger.error(
            "GitHub API error publishing verified answer",
            status=e.response.status_code,
            detail=e.response.text[:200],
            answer_id=answer_id,
        )
        raise GitHubPublishError(
            f"GitHub returned {e.response.status_code}: {e.response.text[:200]}"
        ) from e
    except Exception as e:
        logger.error(
            "Failed to publish verified answer to GitHub",
            error=str(e),
            answer_id=answer_id,
        )
        raise GitHubPublishError(f"Unexpected error publishing to GitHub: {e}") from e


async def remove_draft(
    submission_id: str,
    query: str,
    reason: str | None = None,
    reviewer: str | None = None,
) -> bool:
    """Remove a notebook from the drafts folder (e.g., on rejection).

    Returns True if deleted, False otherwise. ``reason`` and ``reviewer``
    are recorded in the deletion commit message so rejections leave the
    same audit trail as approvals (#112).
    """
    settings = load_github_settings()
    if not is_publishing_active(settings):
        return False

    repo = settings["repo"]
    branch = settings["branch"]
    token = settings["token"]
    folder = settings["drafts_folder"]
    filename = _notebook_filename(submission_id, query)
    path = f"{folder}/{filename}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            deleted = await _delete_file(
                client,
                repo,
                path,
                _build_commit_message(
                    f"Remove rejected draft: {query[:60]}", reason, reviewer
                ),
                token,
                branch,
            )
        if deleted:
            logger.info("Rejected draft removed from GitHub", path=path)
        return deleted
    except Exception as e:
        logger.error("Failed to remove draft from GitHub", error=str(e))
        return False


async def fetch_notebook(path: str) -> dict[str, Any] | None:
    """Fetch a notebook JSON file from the configured GitHub repo.

    Returns the parsed notebook JSON, or None if GitHub is disabled,
    the file is missing, or the request fails.
    """
    settings = load_github_settings()
    if not is_publishing_active(settings) or not path:
        return None

    repo = settings["repo"]
    branch = settings["branch"]
    token = settings["token"]
    url = f"{_api_url(repo)}/{path}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                url, headers=_headers(token), params={"ref": branch}
            )
        if resp.status_code != 200:
            logger.warning(
                "GitHub fetch returned non-200",
                status=resp.status_code,
                path=path,
            )
            return None
        data = resp.json()
        encoded = data.get("content", "")
        if not encoded:
            return None
        raw = base64.b64decode(encoded)
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to fetch notebook from GitHub", error=str(e), path=path)
        return None


async def _list_folder_files(
    folder: str, *, strict: bool = False, suffix: str = ".ipynb"
) -> list[dict[str, Any]]:
    """Internal: list files under a folder in the configured repo, filtered
    by ``suffix`` (default ``.ipynb`` for backwards-compatible behavior).

    Pass ``suffix=".json"`` (or whatever) for callers that publish
    something other than notebooks — e.g. ``bootstrap_answers_from_github``
    needs ``.json`` because each verified answer is published as
    ``<answer_id>.json``. Without the parameter, the old ``.ipynb``-only
    filter would silently drop every answer file and bootstrap would
    report ``checked: 0`` (an earlier adversarial review).

    Returns ``[]`` for an empty/nonexistent folder regardless of mode.

    * ``strict=False`` (default — for the lenient public ``list_*`` wrappers):
      swallow all other failures (disabled, auth, network, 5xx, malformed
      JSON, non-list response) and return ``[]``. Preserves the historical
      contract used by ``sync_all_from_github`` and similar background
      callers that prefer "fail soft → empty list" over surfacing
      transient errors.
    * ``strict=True`` (for ``reconcile_drafts_verified``): raise
      :class:`GitHubPublishError` on any failure that prevented us from
      reading the folder, including malformed JSON or a non-list response
      (which usually means the folder path is misconfigured and we landed
      on a file's metadata blob, e.g. ``contents/drafts/foo.ipynb`` rather
      than a directory listing). An admin endpoint must distinguish "I
      checked and there's nothing to do" from "I couldn't check at all".
    """
    settings = load_github_settings()
    if not is_publishing_active(settings):
        if strict:
            raise GitHubPublishError("GitHub publishing is not active (not configured or paused)")
        return []

    repo = settings["repo"]
    branch = settings["branch"]
    token = settings["token"]
    url = f"{_api_url(repo)}/{folder}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                url, headers=_headers(token), params={"ref": branch}
            )
    except Exception as e:
        logger.warning(f"Failed to list {folder} on GitHub", error=str(e))
        if strict:
            raise GitHubPublishError(
                f"Failed to list {folder} on GitHub: {e}"
            ) from e
        return []

    # A missing folder (404) is a legitimate "empty" state — e.g. no drafts
    # have ever been published — and is treated as empty in both modes.
    if resp.status_code == 404:
        return []
    if resp.status_code != 200:
        if strict:
            raise GitHubPublishError(
                f"GitHub returned {resp.status_code} listing {folder}: "
                f"{resp.text[:200]}"
            )
        return []

    # JSON parse + shape validation are inside the same lenient/strict
    # branch. A 200 response with invalid JSON or a non-list body (e.g. the
    # folder path actually pointed at a file, so GitHub returned an object
    # describing that file's metadata) is a genuine failure of the listing
    # contract — surface it in strict mode rather than reporting an empty
    # folder.
    try:
        data = resp.json()
    except Exception as e:
        logger.warning(
            f"Failed to parse JSON listing {folder} on GitHub", error=str(e)
        )
        if strict:
            raise GitHubPublishError(
                f"Could not parse GitHub response listing {folder}: {e}"
            ) from e
        return []

    if not isinstance(data, list):
        msg = (
            f"GitHub returned a non-list body listing {folder}; the folder "
            "path may be misconfigured or pointing at a file"
        )
        logger.warning(msg, folder=folder, response_type=type(data).__name__)
        if strict:
            raise GitHubPublishError(msg)
        return []

    return [
        {"name": f.get("name"), "path": f.get("path"), "sha": f.get("sha")}
        for f in data
        if f.get("type") == "file" and f.get("name", "").endswith(suffix)
    ]


async def list_verified_files() -> list[dict[str, Any]]:
    """List files in the verified folder of the configured repo.

    Returns a list of {name, path, sha} dicts, or [] when GitHub is
    disabled or the request fails. Lenient — see
    :func:`_list_folder_files` for the strict variant used by reconcile.
    """
    settings = load_github_settings()
    if not is_publishing_active(settings):
        return []
    return await _list_folder_files(settings["verified_folder"], strict=False)



async def list_draft_files() -> list[dict[str, Any]]:
    """List files in the drafts folder of the configured repo.

    Mirror of :func:`list_verified_files`; returns ``[]`` when GitHub is
    disabled or the request fails (e.g. the drafts folder doesn't exist).
    Used by :func:`reconcile_drafts_verified` (which uses the strict
    internal variant — see :func:`_list_folder_files`).
    """
    settings = load_github_settings()
    if not is_publishing_active(settings):
        return []
    return await _list_folder_files(settings["drafts_folder"], strict=False)


async def _is_file_gone(
    client: httpx.AsyncClient,
    repo: str,
    path: str,
    token: str,
    branch: str,
) -> bool:
    """Status-aware existence check: True only on a definite 404.

    Used by :func:`reconcile_drafts_verified` to disambiguate a
    ``_delete_file → False`` return: was the file already gone (a
    concurrent reconcile cleaned it — idempotent success), or did the
    delete actually fail (5xx, persistent 409 — real failure)?
    Returns False on any other response or on an exception, because we
    cannot prove the file is gone.
    """
    url = f"{_api_url(repo)}/{path}"
    try:
        resp = await client.get(url, headers=_headers(token), params={"ref": branch})
    except Exception:
        return False
    return resp.status_code == 404


async def reconcile_drafts_verified() -> dict[str, Any]:
    """Find and clean up drafts that are also present in the verified folder.

    The publish_verified() flow creates the verified file and then deletes
    the draft. If the delete failed (transient race, 5xx, draft never
    existed), the notebook can end up in BOTH folders. This function lists
    both folders, finds filenames that appear in both, and tries to delete
    the draft copy.

    Returns a report:
        {
            "checked": <n drafts inspected>,
            "duplicates_found": <n filenames in both folders>,
            "cleaned": [<draft path>, ...],
            "failed": [{"path": <draft path>, "reason": "..."}, ...],
            "disabled": <bool — true when GitHub publishing is off>,
        }

    Raises :class:`GitHubPublishError` when the underlying listing call
    fails (auth, network, GitHub 5xx). The endpoint maps that to HTTP 502
    so an admin sees the real failure instead of a misleading "checked 0"
    no-op report.

    Safe to call repeatedly: if a draft is already gone by the time we
    try to delete it (e.g. a concurrent reconcile or another writer), the
    existence check confirms the file is gone and we count it as cleaned
    rather than a failure — preserving the idempotency contract.
    Admin-only; exposed via POST /api/v1/notebooks/reconcile-drafts.
    """
    settings = load_github_settings()
    if not is_publishing_active(settings):
        return {
            "checked": 0,
            "duplicates_found": 0,
            "cleaned": [],
            "failed": [],
            "disabled": True,
        }

    repo = settings["repo"]
    branch = settings["branch"]
    token = settings["token"]
    drafts_folder = settings["drafts_folder"]
    verified_folder = settings["verified_folder"]

    # Strict listing — surfaces auth/network/5xx as GitHubPublishError so we
    # never report "nothing to do" when in fact we couldn't see GitHub.
    drafts = await _list_folder_files(drafts_folder, strict=True)
    verified = await _list_folder_files(verified_folder, strict=True)
    verified_names = {v["name"] for v in verified if v.get("name")}
    duplicates = [d for d in drafts if d.get("name") in verified_names]

    cleaned: list[str] = []
    failed: list[dict[str, str]] = []
    if duplicates:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for d in duplicates:
                path = d.get("path") or f"{drafts_folder}/{d.get('name')}"
                try:
                    ok = await _delete_file(
                        client,
                        repo,
                        path,
                        f"Reconcile: drop duplicate draft {d.get('name')}",
                        token,
                        branch,
                    )
                except Exception as e:
                    logger.warning(
                        "Reconcile draft delete raised",
                        path=path,
                        error=str(e),
                    )
                    failed.append({"path": path, "reason": str(e)})
                    continue
                if ok:
                    cleaned.append(path)
                    continue
                # _delete_file returned False — disambiguate with an explicit
                # existence check before reporting failure. If the file is
                # gone, a concurrent cleanup won the race and reconcile is
                # idempotent; if it's still there, the delete really failed.
                if await _is_file_gone(client, repo, path, token, branch):
                    cleaned.append(path)
                else:
                    failed.append(
                        {"path": path, "reason": "delete returned False"}
                    )

    logger.info(
        "Reconciled drafts vs verified",
        checked=len(drafts),
        duplicates_found=len(duplicates),
        cleaned=len(cleaned),
        failed=len(failed),
    )
    return {
        "checked": len(drafts),
        "duplicates_found": len(duplicates),
        "cleaned": cleaned,
        "failed": failed,
        "disabled": False,
    }


async def test_connection() -> dict[str, Any]:
    """Test the GitHub connection with current settings.

    Returns a dict with 'ok' bool and 'message' string.
    """
    settings = load_github_settings()
    if not settings.get("token"):
        return {"ok": False, "message": "No GitHub token configured"}

    repo = settings["repo"]
    token = settings["token"]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{repo}",
                headers=_headers(token),
            )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "ok": True,
                "message": f"Connected to {data['full_name']}",
                "repo_name": data["full_name"],
                "private": data["private"],
                "default_branch": data["default_branch"],
            }
        elif resp.status_code == 401:
            return {"ok": False, "message": "Invalid GitHub token"}
        elif resp.status_code == 404:
            return {"ok": False, "message": f"Repository '{repo}' not found"}
        else:
            return {"ok": False, "message": f"GitHub API error: {resp.status_code}"}
    except Exception as e:
        return {"ok": False, "message": f"Connection failed: {str(e)}"}
