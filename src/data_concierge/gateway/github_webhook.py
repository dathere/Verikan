"""GitHub webhook handler — issue #46 step 7 (push) + issue #74 (pull_request).

Receives webhook events from the verified-notebook GitHub repo and
keeps the local index in sync, completing the GitHub-as-SSOT loop
started in steps 4–6:

* Step 4 embedded provenance in ``nb.metadata.data_concierge``.
* Step 5 (bootstrap) rebuilt the local index from the full GitHub repo.
* Step 6 (sync_one) handled per-path upsert.
* Step 7 (this module) wires the per-path operations to GitHub push
  events so admin edits in GitHub propagate within seconds instead of
  requiring a manual bootstrap.
* Issue #74 added ``pull_request`` handling: lifecycle audit + admin
  notification, plus push-result annotation with the originating PR # so
  the admin UI can link straight to GitHub's native diff view.

Endpoint contract:

* ``POST /api/v1/github/webhook`` (public — GitHub calls this directly).
* GitHub sends three headers we care about:

  - ``X-Hub-Signature-256``: ``sha256=<hex>`` HMAC of the raw body
    using the shared ``webhook_secret``. Mismatch → 401.
  - ``X-GitHub-Event``: ``ping`` (pong response), ``push`` (processed),
    ``pull_request`` (audited + notified), anything else (ignored, 200).
  - ``X-GitHub-Delivery``: UUID. Deduplicated so redeliveries are no-ops.

* Push events: only pushes to the configured branch are processed.
  Within each commit, paths inside ``verified_folder`` (notebooks) AND
  ``verified_answers_folder`` (answers) are dispatched. ADD/MODIFY →
  upsert via :func:`sync_one_from_github` /
  :func:`sync_one_answer_from_github`; REMOVE → delete via
  :func:`delete_local_by_github_path` /
  :func:`delete_local_answer_by_github_path`. The LAST operation per
  path wins if multiple commits touch the same file. Paths outside
  both folders are ignored. When the push's head commit is a merge or
  squash-merge, the parsed PR # plus ``html_url`` (PR landing page)
  and ``diff_url`` (PR Files Changed tab) are surfaced in the
  response.

* Pull-request events: only ``opened``, ``reopened``, and ``closed``
  actions targeting the configured base branch are recorded. A
  per-event entry is appended to a persistent audit log
  (``_internal/github_pr_audit.json``, capped at the most recent 500
  entries) and an admin email is sent via
  :func:`notify_pr_event`. ``closed`` events are split into ``merged``
  vs ``closed_unmerged`` so the audit log and notifications can show
  the actual outcome. No local index mutation — file sync still flows
  through the push event GitHub fires on merge.

* Returns 503 when ``webhook_secret`` is not configured.

Idempotency: seen ``X-GitHub-Delivery`` IDs are persisted in storage
(capped at the most recent 1000) so a redelivery — push or
pull_request — does not re-process the same event.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/github", tags=["github-webhook"])

# Idempotency: seen delivery IDs persisted across restarts.
_DELIVERIES_KEY = "_internal/github_webhook_deliveries.json"
_MAX_DELIVERIES = 1000


def _load_seen_deliveries() -> list[str]:
    try:
        data = storage.read_json(_DELIVERIES_KEY)
    except Exception as e:
        logger.warning("Failed to load seen-deliveries log", error=str(e))
        return []
    if not isinstance(data, dict):
        return []
    ids = data.get("ids")
    if not isinstance(ids, list):
        return []
    return [str(x) for x in ids if isinstance(x, (str, int))]


def _mark_delivery_seen(delivery_id: str) -> None:
    """Append ``delivery_id`` to the persisted seen-deliveries log.

    Capped at the most recent ``_MAX_DELIVERIES`` entries to keep the
    file bounded. Best-effort: a storage failure logs and continues so
    the webhook still processes the event (the cost of duplicate work
    on a future redelivery is small compared to losing a real push).
    """
    ids = _load_seen_deliveries()
    if delivery_id in ids:
        return
    ids.append(delivery_id)
    if len(ids) > _MAX_DELIVERIES:
        ids = ids[-_MAX_DELIVERIES:]
    try:
        storage.write_json(_DELIVERIES_KEY, {"ids": ids})
    except Exception as e:
        logger.warning(
            "Failed to persist seen delivery — webhook will still proceed",
            delivery=delivery_id,
            error=str(e),
        )


def _is_delivery_seen(delivery_id: str) -> bool:
    return delivery_id in _load_seen_deliveries()


def _verify_signature(
    secret: str, body: bytes, signature_header: str | None
) -> bool:
    """HMAC-SHA256 constant-time comparison.

    Returns False on any of: empty secret, missing header, wrong prefix,
    mismatch. Constant-time comparison prevents timing-based secret
    extraction.
    """
    if not secret or not signature_header:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def _collect_path_actions(
    payload: dict[str, Any], folder_prefix: str
) -> dict[str, str]:
    """Walk commits in push order; return ``{path: "upsert"|"delete"}``.

    Within a single push, a path can appear in multiple buckets across
    commits (e.g. added in commit A, modified in commit B, removed in
    commit C). The LAST operation per path wins — that matches the
    final on-GitHub state we need to mirror locally.

    Only paths starting with ``folder_prefix`` (the verified-notebook
    folder + trailing slash) are included — pushes that don't touch
    verified notebooks at all return an empty dict.
    """
    final_actions: dict[str, str] = {}
    for commit in payload.get("commits") or []:
        if not isinstance(commit, dict):
            continue
        for p in commit.get("added") or []:
            if isinstance(p, str) and p.startswith(folder_prefix):
                final_actions[p] = "upsert"
        for p in commit.get("modified") or []:
            if isinstance(p, str) and p.startswith(folder_prefix):
                final_actions[p] = "upsert"
        for p in commit.get("removed") or []:
            if isinstance(p, str) and p.startswith(folder_prefix):
                final_actions[p] = "delete"
    return final_actions



# PR audit log: persisted JSON list of recent pull_request events for
# admin visibility. Capped to bound storage; older entries roll off.
_PR_AUDIT_KEY = "_internal/github_pr_audit.json"
_MAX_PR_AUDIT = 500

# GitHub default merge-commit subject:  "Merge pull request #N from owner/branch"
# GitHub default squash-merge subject:  "Title text (#N)"
# Rebase merges leave no PR # in the message — those simply won't annotate.
_MERGE_PR_PATTERN = re.compile(r"^Merge pull request #(\d+) from ")
_SQUASH_PR_PATTERN = re.compile(r"\(#(\d+)\)\s*$")


def _record_pr_event(entry: dict[str, Any]) -> None:
    """Append a PR lifecycle event to the persistent audit log.

    Best-effort: a storage failure is logged but never raised — losing
    an audit row is preferable to failing the webhook and triggering a
    GitHub retry storm.
    """
    try:
        data = storage.read_json(_PR_AUDIT_KEY)
    except Exception:
        data = None
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        entries = []
    entries.append(entry)
    if len(entries) > _MAX_PR_AUDIT:
        entries = entries[-_MAX_PR_AUDIT:]
    try:
        storage.write_json(_PR_AUDIT_KEY, {"entries": entries})
    except Exception as e:
        logger.warning(
            "Failed to persist PR audit entry",
            error=str(e),
            pr=entry.get("pr_number"),
        )


def _parse_pr_number_from_commit_message(message: str | None) -> int | None:
    """Best-effort PR # extraction from a merge or squash-merge commit subject.

    Returns None for rebase merges (no PR # in message) or any commit
    that doesn't match GitHub's default merge-message templates.
    """
    if not message:
        return None
    first_line = message.splitlines()[0]
    for pat in (_MERGE_PR_PATTERN, _SQUASH_PR_PATTERN):
        m = pat.search(first_line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def _pr_urls_from_repository(
    repository: dict[str, Any] | None, pr_number: int | None
) -> tuple[str, str]:
    """Build (html_url, diff_url) for a PR given the push payload's repository
    object and a parsed PR number. Returns ("", "") if either is missing.

    ``diff_url`` points at the Files Changed tab so admins land on
    GitHub's native diff view directly.
    """
    if not pr_number or not isinstance(repository, dict):
        return ("", "")
    repo_url = str(repository.get("html_url") or "").rstrip("/")
    if not repo_url:
        return ("", "")
    html_url = f"{repo_url}/pull/{pr_number}"
    return (html_url, f"{html_url}/files")


async def _handle_pull_request_event(
    *,
    body: bytes,
    gh_settings: dict[str, Any],
    settings_status: str,
    x_github_delivery: str | None,
) -> dict[str, Any]:
    """Audit + notify for a verified ``pull_request`` event.

    Mirrors the push branch's pre-dispatch flow (dedup → JSON parse →
    pause guard) so the two event types share retry semantics:

    * Dedup before parse — redeliveries are cheap no-ops.
    * Fail-closed settings → 503 so GitHub retries.
    * Admin pause → 200 ``skipped_paused`` so deliveries don't pile up
      in GitHub's retry queue while paused.
    * Off-target ``base.ref`` and uninteresting actions → 200 ``ignored``
      and mark seen, so a redelivery doesn't re-evaluate.

    Only the actions that materially change a PR's state are recorded
    (``opened``, ``reopened``, ``closed``). Drive-by edits like
    ``synchronize``, ``edited``, label changes, and review activity are
    treated as noise here — push events already carry the file-sync
    signal we care about.
    """
    # Imports here to avoid circular imports at module load.
    from data_concierge.gateway.admin_notifications import notify_pr_event
    from data_concierge.gateway.github_publisher import is_publishing_active

    # Dedup BEFORE parse so retries cost only a storage read.
    if x_github_delivery and _is_delivery_seen(x_github_delivery):
        logger.info(
            "GitHub webhook: duplicate PR delivery skipped",
            delivery=x_github_delivery,
        )
        return {
            "status": "duplicate_delivery",
            "delivery": x_github_delivery,
        }

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON body: {e}",
        ) from e
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pull_request event payload must be a JSON object",
        )

    # Pause guard — same retry-vs-pause distinction as push (roborev
    # #2569, #2572). A fail-closed settings load is transient; an
    # intentional admin pause is steady state.
    if not is_publishing_active(gh_settings):
        if settings_status == "fail_closed":
            logger.error(
                "GitHub webhook: settings unreadable on PR event; "
                "returning 503 so GitHub retries",
                delivery=x_github_delivery,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "GitHub settings are temporarily unreadable; "
                    "cannot dispatch webhook. Retry after storage recovers."
                ),
            )
        logger.info(
            "GitHub webhook: publishing paused; skipping PR audit",
            delivery=x_github_delivery,
        )
        return {
            "status": "skipped_paused",
            "delivery": x_github_delivery,
        }

    action = str(payload.get("action") or "")
    pr = payload.get("pull_request")
    if not isinstance(pr, dict):
        if x_github_delivery:
            _mark_delivery_seen(x_github_delivery)
        return {
            "status": "ignored",
            "reason": "missing pull_request object",
            "action": action,
        }

    # Action filter: only lifecycle-relevant transitions. Mark seen on
    # uninteresting actions so retries don't re-evaluate the filter.
    if action not in ("opened", "reopened", "closed"):
        logger.info(
            "GitHub webhook: ignoring PR action",
            action=action,
            delivery=x_github_delivery,
        )
        if x_github_delivery:
            _mark_delivery_seen(x_github_delivery)
        return {
            "status": "ignored",
            "reason": "uninteresting action",
            "action": action,
        }

    # Base-branch filter: only PRs targeting the configured branch.
    configured_branch = gh_settings.get("branch") or "main"
    base_ref = ""
    base = pr.get("base")
    if isinstance(base, dict):
        base_ref = str(base.get("ref") or "")
    if base_ref != configured_branch:
        logger.info(
            "GitHub webhook: ignoring PR against non-target base",
            base_ref=base_ref,
            expected=configured_branch,
            delivery=x_github_delivery,
        )
        if x_github_delivery:
            _mark_delivery_seen(x_github_delivery)
        return {
            "status": "ignored",
            "reason": "base ref mismatch",
            "base_ref": base_ref,
            "expected": configured_branch,
        }

    pr_number = pr.get("number") if isinstance(pr.get("number"), int) else None
    pr_title = str(pr.get("title") or "")
    pr_html_url = str(pr.get("html_url") or "").rstrip("/")
    pr_diff_url = f"{pr_html_url}/files" if pr_html_url else ""
    merged = bool(pr.get("merged"))
    head_ref = ""
    head = pr.get("head")
    if isinstance(head, dict):
        head_ref = str(head.get("ref") or "")
    user = pr.get("user")
    author = str(user.get("login") or "") if isinstance(user, dict) else ""

    # Derive a sub-action ``outcome`` so admins can quickly distinguish
    # a merge from a close-without-merge — both share action=closed.
    if action == "closed":
        outcome = "merged" if merged else "closed_unmerged"
    else:
        outcome = action

    entry: dict[str, Any] = {
        "delivery": x_github_delivery,
        "received_at": datetime.now(UTC).isoformat(),
        "action": action,
        "outcome": outcome,
        "pr_number": pr_number,
        "title": pr_title,
        "author": author,
        "html_url": pr_html_url,
        "diff_url": pr_diff_url,
        "base_ref": base_ref,
        "head_ref": head_ref,
        "merged": merged,
    }
    _record_pr_event(entry)

    # Notification is fire-and-forget — it has its own try/except
    # internally but we also guard here so a notification import-time
    # failure can't poison the audit success.
    try:
        await notify_pr_event(
            action=action,
            outcome=outcome,
            pr_number=pr_number,
            title=pr_title,
            author=author,
            html_url=pr_html_url,
            diff_url=pr_diff_url,
            base_ref=base_ref,
        )
    except Exception as e:
        logger.warning(
            "Failed to dispatch PR admin notification",
            error=str(e),
            pr=pr_number,
        )

    if x_github_delivery:
        _mark_delivery_seen(x_github_delivery)

    logger.info(
        "GitHub webhook: PR event processed",
        delivery=x_github_delivery,
        action=action,
        outcome=outcome,
        pr=pr_number,
        base_ref=base_ref,
    )
    return {
        "status": "processed",
        "event": "pull_request",
        "delivery": x_github_delivery,
        "action": action,
        "outcome": outcome,
        "pr_number": pr_number,
        "html_url": pr_html_url,
        "diff_url": pr_diff_url,
        "base_ref": base_ref,
    }


@router.post("/webhook")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
) -> dict[str, Any]:
    """GitHub webhook handler. See module docstring for the full contract."""
    # Imports here to avoid circular imports at module load.
    from data_concierge.gateway.github_publisher import (
        is_publishing_active,
        load_github_settings_with_status,
    )
    from data_concierge.gateway.verified_notebooks import (
        delete_local_answer_by_github_path,
        delete_local_by_github_path,
        sync_one_answer_from_github,
        sync_one_from_github,
    )

    gh_settings, settings_status = load_github_settings_with_status()
    secret = gh_settings.get("webhook_secret") or ""
    if not secret:
        # Distinct from 401: a missing secret means the operator hasn't
        # finished configuring the webhook, not that someone is sending
        # bad signatures.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GitHub webhook is not configured. "
                "Set ``webhook_secret`` in GitHub settings."
            ),
        )

    body = await request.body()
    if not _verify_signature(secret, body, x_hub_signature_256):
        logger.warning(
            "GitHub webhook: signature verification failed",
            delivery=x_github_delivery,
            gh_event=x_github_event,
            has_header=bool(x_hub_signature_256),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Hub-Signature-256",
        )

    # Ping: GitHub sends one when the webhook is first configured.
    # Reply 200 so the configuration UI shows green.
    if x_github_event == "ping":
        logger.info(
            "GitHub webhook: ping received", delivery=x_github_delivery
        )
        return {"status": "pong", "delivery": x_github_delivery}

    # Pull-request lifecycle events: audit + admin notification, no
    # local index mutation (file sync still flows through the push event
    # produced by a merge). Handled before the non-push fallback so it
    # can use a self-contained mini-flow that reuses dedup + pause-guard
    # but skips the path-dispatch machinery.
    if x_github_event == "pull_request":
        return await _handle_pull_request_event(
            body=body,
            gh_settings=gh_settings,
            settings_status=settings_status,
            x_github_delivery=x_github_delivery,
        )

    if x_github_event != "push":
        logger.info(
            "GitHub webhook: ignoring non-push event",
            gh_event=x_github_event,
            delivery=x_github_delivery,
        )
        return {"status": "ignored", "event": x_github_event}

    # Idempotency: dedupe based on delivery ID. Do this BEFORE parsing
    # the body so we cheaply skip retries.
    if x_github_delivery and _is_delivery_seen(x_github_delivery):
        logger.info(
            "GitHub webhook: duplicate delivery skipped",
            delivery=x_github_delivery,
        )
        return {
            "status": "duplicate_delivery",
            "delivery": x_github_delivery,
        }

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON body: {e}",
        ) from e
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Push event payload must be a JSON object",
        )

    # Pause kill-switch: halt ALL push dispatch — upserts AND deletes —
    # while publishing is paused. Without this guard, ``sync_one_*``
    # short-circuits on its own (the inline ``is_publishing_active``
    # check returns ``skipped_disabled``), but ``delete_local_*`` runs
    # unconditionally and would mutate local state even though the UI
    # says all GitHub I/O is halted (an earlier adversarial review).
    #
    # This MUST run BEFORE the branch filter (an earlier adversarial review). If the
    # settings load failed closed, ``gh_settings`` holds env/default
    # values; comparing the push ref against a fail-closed default
    # branch would treat a legitimate push to the real (unreadable)
    # branch as "ref mismatch" and prematurely mark the delivery seen.
    #
    # HTTP status matters for retry semantics (an earlier adversarial review):
    # GitHub treats 2xx as "delivered successfully, do not retry" and
    # non-2xx as retryable up to ~8 times over ~8 hours.
    # - settings_status == "fail_closed" is a TRANSIENT error
    #   (corrupt file, brief storage outage); raise 503 so GitHub
    #   retries and the delivery isn't lost when storage recovers.
    # - Intentional admin pause is a STEADY state; return 200 with
    #   status="skipped_paused" without marking seen. Drops here are
    #   expected — the admin uses "Sync from GitHub" to catch up on
    #   resume.
    if not is_publishing_active(gh_settings):
        if settings_status == "fail_closed":
            logger.error(
                "GitHub webhook: settings unreadable; returning 503 "
                "so GitHub retries the delivery",
                delivery=x_github_delivery,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "GitHub settings are temporarily unreadable; "
                    "cannot dispatch webhook. Retry after storage recovers."
                ),
            )
        logger.info(
            "GitHub webhook: publishing paused; skipping push dispatch",
            delivery=x_github_delivery,
        )
        return {
            "status": "skipped_paused",
            "delivery": x_github_delivery,
        }

    # Branch filter: only process pushes to our configured branch.
    configured_branch = gh_settings.get("branch") or "main"
    expected_ref = f"refs/heads/{configured_branch}"
    if payload.get("ref") != expected_ref:
        logger.info(
            "GitHub webhook: ignoring push to non-target ref",
            ref=payload.get("ref"),
            expected=expected_ref,
            delivery=x_github_delivery,
        )
        # Mark seen so identical retries don't re-evaluate the filter.
        if x_github_delivery:
            _mark_delivery_seen(x_github_delivery)
        return {
            "status": "ignored",
            "reason": "ref mismatch",
            "ref": payload.get("ref"),
            "expected": expected_ref,
        }

    # Collect actions for BOTH content types in this push. The prefixes
    # are folder-name + "/" so ``verified-answers/foo.json`` can't
    # accidentally match the ``verified/`` filter (``verified/`` doesn't
    # match ``verified-answers/...`` because the slash differs).
    verified_folder = gh_settings.get("verified_folder") or "verified"
    notebook_prefix = verified_folder.rstrip("/") + "/"
    answers_folder = gh_settings.get("verified_answers_folder") or "verified-answers"
    answer_prefix = answers_folder.rstrip("/") + "/"

    notebook_actions = _collect_path_actions(payload, notebook_prefix)
    answer_actions = _collect_path_actions(payload, answer_prefix)

    upserts: list[dict[str, Any]] = []
    deletes: list[dict[str, Any]] = []

    # Notebook dispatch (step 7 behavior, unchanged).
    for path, action in notebook_actions.items():
        if action == "upsert":
            try:
                upserts.append(await sync_one_from_github(path))
            except Exception as e:
                logger.error(
                    "GitHub webhook: sync_one (notebook) raised",
                    path=path,
                    error=str(e),
                )
                upserts.append({"status": "failed", "path": path, "reason": str(e)})
        else:
            try:
                deletes.append(delete_local_by_github_path(path))
            except Exception as e:
                logger.error(
                    "GitHub webhook: delete (notebook) raised",
                    path=path,
                    error=str(e),
                )
                deletes.append({"status": "failed", "path": path, "reason": str(e)})

    # Answer dispatch (issue #46 step 9 PR C). Mirrors the notebook
    # branch — same retry-on-failed semantics, same per-path isolation.
    for path, action in answer_actions.items():
        if action == "upsert":
            try:
                upserts.append(await sync_one_answer_from_github(path))
            except Exception as e:
                logger.error(
                    "GitHub webhook: sync_one (answer) raised",
                    path=path,
                    error=str(e),
                )
                upserts.append({"status": "failed", "path": path, "reason": str(e)})
        else:
            try:
                deletes.append(delete_local_answer_by_github_path(path))
            except Exception as e:
                logger.error(
                    "GitHub webhook: delete (answer) raised",
                    path=path,
                    error=str(e),
                )
                deletes.append({"status": "failed", "path": path, "reason": str(e)})

    # Idempotency vs retryability: mark this delivery seen ONLY if every
    # dispatched op succeeded (or was a deterministic skip). If any path
    # came back ``status="failed"`` — a transient GitHub fetch error,
    # network glitch, or sync_one/delete raising — leave the delivery
    # UNMARKED so a GitHub redelivery can retry. Otherwise a single
    # transient failure would mark the local index permanently stale
    # for that path. Successful upserts on retry are idempotent (step 6
    # is upsert-shaped), so re-running them is cheap and correct.
    any_failed = any(
        r.get("status") == "failed" for r in (*upserts, *deletes)
    )
    if x_github_delivery and not any_failed:
        _mark_delivery_seen(x_github_delivery)

    # Push annotation: when the head commit is a PR merge (or squash
    # merge), surface the PR # + URLs so the admin UI can link straight
    # to GitHub's native diff view for the originating PR. Rebase merges
    # leave no PR # in the commit message and will surface None.
    head_commit = payload.get("head_commit") or {}
    head_message = (
        head_commit.get("message") if isinstance(head_commit, dict) else None
    )
    pr_number = _parse_pr_number_from_commit_message(head_message)
    pr_html_url, pr_diff_url = _pr_urls_from_repository(
        payload.get("repository"), pr_number
    )

    logger.info(
        "GitHub webhook: push processed",
        delivery=x_github_delivery,
        ref=expected_ref,
        upserts=len(upserts),
        deletes=len(deletes),
        any_failed=any_failed,
        marked_seen=bool(x_github_delivery and not any_failed),
        pr_number=pr_number,
    )
    return {
        "status": "processed",
        "delivery": x_github_delivery,
        "ref": expected_ref,
        "upserts": upserts,
        "deletes": deletes,
        "any_failed": any_failed,
        "pr_number": pr_number,
        "pr_html_url": pr_html_url,
        "pr_diff_url": pr_diff_url,
    }
