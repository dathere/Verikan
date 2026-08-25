"""Admin email notifications for new submissions.

Sends a notification email to configured admin recipients whenever a new
notebook or quick-answer submission is created. Configuration is via
SMTP environment variables on ``Settings`` and is fully optional — when
disabled or unconfigured, notifications are silently skipped.
"""

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger

logger = get_logger(__name__)


def _recipients() -> list[str]:
    raw = (settings.admin_notification_recipients or "").strip()
    if not raw:
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


def _send_smtp(subject: str, body: str, to: list[str]) -> None:
    if not (settings.smtp_host and settings.smtp_from_email and to):
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from_email
    msg["To"] = ", ".join(to)
    msg.set_content(body)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
        if settings.smtp_use_tls:
            smtp.starttls()
        if settings.smtp_username:
            smtp.login(
                settings.smtp_username,
                settings.smtp_password.get_secret_value(),
            )
        smtp.send_message(msg)


async def notify_new_submission(
    *,
    kind: str,
    query: str,
    submitted_by: str,
    submitter_email: str | None,
    submission_id: str,
) -> None:
    """Send an admin notification for a new submission.

    Runs the blocking SMTP send in a thread so it does not block the
    request event loop. All errors are logged but never raised.
    """
    if not settings.admin_notifications_enabled:
        return

    recipients = _recipients()
    if not recipients:
        logger.debug("Admin notifications enabled but no recipients configured")
        return

    label = "Notebook" if kind == "notebook" else "Quick Answer"
    submitter_line = submitted_by
    if submitter_email and submitter_email != submitted_by:
        submitter_line = f"{submitted_by} <{submitter_email}>"

    base = (settings.app_base_url or "").rstrip("/")
    review_url = f"{base}/admin" if base else "/admin"

    subject = f"[Data Concierge] New {label.lower()} submission to review"
    body = (
        f"A new {label.lower()} has been submitted for admin review.\n\n"
        f"Submitter: {submitter_line}\n"
        f"Question: {query}\n"
        f"Submission ID: {submission_id}\n\n"
        f"Review it here: {review_url}\n"
    )

    try:
        await asyncio.to_thread(_send_smtp, subject, body, recipients)
        logger.info(
            "Admin notification sent",
            kind=kind,
            recipients=len(recipients),
            submission_id=submission_id,
        )
    except Exception as e:
        logger.warning(
            "Failed to send admin notification",
            error=str(e),
            kind=kind,
            submission_id=submission_id,
        )


async def notify_pr_event(
    *,
    action: str,
    outcome: str,
    pr_number: int | None,
    title: str,
    author: str,
    html_url: str,
    diff_url: str,
    base_ref: str,
) -> None:
    """Send an admin notification for a pull_request lifecycle event.

    ``outcome`` is the post-filtered action label — ``opened``,
    ``reopened``, ``merged``, or ``closed_unmerged``. ``diff_url``
    points at the PR's Files Changed tab so the link drops the admin
    straight into GitHub's native diff view.

    Mirrors ``notify_new_submission``: async, blocking SMTP runs in a
    thread, all errors swallowed and logged.
    """
    if not settings.admin_notifications_enabled:
        return

    recipients = _recipients()
    if not recipients:
        logger.debug("Admin notifications enabled but no recipients configured")
        return

    pr_label = f"#{pr_number}" if pr_number else "(unknown number)"
    outcome_human = {
        "opened": "opened",
        "reopened": "reopened",
        "merged": "merged",
        "closed_unmerged": "closed without merging",
    }.get(outcome, outcome)

    subject = f"[Data Concierge] Notebook repo PR {pr_label} {outcome_human}"
    lines = [
        f"A pull request against ``{base_ref}`` was {outcome_human}.",
        "",
        f"PR: {pr_label} — {title}",
        f"Author: {author or '(unknown)'}",
        f"Action: {action}",
    ]
    if html_url:
        lines.append(f"View PR:    {html_url}")
    if diff_url:
        lines.append(f"View diff:  {diff_url}")
    body = "\n".join(lines) + "\n"

    try:
        await asyncio.to_thread(_send_smtp, subject, body, recipients)
        logger.info(
            "Admin PR notification sent",
            recipients=len(recipients),
            pr=pr_number,
            outcome=outcome,
        )
    except Exception as e:
        logger.warning(
            "Failed to send admin PR notification",
            error=str(e),
            pr=pr_number,
            outcome=outcome,
        )
