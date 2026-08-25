"""Tests for ``webhook_secret`` handling in the GitHub settings endpoint
(issue #46, step 7 — an earlier adversarial review finding 2).

The webhook handler in ``gateway/github_webhook.py`` reads
``webhook_secret`` from ``load_github_settings()``. Admins need to be
able to set it through the existing ``/api/v1/settings/github``
endpoint, with two operational rules:

* GET never echoes the raw secret (security) — only exposes a
  ``webhook_secret_set`` boolean.
* POST follows a "blank = preserve" convention (matches how the
  ``token`` field already works) — admins who can't see the current
  secret must not accidentally wipe it by submitting a partial form.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_concierge.gateway import github_publisher
from data_concierge.gateway.router import (
    GitHubSettingsRequest,
    get_github_settings,
    update_github_settings,
)


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(github_publisher, "storage", test_storage)


class TestGetExposesMaskedStatus:
    async def test_no_secret_returns_webhook_secret_set_false(
        self, tmp_storage: None
    ) -> None:
        body = await get_github_settings(_admin={"user": "admin"})
        assert body["webhook_secret_set"] is False

    async def test_existing_secret_returns_webhook_secret_set_true(
        self, tmp_storage: None
    ) -> None:
        github_publisher.save_github_settings(
            {**github_publisher.load_github_settings(),
             "webhook_secret": "very-secret-value"}
        )
        body = await get_github_settings(_admin={"user": "admin"})
        assert body["webhook_secret_set"] is True
        # The raw secret must NEVER appear in any field of the response.
        assert "very-secret-value" not in str(body)


class TestPostWebhookSecretWriting:
    async def test_non_empty_value_replaces_secret(
        self, tmp_storage: None
    ) -> None:
        await update_github_settings(
            request=GitHubSettingsRequest(
                enabled=True,
                repo="owner/repo",
                webhook_secret="initial-secret",
            ),
            _admin={"user": "admin"},
        )
        assert (
            github_publisher.load_github_settings().get("webhook_secret")
            == "initial-secret"
        )

        # Rotate by submitting a new non-empty value.
        await update_github_settings(
            request=GitHubSettingsRequest(
                enabled=True,
                repo="owner/repo",
                webhook_secret="rotated-secret",
            ),
            _admin={"user": "admin"},
        )
        assert (
            github_publisher.load_github_settings().get("webhook_secret")
            == "rotated-secret"
        )

    async def test_blank_preserves_existing_secret(
        self, tmp_storage: None
    ) -> None:
        """Critical correctness property: admins can't see the current
        webhook_secret (GET masks it), so a blank submission MUST NOT
        clear it — same rule as the ``token`` field."""
        github_publisher.save_github_settings(
            {**github_publisher.load_github_settings(),
             "webhook_secret": "stays-the-same"}
        )

        await update_github_settings(
            request=GitHubSettingsRequest(
                enabled=True,
                repo="owner/repo",
                webhook_secret="",  # blank
            ),
            _admin={"user": "admin"},
        )
        assert (
            github_publisher.load_github_settings().get("webhook_secret")
            == "stays-the-same"
        )

    async def test_post_response_reports_webhook_secret_set(
        self, tmp_storage: None
    ) -> None:
        body = await update_github_settings(
            request=GitHubSettingsRequest(
                enabled=True,
                repo="owner/repo",
                webhook_secret="new",
            ),
            _admin={"user": "admin"},
        )
        assert body["webhook_secret_set"] is True
