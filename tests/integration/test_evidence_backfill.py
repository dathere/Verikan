"""Backfill: build + store evidence packages for already-approved notebooks.

Deploying does not retro-generate packages, so the backfill endpoint makes the
whole existing verified library verifiable in one pass. This drives the real
endpoint against isolated storage with signing enabled and asserts each notebook
gets a signed, servable, content-addressed commitment + package.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any

import pytest
from pydantic import SecretStr

import data_concierge.gateway.router  # noqa: F401  (ensure module is imported)
from data_concierge.core.config import settings
from data_concierge.data_layer.storage import LocalStorage
from data_concierge.gateway import verified_notebooks as vn
from data_concierge.gateway.evidence import (
    canonicalize_jcs,
    evidence_commitment_storage_key,
    evidence_package_storage_key,
)

# The gateway package re-exports the `router` APIRouter, shadowing the module
# attribute — fetch the real module object from sys.modules to patch its globals.
router_module = sys.modules["data_concierge.gateway.router"]

# RFC 8032 §7.3 seed — a real Ed25519 key so packages actually sign.
_SEED = "833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42"
_NB = {
    "cells": [{"cell_type": "code", "source": ["print(1)"]}],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


@pytest.fixture
def storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> LocalStorage:
    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(vn, "storage", test_storage)
    monkeypatch.setattr(router_module, "storage", test_storage)
    # Enable signing with a real key so the backfill stores signed packages.
    monkeypatch.setattr(settings, "evidence_signing_enabled", True)
    monkeypatch.setattr(settings, "evidence_signing_key_seed", SecretStr(_SEED))
    monkeypatch.setattr(settings, "evidence_signing_kid", "dathere:data-concierge-2026")
    return test_storage


def _make_verified(query: str) -> None:
    sub = vn.submit_notebook(query=query, answer="ans", notebook_json=_NB, filename="n.ipynb")
    vn.approve_notebook(sub.submission_id, reviewed_by="admin", admin_notes="ok")


async def test_backfill_stores_signed_packages_for_all(storage: LocalStorage) -> None:
    _make_verified("What is the violent crime rate in Pittsburgh?")
    _make_verified("Show 311 noise complaints by neighborhood")

    result = await router_module.backfill_evidence_packages(_admin={"user": "admin"})

    assert result["count"] == 2
    assert result["stored"] == 2
    for entry in result["results"]:
        assert entry["signed"] is True
        assert entry["commitmentUrl"].endswith("/commitment")
        assert entry["verifyUrl"].startswith("https://typedstandards.org/verify?url=")
        h = entry["packageHash"]
        # Stored + servable, and the canonical package re-hashes to packageHash.
        commitment = storage.read_json(evidence_commitment_storage_key(h))
        package = storage.read_json(evidence_package_storage_key(h))
        assert commitment["packageHash"] == h
        assert commitment["signature"]["kid"] == "dathere:data-concierge-2026"
        assert commitment["packageUrl"].endswith(f"/api/evidence/{h}/package")
        assert hashlib.sha256(canonicalize_jcs(package)).hexdigest() == h


async def test_list_endpoint_exposes_verify_url_after_backfill(storage: LocalStorage) -> None:
    _make_verified("a query")
    await router_module.backfill_evidence_packages(_admin={"user": "admin"})
    listing = await router_module.list_verified_notebooks()
    row = listing["notebooks"][0]
    assert row["evidence_package_hash"]
    assert row["evidence_verify_url"].startswith("https://typedstandards.org/verify?url=https://")
    assert row["evidence_verify_url"].endswith("/commitment")


async def test_list_endpoint_verify_url_none_before_backfill(storage: LocalStorage) -> None:
    _make_verified("a query")
    listing = await router_module.list_verified_notebooks()
    assert listing["notebooks"][0]["evidence_verify_url"] is None


async def test_backfill_idempotent(storage: LocalStorage) -> None:
    _make_verified("a query")
    first = await router_module.backfill_evidence_packages(_admin={"user": "admin"})
    second = await router_module.backfill_evidence_packages(_admin={"user": "admin"})
    # Content-addressed: same hash both runs.
    assert first["results"][0]["packageHash"] == second["results"][0]["packageHash"]


async def test_backfill_skips_when_signing_off(
    storage: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "evidence_signing_enabled", False)
    _make_verified("a query")
    result = await router_module.backfill_evidence_packages(_admin={"user": "admin"})
    assert result["count"] == 1
    assert result["stored"] == 0  # unsigned -> not stored
    assert result["results"][0]["signed"] is False
