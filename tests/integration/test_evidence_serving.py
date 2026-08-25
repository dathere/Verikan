"""The public, verifier-facing evidence endpoints (commitment / package / registry).

The typedstandards.org browser verifier resolves these exact paths cross-origin
and rejects any non-JSON content-type — so these assert JSON bodies, an open CORS
header, and the round-trip from stored artifact to served response.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_concierge.data_layer import storage as storage_module
from data_concierge.data_layer.storage import LocalStorage
from data_concierge.gateway import evidence_router
from data_concierge.gateway.evidence import (
    canonicalize_jcs,
    commitment_endpoint_url,
    evidence_commitment_storage_key,
    evidence_package_storage_key,
    package_endpoint_url,
)


@pytest.fixture
def client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    test_storage = LocalStorage(tmp_path)
    # The endpoints lazily `from ...storage import storage`, resolving the
    # module-level singleton — patch it so reads hit the test filesystem.
    monkeypatch.setattr(storage_module, "storage", test_storage)
    app = FastAPI()
    app.include_router(evidence_router.router)
    return TestClient(app)


PKG = {"metadata": {"schemaVersion": "0.1.0"}, "contentProfile": "datHere", "output": "hi"}
PKG_HASH = hashlib.sha256(canonicalize_jcs(PKG)).hexdigest()
COMMITMENT = {
    "evidenceProtocolVersion": "0.1.0",
    "packageHash": PKG_HASH,
    "packageUrl": package_endpoint_url(PKG_HASH),
    "signature": {
        "signature": "x",
        "publicKey": "y",
        "algorithm": "Ed25519ph",
        "kid": "dathere:k1",
    },
    "trustRegistryUrl": "https://data-concierge.dathere.com/.well-known/typed-publisher.json",
}


def _seed(tmp_storage: LocalStorage) -> None:
    tmp_storage.write_json(evidence_package_storage_key(PKG_HASH), PKG)
    tmp_storage.write_json(evidence_commitment_storage_key(PKG_HASH), COMMITMENT)


class TestCommitmentEndpoint:
    def test_serves_commitment_as_json_with_cors(self, client, tmp_path) -> None:
        _seed(LocalStorage(tmp_path))
        resp = client.get(f"/api/evidence/{PKG_HASH}/commitment")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        assert resp.headers["access-control-allow-origin"] == "*"
        body = resp.json()
        assert body["packageHash"] == PKG_HASH
        assert body["signature"]["kid"] == "dathere:k1"

    def test_404_when_absent(self, client) -> None:
        resp = client.get("/api/evidence/" + "0" * 64 + "/commitment")
        assert resp.status_code == 404
        assert resp.headers["access-control-allow-origin"] == "*"


class TestPackageEndpoint:
    def test_serves_package_that_rehashes_to_commitment(self, client, tmp_path) -> None:
        _seed(LocalStorage(tmp_path))
        resp = client.get(f"/api/evidence/{PKG_HASH}/package")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        # The verifier's core invariant: the served package re-hashes to the
        # commitment's packageHash.
        served = resp.json()
        assert hashlib.sha256(canonicalize_jcs(served)).hexdigest() == PKG_HASH

    def test_packageurl_points_at_this_endpoint(self) -> None:
        assert package_endpoint_url(PKG_HASH).endswith(f"/api/evidence/{PKG_HASH}/package")
        assert commitment_endpoint_url(PKG_HASH).endswith(f"/api/evidence/{PKG_HASH}/commitment")


class TestRegistryEndpoint:
    def test_registry_served_with_cors(self, client) -> None:
        resp = client.get("/.well-known/typed-publisher.json")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        assert resp.headers["access-control-allow-origin"] == "*"
        assert "keys" in resp.json()

    def test_legacy_alias(self, client) -> None:
        resp = client.get("/.well-known/evidence-public-keys.json")
        assert resp.status_code == 200
        assert "keys" in resp.json()
