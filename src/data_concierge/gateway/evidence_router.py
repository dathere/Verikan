"""Public, verifier-facing evidence endpoints (Typed Standards §8.3.3, §8.8).

Mounted at the site root (no ``/api/v1`` prefix) because the typedstandards.org
verifier resolves these exact paths:

- ``/.well-known/typed-publisher.json`` — the trust registry (§8.3.3); lets a
  verifier resolve a package's ``signature.kid`` to datHere's public key.
- ``/api/evidence/{hash}/commitment`` — the commitment view (§8.8.1); the URL a
  reader pastes into the verifier. The verifier reads ``packageHash`` +
  ``signature`` + ``trustRegistryUrl`` here.
- ``/api/evidence/{hash}/package`` — the canonical package (§8.1) the
  commitment's ``packageUrl`` points at; the verifier re-hashes it.

All three are public and CORS-open: the browser verifier fetches them
cross-origin, and they expose only already-signed evidence (the signature, not
the transport, is the trust). They return ``application/json`` — the verifier
rejects any other content-type, which is why a GitHub-raw URL (``text/plain``)
cannot serve as a commitment/package URL.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

# The browser verifier does a simple cross-origin GET (no custom headers, so no
# preflight); an open ACAO on the response is sufficient.
_CORS = {"Access-Control-Allow-Origin": "*"}


@router.get("/.well-known/typed-publisher.json")
@router.get("/.well-known/evidence-public-keys.json")  # legacy alias (spec §8.3.3)
async def typed_publisher_registry() -> JSONResponse:
    """Serve datHere's Typed Standards trust registry (spec §8.3.3).

    Empty ``keys`` until a signing key is configured. Canonical and legacy
    well-known paths return byte-identical content.
    """
    from data_concierge.gateway.evidence_signing import build_trust_registry

    return JSONResponse(build_trust_registry(), headers=_CORS)


@router.get("/api/evidence/{package_hash}/commitment")
async def evidence_commitment(package_hash: str) -> JSONResponse:
    """Serve a published package's commitment view (spec §8.8.1) as JSON."""
    from data_concierge.data_layer.storage import storage
    from data_concierge.gateway.evidence import evidence_commitment_storage_key

    data = storage.read_json(evidence_commitment_storage_key(package_hash))
    if not data:
        return JSONResponse({"error": "not found"}, status_code=404, headers=_CORS)
    return JSONResponse(data, headers=_CORS)


@router.get("/api/evidence/{package_hash}/package")
async def evidence_package(package_hash: str) -> JSONResponse:
    """Serve the canonical evidence package (spec §8.1) as JSON.

    The commitment view's ``packageUrl`` points here; the verifier fetches it and
    recomputes ``sha256(JCS(package))`` to confirm it equals ``packageHash``.
    """
    from data_concierge.data_layer.storage import storage
    from data_concierge.gateway.evidence import evidence_package_storage_key

    data = storage.read_json(evidence_package_storage_key(package_hash))
    if not data:
        return JSONResponse({"error": "not found"}, status_code=404, headers=_CORS)
    return JSONResponse(data, headers=_CORS)
