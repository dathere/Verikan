"""RFC 3161 timestamp + Sigstore Rekor attestation legs (spec §8.3.2).

These add the two public-infrastructure proofs the verifier checks beyond the
signature: an RFC 3161 trusted timestamp (proves the package existed by a time)
and a Sigstore Rekor transparency-log entry (makes the signing event auditable).
Both are **best-effort and OFF by default** — failures are swallowed and the
corresponding commitment-view field is simply omitted (spec §8.3.2: "failures
persist as null and the package remains queryable").

What the verifier (`@typedstandards/verify-core`) expects, and what we produce:

- **RFC 3161 (check #7):** ``messageImprint.hashedMessage`` MUST equal the
  package hash under SHA-256. The package hash is itself a SHA-256 digest, so
  the imprint carries those 32 bytes directly (we do not re-hash). ``certReq``
  is TRUE so FreeTSA embeds its signing-cert chain (the verifier chains it to a
  pinned FreeTSA root). We store the extracted ``timeStampToken`` (the CMS
  SignedData ContentInfo), base64, as ``rfc3161Timestamp``.
- **Rekor (check #8):** a ``hashedrekord`` v0.0.1 entry whose
  ``spec.data.hash`` is ``{algorithm: "sha512", value: SHA-512(utf8(packageHashHex))}``
  — the Ed25519ph prehash, matching verify-core's ``rekorHashForPackage`` — with
  our Ed25519ph signature and the public key (PEM). We store ``rekorEntryId``
  and the inclusion proof (base64 JSON) as ``rekorInclusionProof``.

The TimeStampReq is a small fixed DER structure, hand-built here (no ASN.1 dep);
the token is extracted from the TimeStampResp with a minimal DER reader. Rekor is
plain JSON over HTTPS.
"""

from __future__ import annotations

import base64
import hashlib
import json

import httpx

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.gateway.evidence import SignatureEnvelope

logger = get_logger(__name__)

_SHA256_ALGID_DER = bytes.fromhex("300d06096086480165030402010500")  # SHA-256 + NULL


# ---------------------------------------------------------------------------
# Minimal DER (only what the TimeStampReq / TimeStampResp need)
# ---------------------------------------------------------------------------
def _der_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def _tlv(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + _der_len(len(value)) + value


def _read_tlv(data: bytes, off: int) -> tuple[int, bytes, int, bytes]:
    """Read one DER TLV at ``off``; return (tag, value, next_off, full_tlv)."""
    tag = data[off]
    i = off + 1
    length = data[i]
    i += 1
    if length & 0x80:
        nbytes = length & 0x7F
        length = int.from_bytes(data[i : i + nbytes], "big")
        i += nbytes
    value = data[i : i + length]
    return tag, value, i + length, data[off : i + length]


def build_tsq(package_hash_hex: str) -> bytes:
    """Build a DER RFC 3161 TimeStampReq for ``package_hash_hex`` (SHA-256 imprint)."""
    digest = bytes.fromhex(package_hash_hex)
    if len(digest) != 32:
        raise ValueError("package hash must be a 32-byte SHA-256 digest (64 hex chars)")
    version = _tlv(0x02, b"\x01")  # INTEGER 1
    hashed_message = _tlv(0x04, digest)  # OCTET STRING
    message_imprint = _tlv(0x30, _SHA256_ALGID_DER + hashed_message)
    cert_req = _tlv(0x01, b"\xff")  # BOOLEAN TRUE — embed the signing-cert chain
    return _tlv(0x30, version + message_imprint + cert_req)


def extract_timestamp_token(tsr_der: bytes) -> bytes | None:
    """Extract the ``timeStampToken`` (CMS ContentInfo) from a TimeStampResp.

    Returns ``None`` when the TSA did not grant (status not 0/1) or no token is
    present. The token's raw DER is what the verifier parses as SignedData.
    """
    try:
        _, body, _, _ = _read_tlv(tsr_der, 0)  # outer SEQUENCE
        # First child: PKIStatusInfo SEQUENCE { status INTEGER, ... }
        _, status_info, after_status, _ = _read_tlv(body, 0)
        _, status_val, _, _ = _read_tlv(status_info, 0)
        status = int.from_bytes(status_val, "big") if status_val else -1
        if status not in (0, 1):  # 0=granted, 1=grantedWithMods
            logger.warning("FreeTSA did not grant timestamp", status=status)
            return None
        if after_status >= len(body):
            return None
        # Next child: the timeStampToken (ContentInfo) — return its full TLV.
        _, _, _, token_tlv = _read_tlv(body, after_status)
        return token_tlv
    except Exception as exc:
        logger.warning("Failed to parse TimeStampResp", error=str(exc))
        return None


async def request_freetsa_timestamp(package_hash_hex: str) -> str | None:
    """Request an RFC 3161 token from FreeTSA; return base64 token or None."""
    try:
        tsq = build_tsq(package_hash_hex)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                settings.evidence_freetsa_url,
                content=tsq,
                headers={"Content-Type": "application/timestamp-query"},
            )
        if resp.status_code != 200:
            logger.warning("FreeTSA request failed", status=resp.status_code)
            return None
        token = extract_timestamp_token(resp.content)
        if token is None:
            return None
        return base64.b64encode(token).decode("ascii")
    except Exception as exc:
        logger.warning("FreeTSA timestamp leg failed (non-blocking)", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Sigstore Rekor (hashedrekord)
# ---------------------------------------------------------------------------
def rekor_hash_for_package(package_hash_hex: str) -> str:
    """SHA-512 hex of the UTF-8 package-hash hex — the Ed25519ph prehash Rekor stores.

    Mirrors verify-core's ``rekorHashForPackage`` exactly so producer and verifier
    compute the same value.
    """
    return hashlib.sha512(package_hash_hex.encode("utf-8")).hexdigest()


def _spki_b64_to_pem_b64(spki_b64: str) -> str:
    """Wrap a base64 SPKI-DER key as PEM, then base64 it (Rekor publicKey.content)."""
    body = "\n".join(spki_b64[i : i + 64] for i in range(0, len(spki_b64), 64))
    pem = f"-----BEGIN PUBLIC KEY-----\n{body}\n-----END PUBLIC KEY-----\n"
    return base64.b64encode(pem.encode("ascii")).decode("ascii")


def build_rekor_hashedrekord(package_hash_hex: str, signature: SignatureEnvelope) -> dict:
    """Build the Rekor ``hashedrekord`` v0.0.1 entry body for a signed package."""
    return {
        "apiVersion": "0.0.1",
        "kind": "hashedrekord",
        "spec": {
            "data": {
                "hash": {
                    "algorithm": "sha512",
                    "value": rekor_hash_for_package(package_hash_hex),
                }
            },
            "signature": {
                "content": signature.signature,
                "publicKey": {"content": _spki_b64_to_pem_b64(signature.publicKey)},
            },
        },
    }


async def submit_rekor_entry(
    package_hash_hex: str, signature: SignatureEnvelope
) -> tuple[str, str | None] | None:
    """Submit a hashedrekord entry to Rekor; return (entryId, inclusionProof_b64)."""
    if not signature.signature:
        return None  # unsigned package — nothing to log
    try:
        entry = build_rekor_hashedrekord(package_hash_hex, signature)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{settings.evidence_rekor_url}/api/v1/log/entries",
                json=entry,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
        if resp.status_code not in (200, 201):
            logger.warning("Rekor submission failed", status=resp.status_code)
            return None
        result = resp.json()
        entry_id = next(iter(result))
        record = result[entry_id]
        proof = record.get("verification", {}).get("inclusionProof")
        proof_b64 = (
            base64.b64encode(json.dumps(proof).encode("utf-8")).decode("ascii") if proof else None
        )
        return entry_id, proof_b64
    except Exception as exc:
        logger.warning("Rekor leg failed (non-blocking)", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def obtain_attestations(
    package_hash_hex: str, signature: SignatureEnvelope
) -> dict[str, str]:
    """Run the enabled attestation legs; return commitment-view fields to merge.

    Each leg is independent and best-effort: a failure (or a disabled leg) just
    omits its field. Keys: ``rfc3161Timestamp``, ``rekorEntryId``,
    ``rekorInclusionProof``.
    """
    out: dict[str, str] = {}
    if settings.evidence_timestamp_enabled:
        token = await request_freetsa_timestamp(package_hash_hex)
        if token:
            out["rfc3161Timestamp"] = token
    if settings.evidence_rekor_enabled:
        rekor = await submit_rekor_entry(package_hash_hex, signature)
        if rekor:
            out["rekorEntryId"] = rekor[0]
            if rekor[1]:
                out["rekorInclusionProof"] = rekor[1]
    return out
