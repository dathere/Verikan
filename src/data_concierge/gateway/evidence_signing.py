"""Ed25519ph signing + trust registry for Typed Standards evidence packages.

Implements the cryptographic envelope (Typed Standards Spec §8.3): the signer
Ed25519ph-signs the UTF-8 bytes of the package's envelope-hash hex string
(§8.3.1, step 4), reports an SPKI-DER public key, and the matching trust
registry is served at ``/.well-known/typed-publisher.json`` (§8.3.3) so any
verifier can resolve ``signature.kid`` to datHere's key independently of the
git host. This is the real signer that drops into the :class:`EvidenceSigner`
seam in ``gateway/evidence.py``.

**Why a vendored Ed25519ph.** The verifier (``@typedstandards/verify-core``)
checks signatures with ``ed25519ph`` — the RFC 8032 prehash variant with the
``dom2`` domain prefix. Neither ``cryptography`` nor PyNaCl exposes Ed25519ph,
and it cannot be synthesized from plain Ed25519 (the dom2 prefix feeds the
algorithm's internal hashes). So the ph routine below is a compact, dependency
-free RFC 8032 reference, **pinned to the official RFC 8032 §7.1 (pure) and
§7.3 (ph) test vectors** in ``tests/unit/test_evidence_signing.py``. It signs
short (64-char hex) messages, so the affine reference's performance is a
non-issue. It is used only for the publisher's signing key over *public*
evidence — not for secrets.

**Off by default.** :func:`get_active_signer` returns a real signer only when
``settings.evidence_signing_enabled`` is true *and* a key seed is configured;
otherwise it returns the builder's :class:`UnsignedSigner`. Key custody (env /
Secret Manager / KMS) is the deployer's decision — this module only reads the
already-resolved seed.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import UTC, datetime
from typing import Any

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.gateway.evidence import (
    EVIDENCE_HOST,
    TRUST_REGISTRY_URL,
    EvidenceSigner,
    SignatureEnvelope,
    UnsignedSigner,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# RFC 8032 Ed25519 / Ed25519ph reference (public-domain). Validated against the
# official test vectors (RFC 8032 §7.1 pure, §7.3 ph) — see the test module.
# ---------------------------------------------------------------------------
_b = 256
_q = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493

# dom2 prefix for Ed25519ph (phflag=1, empty context). RFC 8032 §5.1: for plain
# Ed25519 dom2 is the empty string; for the ph variant it is this prefix.
_DOM2_PH = b"SigEd25519 no Ed25519 collisions" + bytes([1]) + bytes([0])


def _sha512(m: bytes) -> bytes:
    return hashlib.sha512(m).digest()


def _sha512_int(m: bytes) -> int:
    return int.from_bytes(_sha512(m), "little")


def _inv(x: int) -> int:
    return pow(x, _q - 2, _q)


_d = (-121665 * _inv(121666)) % _q
_I = pow(2, (_q - 1) // 4, _q)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(_d * y * y + 1)
    x = pow(xx, (_q + 3) // 8, _q)
    if (x * x - xx) % _q != 0:
        x = (x * _I) % _q
    if x % 2 != 0:
        x = _q - x
    return x


_By = (4 * _inv(5)) % _q
_Bx = _xrecover(_By)
_B = (_Bx % _q, _By % _q)


def _edwards(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + _d * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - _d * x1 * x2 * y1 * y2)
    return (x3 % _q, y3 % _q)


def _scalarmult(p: tuple[int, int], e: int) -> tuple[int, int]:
    if e == 0:
        return (0, 1)
    q = _scalarmult(p, e // 2)
    q = _edwards(q, q)
    if e & 1:
        q = _edwards(q, p)
    return q


def _encodepoint(p: tuple[int, int]) -> bytes:
    x, y = p
    bits = [(y >> i) & 1 for i in range(_b - 1)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(_b // 8))


def _clamp_scalar(h: bytes) -> int:
    a: int = 2 ** (_b - 2)
    for i in range(3, _b - 2):
        a += (1 << i) * ((h[i // 8] >> (i % 8)) & 1)
    return int(a)


def _public_key_raw(seed: bytes) -> bytes:
    """Derive the 32-byte raw Ed25519 public key from a 32-byte seed."""
    h = _sha512(seed)
    a = _clamp_scalar(h)
    return _encodepoint(_scalarmult(_B, a))


def _sign(seed: bytes, msg: bytes, *, prehash: bool) -> bytes:
    """RFC 8032 Ed25519 / Ed25519ph signature (64 bytes)."""
    h = _sha512(seed)
    a = _clamp_scalar(h)
    prefix = h[_b // 8 : _b // 4]
    a_pub = _encodepoint(_scalarmult(_B, a))
    dom = _DOM2_PH if prehash else b""
    phm = _sha512(msg) if prehash else msg
    r = _sha512_int(dom + prefix + phm) % _L
    big_r = _encodepoint(_scalarmult(_B, r))
    k = _sha512_int(dom + big_r + a_pub + phm) % _L
    s = (r + k * a) % _L
    return big_r + s.to_bytes(_b // 8, "little")


# ---------------------------------------------------------------------------
# SPKI DER framing — the form the verifier's extractRawPublicKey expects
# ---------------------------------------------------------------------------
# Fixed 12-byte SPKI prefix for a bare Ed25519 public key (OID 1.3.101.112),
# followed by the 32-byte raw key → 44 bytes total.
_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


def _raw_pubkey_to_spki_b64(raw_pub: bytes) -> str:
    if len(raw_pub) != 32:
        raise ValueError("Ed25519 raw public key must be 32 bytes")
    return base64.b64encode(_ED25519_SPKI_PREFIX + raw_pub).decode("ascii")


def generate_seed_hex() -> str:
    """Generate a fresh 32-byte Ed25519 private seed as 64 hex chars."""
    return secrets.token_bytes(32).hex()


# ---------------------------------------------------------------------------
# Signer
# ---------------------------------------------------------------------------
class Ed25519phSigner:
    """Real :class:`EvidenceSigner`: Ed25519ph over the envelope-hash hex (§8.3.1)."""

    signed = True

    def __init__(self, seed_hex: str, kid: str) -> None:
        seed = bytes.fromhex(seed_hex)
        if len(seed) != 32:
            raise ValueError("Ed25519 seed must be 32 bytes (64 hex chars)")
        self._seed = seed
        self._kid = kid
        self._raw_pub = _public_key_raw(seed)
        self._spki_b64 = _raw_pubkey_to_spki_b64(self._raw_pub)

    @property
    def kid(self) -> str:
        return self._kid

    @property
    def public_key_spki_b64(self) -> str:
        return self._spki_b64

    def sign(self, envelope_hash: str) -> SignatureEnvelope:
        # §8.3.1 step 4: sign the UTF-8 bytes of the envelope-hash hex string.
        sig = _sign(self._seed, envelope_hash.encode("utf-8"), prehash=True)
        return SignatureEnvelope(
            signature=base64.b64encode(sig).decode("ascii"),
            publicKey=self._spki_b64,
            algorithm="Ed25519ph",
            kid=self._kid,
        )


# ---------------------------------------------------------------------------
# Factory + trust registry
# ---------------------------------------------------------------------------
def _resolve_seed_hex() -> str | None:
    seed = settings.evidence_signing_key_seed.get_secret_value().strip()
    return seed or None


def get_active_signer() -> EvidenceSigner:
    """Return the configured signer, or :class:`UnsignedSigner` when signing is off."""
    if not settings.evidence_signing_enabled:
        return UnsignedSigner()
    seed_hex = _resolve_seed_hex()
    if not seed_hex:
        logger.warning("Evidence signing enabled but no key seed configured; staying unsigned")
        return UnsignedSigner()
    try:
        return Ed25519phSigner(seed_hex, settings.evidence_signing_kid)
    except Exception as exc:  # bad seed shouldn't crash the publish path
        logger.error("Failed to initialize Ed25519ph signer; staying unsigned", error=str(exc))
        return UnsignedSigner()


def _signer_identity() -> dict[str, str]:
    return {
        "bindingTier": "platform",
        "identifier": f"platform:{EVIDENCE_HOST}",
        "displayName": settings.evidence_signer_display_name,
    }


def build_trust_registry() -> dict[str, Any]:
    """Build the §8.3.3 trust registry served at /.well-known/typed-publisher.json.

    Returns a ``{"keys": [...]}`` object. When signing is unconfigured the
    ``keys`` array is empty (honest: no key to vouch for yet).
    """
    keys: list[dict[str, Any]] = []
    seed_hex = _resolve_seed_hex()
    if settings.evidence_signing_enabled and seed_hex:
        try:
            signer = Ed25519phSigner(seed_hex, settings.evidence_signing_kid)
            keys.append(
                {
                    "kid": signer.kid,
                    "publicKey": signer.public_key_spki_b64,
                    "signerIdentity": _signer_identity(),
                    "status": "active",
                    "activatedAt": getattr(settings, "evidence_key_activated_at", None)
                    or datetime.now(UTC).date().isoformat() + "T00:00:00.000Z",
                    "deprecatedAt": None,
                    "revokedAt": None,
                }
            )
        except Exception as exc:
            logger.error("Trust registry key build failed", error=str(exc))
    return {"keys": keys}


def trust_registry_url() -> str:
    return TRUST_REGISTRY_URL
