"""Tests for Ed25519ph evidence signing + the trust registry.

The vendored RFC 8032 routine is pinned to the official §7.3 Ed25519ph test
vector — this is the proof the produced signatures are exactly what the Typed
Standards verifier (`@typedstandards/verify-core`, noble `ed25519ph`) accepts.
"""

import base64

import pytest

from data_concierge.core.config import settings
from data_concierge.gateway.evidence import UnsignedSigner
from data_concierge.gateway.evidence_signing import (
    Ed25519phSigner,
    _public_key_raw,
    _sign,
    build_trust_registry,
    generate_seed_hex,
    get_active_signer,
)

# RFC 8032 §7.3 — the single official Ed25519ph test vector.
RFC_PH_SEED = "833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42"
RFC_PH_PUBKEY = "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf"
RFC_PH_SIG = (
    "98a70222f0b8121aa9d30f813d683f809e462b469c7ff87639499bb94e6dae41"
    "31f85042463c2a355a2003d062adf5aaa10b8c61e636062aaad11c2a26083406"
)


class TestRFC8032Vector:
    def test_public_key_derivation(self) -> None:
        assert _public_key_raw(bytes.fromhex(RFC_PH_SEED)).hex() == RFC_PH_PUBKEY

    def test_ph_signature_matches_vector(self) -> None:
        sig = _sign(bytes.fromhex(RFC_PH_SEED), b"abc", prehash=True)
        assert sig.hex() == RFC_PH_SIG

    def test_pure_and_ph_differ(self) -> None:
        seed = bytes.fromhex(RFC_PH_SEED)
        assert _sign(seed, b"abc", prehash=True) != _sign(seed, b"abc", prehash=False)


class TestEd25519phSigner:
    def test_signs_envelope_hash(self) -> None:
        signer = Ed25519phSigner(RFC_PH_SEED, "dathere:test")
        env = signer.sign("deadbeef" * 8)
        assert env.algorithm == "Ed25519ph"
        assert env.kid == "dathere:test"
        assert signer.signed
        sig_bytes = base64.b64decode(env.signature)
        assert len(sig_bytes) == 64

    def test_spki_public_key_shape(self) -> None:
        signer = Ed25519phSigner(RFC_PH_SEED, "dathere:test")
        der = base64.b64decode(signer.public_key_spki_b64)
        assert len(der) == 44  # 12-byte prefix + 32-byte key
        assert der[:12] == bytes.fromhex("302a300506032b6570032100")
        assert der[12:].hex() == RFC_PH_PUBKEY

    def test_envelope_kid_matches_signer(self) -> None:
        signer = Ed25519phSigner(RFC_PH_SEED, "dathere:abc")
        assert signer.sign("00").kid == "dathere:abc"

    def test_rejects_bad_seed(self) -> None:
        with pytest.raises(ValueError):
            Ed25519phSigner("abcd", "kid")  # too short

    def test_generated_seed_is_64_hex(self) -> None:
        seed = generate_seed_hex()
        assert len(seed) == 64
        assert bytes.fromhex(seed)  # valid hex, 32 bytes


class TestActiveSignerFactory:
    def test_disabled_returns_unsigned(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "evidence_signing_enabled", False)
        assert isinstance(get_active_signer(), UnsignedSigner)

    def test_enabled_without_seed_stays_unsigned(self, monkeypatch) -> None:
        from pydantic import SecretStr

        monkeypatch.setattr(settings, "evidence_signing_enabled", True)
        monkeypatch.setattr(settings, "evidence_signing_key_seed", SecretStr(""))
        assert isinstance(get_active_signer(), UnsignedSigner)

    def test_enabled_with_seed_returns_real_signer(self, monkeypatch) -> None:
        from pydantic import SecretStr

        monkeypatch.setattr(settings, "evidence_signing_enabled", True)
        monkeypatch.setattr(settings, "evidence_signing_key_seed", SecretStr(RFC_PH_SEED))
        monkeypatch.setattr(settings, "evidence_signing_kid", "dathere:k1")
        signer = get_active_signer()
        assert isinstance(signer, Ed25519phSigner)
        assert signer.kid == "dathere:k1"


class TestTrustRegistry:
    def test_empty_when_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "evidence_signing_enabled", False)
        assert build_trust_registry() == {"keys": []}

    def test_populated_when_enabled(self, monkeypatch) -> None:
        from pydantic import SecretStr

        monkeypatch.setattr(settings, "evidence_signing_enabled", True)
        monkeypatch.setattr(settings, "evidence_signing_key_seed", SecretStr(RFC_PH_SEED))
        monkeypatch.setattr(settings, "evidence_signing_kid", "dathere:k1")
        reg = build_trust_registry()
        assert len(reg["keys"]) == 1
        key = reg["keys"][0]
        assert key["kid"] == "dathere:k1"
        assert key["status"] == "active"
        assert key["deprecatedAt"] is None and key["revokedAt"] is None
        # public key in the registry must match the signer's SPKI
        der = base64.b64decode(key["publicKey"])
        assert der[12:].hex() == RFC_PH_PUBKEY
        assert key["signerIdentity"]["bindingTier"] == "platform"
