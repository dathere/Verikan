"""Tests for the RFC 3161 + Rekor attestation legs (spec §8.3.2).

The pure builders are fully testable offline: the TSQ DER round-trips through
the minimal reader, the token extractor handles a synthetic TimeStampResp, and
the Rekor hash matches verify-core's `rekorHashForPackage` formula exactly.
"""

import base64
import hashlib

from data_concierge.gateway.evidence import SignatureEnvelope
from data_concierge.gateway.evidence_attestation import (
    _read_tlv,
    _tlv,
    build_rekor_hashedrekord,
    build_tsq,
    extract_timestamp_token,
    rekor_hash_for_package,
)

PKG_HASH = hashlib.sha256(b"some package").hexdigest()


class TestRfc3161Request:
    def test_tsq_is_wellformed_der(self) -> None:
        tsq = build_tsq(PKG_HASH)
        tag, body, end, _ = _read_tlv(tsq, 0)
        assert tag == 0x30  # SEQUENCE
        assert end == len(tsq)
        # version INTEGER 1
        vtag, vval, after_v, _ = _read_tlv(body, 0)
        assert vtag == 0x02 and vval == b"\x01"
        # messageImprint SEQUENCE
        itag, imprint, after_i, _ = _read_tlv(body, after_v)
        assert itag == 0x30
        # certReq BOOLEAN TRUE
        ctag, cval, _, _ = _read_tlv(body, after_i)
        assert ctag == 0x01 and cval == b"\xff"

    def test_tsq_imprint_carries_package_hash_under_sha256(self) -> None:
        tsq = build_tsq(PKG_HASH)
        _, body, _, _ = _read_tlv(tsq, 0)
        _, _, after_v, _ = _read_tlv(body, 0)
        _, imprint, _, _ = _read_tlv(body, after_v)
        # imprint = AlgorithmIdentifier(SHA-256) + OCTET STRING(hash)
        atag, _, after_alg, _ = _read_tlv(imprint, 0)
        assert atag == 0x30  # AlgorithmIdentifier
        otag, oval, _, _ = _read_tlv(imprint, after_alg)
        assert otag == 0x04  # OCTET STRING
        assert oval == bytes.fromhex(PKG_HASH)  # the package hash, not re-hashed

    def test_extract_token_from_granted_response(self) -> None:
        # Synthetic TimeStampResp: SEQUENCE { PKIStatusInfo{INTEGER 0}, token SEQUENCE }
        status_info = _tlv(0x30, _tlv(0x02, b"\x00"))
        token = _tlv(0x30, b"\x05\x00")  # a stand-in ContentInfo SEQUENCE
        tsr = _tlv(0x30, status_info + token)
        assert extract_timestamp_token(tsr) == token

    def test_extract_token_rejects_non_granted(self) -> None:
        status_info = _tlv(0x30, _tlv(0x02, b"\x02"))  # status 2 = rejection
        token = _tlv(0x30, b"\x05\x00")
        tsr = _tlv(0x30, status_info + token)
        assert extract_timestamp_token(tsr) is None

    def test_extract_token_absent(self) -> None:
        tsr = _tlv(0x30, _tlv(0x30, _tlv(0x02, b"\x00")))  # status only, no token
        assert extract_timestamp_token(tsr) is None


class TestRekor:
    def test_hash_matches_verify_core_formula(self) -> None:
        # verify-core: rekorHashForPackage = SHA-512(utf8(packageHashHex))
        expected = hashlib.sha512(PKG_HASH.encode("utf-8")).hexdigest()
        assert rekor_hash_for_package(PKG_HASH) == expected

    def test_hashedrekord_shape(self) -> None:
        sig = SignatureEnvelope(
            signature="c2ln",
            publicKey=base64.b64encode(b"x" * 44).decode(),
            algorithm="Ed25519ph",
            kid="dathere:k1",
        )
        entry = build_rekor_hashedrekord(PKG_HASH, sig)
        assert entry["kind"] == "hashedrekord"
        assert entry["apiVersion"] == "0.0.1"
        assert entry["spec"]["data"]["hash"]["algorithm"] == "sha512"
        assert entry["spec"]["data"]["hash"]["value"] == rekor_hash_for_package(PKG_HASH)
        assert entry["spec"]["signature"]["content"] == "c2ln"
        # publicKey.content is base64 of a PEM block
        pem = base64.b64decode(entry["spec"]["signature"]["publicKey"]["content"]).decode()
        assert pem.startswith("-----BEGIN PUBLIC KEY-----")
        assert pem.strip().endswith("-----END PUBLIC KEY-----")
