import io
import pytest
from PIL import Image
from backend.app.core.security import validate_image_bytes, validate_external_url, compute_evidence_seal, verify_evidence_seal
from backend.app.core.exceptions import PixelShieldException

def test_validate_image_bytes_valid_png(sample_portrait_png):
    img, mime = validate_image_bytes(sample_portrait_png)
    assert mime == "image/png"
    assert img.size == (256, 256)

def test_validate_image_bytes_valid_jpeg(sample_jpeg_bytes):
    img, mime = validate_image_bytes(sample_jpeg_bytes)
    assert mime == "image/jpeg"
    assert img.size == (128, 128)

def test_validate_image_bytes_reject_plain_text():
    with pytest.raises(PixelShieldException) as exc_info:
        validate_image_bytes(b"This is a text file pretending to be an image.")
    assert exc_info.value.code == "UNSUPPORTED_FORMAT"

def test_validate_image_bytes_too_large():
    fake_huge = b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024)
    with pytest.raises(PixelShieldException) as exc_info:
        validate_image_bytes(fake_huge)
    assert exc_info.value.code == "FILE_TOO_LARGE"

def test_ssrf_blocks_private_and_loopback():
    with pytest.raises(PixelShieldException) as exc:
        validate_external_url("http://127.0.0.1:8000/secret")
    assert exc.value.code == "SSRF_PROTECTION_BLOCKED"

    with pytest.raises(PixelShieldException) as exc:
        validate_external_url("http://localhost:3000")
    assert exc.value.code == "SSRF_PROTECTION_BLOCKED"

def test_evidence_seal_integrity():
    secret = "test_secret_key_123"
    data = b"screenshot_test_bytes"
    url = "https://example.com/post/1"
    ts = "2026-11-18T12:00:00Z"
    
    hash_val, seal = compute_evidence_seal(secret, data, url, ts)
    assert len(hash_val) == 64
    assert len(seal) == 64
    
    # Valid check
    assert verify_evidence_seal(secret, data, url, ts, seal) is True
    
    # Tampered data check
    tampered_data = b"screenshot_test_bytes_tampered"
    assert verify_evidence_seal(secret, tampered_data, url, ts, seal) is False
