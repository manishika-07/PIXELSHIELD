import io
import hmac
import hashlib
import ipaddress
import socket
from urllib.parse import urlparse
from typing import Tuple, Optional
from PIL import Image
from backend.app.config import settings
from backend.app.core.exceptions import PixelShieldException

# Set decompression bomb guard
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

def validate_image_bytes(raw_bytes: bytes) -> Tuple[Image.Image, str]:
    """
    Validates uploaded image bytes against magic bytes, maximum file size,
    and decompression bombs. Rejects non-image or animated payloads gracefully.
    Returns: (PIL.Image, mime_type)
    """
    if len(raw_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise PixelShieldException(
            status_code=400,
            code="FILE_TOO_LARGE",
            message=f"Uploaded image exceeds the 10 MB limit ({len(raw_bytes)} bytes).",
            hint="Please compress or resize the photo before uploading."
        )
    
    if len(raw_bytes) < 12:
        raise PixelShieldException(
            status_code=400,
            code="INVALID_IMAGE_FILE",
            message="Uploaded file is too small to be a valid image.",
            hint="Please select a valid JPEG, PNG, or WebP photo."
        )
    
    # Check Magic Bytes
    mime = None
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise PixelShieldException(
            status_code=400,
            code="UNSUPPORTED_FORMAT",
            message="Unsupported file format. Only JPEG, PNG, and WebP images are accepted.",
            hint="Convert your photo to PNG or JPEG before uploading."
        )
        
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify() # Verify integrity
        # Re-open after verify() as Pillow docs recommend
        img = Image.open(io.BytesIO(raw_bytes))
        
        # Guard against animated GIF/WebP
        if getattr(img, "is_animated", False):
            raise PixelShieldException(
                status_code=400,
                code="ANIMATED_IMAGE_REJECTED",
                message="Animated images are not supported for deepfake cloaking.",
                hint="Please upload a single still photograph."
            )
            
        return img, mime
    except PixelShieldException:
        raise
    except Image.DecompressionBombError:
        raise PixelShieldException(
            status_code=400,
            code="DECOMPRESSION_BOMB_DETECTED",
            message="Image resolution is dangerously high (decompression bomb risk).",
            hint="Maximum allowed resolution is 16 megapixels."
        )
    except Exception as e:
        raise PixelShieldException(
            status_code=400,
            code="CORRUPTED_IMAGE",
            message="The uploaded image appears corrupted and could not be decoded.",
            hint=str(e)
        )

def validate_external_url(url: str) -> str:
    """
    Validates an external URL to prevent SSRF (Server-Side Request Forgery).
    Blocks private IP addresses, loopback addresses, link-local, and cloud metadata endpoints.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("Only http and https protocols are permitted.")
        
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Invalid URL hostname.")
            
        # Resolve hostname to IP
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)
        
        # Check against prohibited ranges
        if (
            ip.is_loopback or 
            ip.is_private or 
            ip.is_link_local or 
            ip.is_multicast or 
            ip.is_reserved or
            ip_str.startswith("169.254.") # AWS / GCP metadata
        ):
            raise PixelShieldException(
                status_code=400,
                code="SSRF_PROTECTION_BLOCKED",
                message="Target URL resolves to a private, loopback, or cloud-internal IP address.",
                hint="Only public external URLs may be archived as evidence."
            )
            
        return url
    except PixelShieldException:
        raise
    except Exception as e:
        raise PixelShieldException(
            status_code=400,
            code="INVALID_URL",
            message=f"Invalid target URL: {str(e)}",
            hint="Please provide a valid, publicly reachable web address (e.g. https://instagram.com/p/...)"
        )

def compute_evidence_seal(secret_key: str, screenshot_bytes: bytes, target_url: str, server_iso_timestamp: str) -> Tuple[str, str]:
    """
    Calculates SHA-256 hash of screenshot and HMAC-SHA256 seal.
    Returns: (screenshot_hash, hmac_seal)
    """
    screenshot_hash = hashlib.sha256(screenshot_bytes).hexdigest()
    seal_payload = f"{screenshot_hash}|{target_url}|{server_iso_timestamp}".encode("utf-8")
    seal = hmac.new(secret_key.encode("utf-8"), seal_payload, hashlib.sha256).hexdigest()
    return screenshot_hash, seal

def verify_evidence_seal(secret_key: str, screenshot_bytes: bytes, target_url: str, server_iso_timestamp: str, stored_seal: str) -> bool:
    """
    Verifies HMAC seal using constant-time comparison to detect tampering.
    """
    _, calculated_seal = compute_evidence_seal(secret_key, screenshot_bytes, target_url, server_iso_timestamp)
    return hmac.compare_digest(calculated_seal, stored_seal)
