import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def sample_portrait_png() -> bytes:
    """Generates a synthetic 256x256 test portrait image with skin-tone face oval."""
    img = Image.new("RGB", (256, 256), color=(200, 230, 220))
    draw = ImageDraw.Draw(img)
    # Draw face oval
    draw.ellipse([64, 40, 192, 216], fill=(235, 190, 160), outline=(200, 150, 120), width=2)
    # Draw eyes
    draw.ellipse([90, 95, 115, 115], fill=(255, 255, 255))
    draw.ellipse([100, 102, 108, 110], fill=(50, 50, 50))
    draw.ellipse([141, 95, 166, 115], fill=(255, 255, 255))
    draw.ellipse([148, 102, 156, 110], fill=(50, 50, 50))
    # Draw mouth
    draw.arc([105, 150, 151, 180], start=0, end=180, fill=(180, 70, 70), width=3)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
