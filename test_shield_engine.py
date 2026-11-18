import io
import time
import pytest
from PIL import Image

def test_health_and_ready(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert "models" in ready.json()

def test_shield_job_lifecycle(client, sample_portrait_png):
    # 1. Initiate Shield Job
    files = {"image": ("test.png", sample_portrait_png, "image/png")}
    data = {"strength": "50", "mode": "both", "steps": "10"}
    init_res = client.post("/api/v1/shield", files=files, data=data)
    assert init_res.status_code == 202
    res_data = init_res.json()
    assert "job_id" in res_data
    assert res_data["status"] == "queued"
    
    job_id = res_data["job_id"]
    
    # 2. Poll until completed (or timeout)
    max_wait = 30
    start = time.time()
    completed = False
    
    while time.time() - start < max_wait:
        poll_res = client.get(f"/api/v1/jobs/{job_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()
        status = poll_data["status"]
        if status == "completed":
            completed = True
            assert poll_data["progress"] == 100
            assert poll_data["metrics"] is not None
            assert poll_data["metrics"]["psnr_db"] > 20.0 # visually good quality
            assert poll_data["metrics"]["ssim"] > 0.85
            break
        elif status == "failed":
            pytest.fail(f"Job failed: {poll_data.get('error')}")
        time.sleep(0.5)
        
    assert completed is True, "Job did not complete in time"
    
    # 3. Download Result (One-time fetch)
    dl_res = client.get(f"/api/v1/jobs/{job_id}/result")
    assert dl_res.status_code == 200
    assert dl_res.headers["content-type"] == "image/png"
    assert len(dl_res.content) > 100
    
    # Verify downloaded image is valid
    out_img = Image.open(io.BytesIO(dl_res.content))
    assert out_img.size == (256, 256)
    
    # 4. Verify Second Download is 410 Gone (Privacy guarantee)
    second_dl = client.get(f"/api/v1/jobs/{job_id}/result")
    assert second_dl.status_code == 410
