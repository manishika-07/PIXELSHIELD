import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
from backend.app.config import settings
from backend.app.core.exceptions import PixelShieldException

@dataclass
class JobRecord:
    job_id: str
    status: str # queued, processing, completed, failed
    progress: int # 0 to 100
    created_at: float
    result_bytes: Optional[bytes] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    fallback: bool = False
    downloaded: bool = False

class ShieldJobStore:
    def __init__(self, ttl_seconds: int = 600):
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds
        
    def create_job(self, job_id: str) -> JobRecord:
        with self._lock:
            self._purge_expired_locked()
            record = JobRecord(
                job_id=job_id,
                status="queued",
                progress=5,
                created_at=time.time()
            )
            self._jobs[job_id] = record
            return record

    def update_progress(self, job_id: str, progress: int, status: str = "processing"):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress = min(100, max(0, progress))
                self._jobs[job_id].status = status

    def complete_job(self, job_id: str, result_bytes: bytes, metrics: Dict[str, Any], fallback: bool = False):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = "completed"
                job.progress = 100
                job.result_bytes = result_bytes
                job.metrics = metrics
                job.fallback = fallback

    def fail_job(self, job_id: str, error_message: str):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = error_message
                job.result_bytes = None

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            self._purge_expired_locked()
            return self._jobs.get(job_id)

    def retrieve_and_delete_result(self, job_id: str) -> bytes:
        """
        One-time download: returns result image bytes and immediately purges them
        from memory to fulfill privacy-by-design guarantees.
        """
        with self._lock:
            if job_id not in self._jobs:
                raise PixelShieldException(
                    status_code=404,
                    code="JOB_NOT_FOUND",
                    message="Shielding job not found or expired.",
                    hint="Please submit your photo again."
                )
            job = self._jobs[job_id]
            if job.status != "completed":
                raise PixelShieldException(
                    status_code=400,
                    code="JOB_NOT_READY",
                    message=f"Job is currently {job.status} ({job.progress}%). Result is not yet available."
                )
            if job.downloaded or job.result_bytes is None:
                raise PixelShieldException(
                    status_code=410,
                    code="ALREADY_DOWNLOADED",
                    message="Protected image has already been downloaded and purged from memory for privacy.",
                    hint="Photo Shield never retains images after download."
                )
            
            # Extract and immediately erase bytes from storage
            data = job.result_bytes
            job.result_bytes = None
            job.downloaded = True
            return data

    def _purge_expired_locked(self):
        now = time.time()
        expired_ids = [
            jid for jid, job in self._jobs.items()
            if (now - job.created_at) > self._ttl_seconds
        ]
        for jid in expired_ids:
            del self._jobs[jid]

# Global singleton
job_store = ShieldJobStore(ttl_seconds=settings.JOB_TTL_SECONDS)
