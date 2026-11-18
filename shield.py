from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class ShieldInitResponse(BaseModel):
    job_id: str
    status: str = "queued"
    estimated_sec: float
    message: str

class ShieldMetrics(BaseModel):
    psnr_db: float
    ssim: float
    embedding_distance_before_after: Optional[float] = None
    latent_distance_before_after: Optional[float] = None
    face_detected: bool = True
    fallback: bool = False

class ShieldJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    created_at: float
    metrics: Optional[ShieldMetrics] = None
    error: Optional[str] = None
    limits_note: str

class PlatformSurvival(BaseModel):
    survival_ratio: float
    rating: Literal["Strong", "Medium", "Weaker"]
    retained_effect_pct: float

class StressTestResponse(BaseModel):
    overall_score: float = Field(ge=0.0, le=10.0)
    platforms: Dict[str, PlatformSurvival]
    honest_limits: str
    measured_at_utc: str
