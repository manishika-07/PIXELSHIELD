from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class ForensicsMetrics(BaseModel):
    ela_mean_error: float
    ela_max_error: float
    exif_software_detected: Optional[str] = None
    camera_hardware_present: bool
    fft_periodicity_anomaly: bool
    noise_variance_inconsistency: float
    lighting_gradient_consistent: bool

class ForensicsAuthenticityResponse(BaseModel):
    suspicion_band: Literal["Low suspicion", "Medium suspicion", "High suspicion"]
    confidence_score: float = Field(ge=0.0, le=100.0)
    ela_heatmap_base64: str
    metrics: ForensicsMetrics
    plain_language_explanation: str
    forensic_disclaimer: str
