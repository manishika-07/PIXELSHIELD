from typing import Optional, Dict, Literal
from pydantic import BaseModel, Field

class FingerprintSpec(BaseModel):
    version: str = "1.0"
    hash_types: list[str] = ["sha256", "phash", "dhash", "whash"]
    phash_size: int = 8
    dhash_size: int = 8
    description: str

class FingerprintResponse(BaseModel):
    sha256: str
    phash: str
    dhash: str
    whash: str
    spec: FingerprintSpec
    privacy_notice: str

class FingerprintMatchRequest(BaseModel):
    target: Dict[str, str]
    candidate: Dict[str, str]

class FingerprintMatchResponse(BaseModel):
    match_type: Literal["exact", "lookalike", "different"]
    hamming_distance_phash: int
    hamming_distance_dhash: int
    hamming_distance_whash: int
    confidence_pct: float
    explanation: str

class RegistryRegisterRequest(BaseModel):
    sha256: str
    phash: str
    dhash: Optional[str] = None
    whash: Optional[str] = None

class RegistryRegisterResponse(BaseModel):
    registered: bool
    registry_id: str
    timestamp: str

class RegistryCheckResponse(BaseModel):
    found: bool
    match_type: Optional[str] = None
    registered_at: Optional[str] = None
    registry_id: Optional[str] = None
