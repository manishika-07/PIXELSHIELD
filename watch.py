from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class WatchSubscribeRequest(BaseModel):
    device_token: str
    consent: bool = True
    hashes: List[str] # List of pHash/sha256 hex strings

class WatchSubscribeResponse(BaseModel):
    subscription_id: str
    active: bool
    monitored_hashes_count: int
    privacy_guarantee: str
    next_scheduled_scan: str

class MatchItem(BaseModel):
    url: str
    similarity_score: float
    matched_hash: str
    source_title: Optional[str] = "External Web Appearance"
    detected_at: str
    report_payload: Dict[str, Any]

class WatchScanRequest(BaseModel):
    hashes: List[str]
    device_token: Optional[str] = None

class WatchScanResponse(BaseModel):
    matches: List[MatchItem]
    count: int
    provider_used: str
    scanned_at_utc: str
