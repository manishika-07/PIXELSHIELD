from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

class EvidenceCreateResponse(BaseModel):
    evidence_id: str
    target_url: str
    screenshot_hash: str
    seal: str
    server_timestamp: str
    tamper_proof: bool = True
    chain_of_custody_note: str

class EvidenceItem(BaseModel):
    id: str
    target_url: str
    notes: Optional[str] = None
    screenshot_hash: str
    seal: str
    server_timestamp: str

class EvidenceListResponse(BaseModel):
    items: List[EvidenceItem]
    count: int

class EvidenceVerifyResponse(BaseModel):
    evidence_id: str
    valid: bool
    tampered: bool
    calculated_seal: str
    stored_seal: str
    verified_at_utc: str
    status_summary: str
