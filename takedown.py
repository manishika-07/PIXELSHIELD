from typing import Optional, List
from pydantic import BaseModel, Field

class TakedownGenerateRequest(BaseModel):
    platform: str
    urls: List[str]
    language: str = "en" # en, hi, ta, bn, mr, te
    incident_date: Optional[str] = None
    victim_name: Optional[str] = "Concerned Citizen"
    evidence_id: Optional[str] = None

class LegalCitation(BaseModel):
    code: str
    title: str
    summary: str

class TakedownGenerateResponse(BaseModel):
    platform: str
    reporting_url: str
    grievance_email: Optional[str] = None
    statutory_sla_hours: int
    complaint_text: str
    legal_citations: List[LegalCitation]
    step_by_step_instructions: List[str]
    disclaimer: str
