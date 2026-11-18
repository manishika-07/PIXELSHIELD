from typing import Optional, Any
from pydantic import BaseModel, Field

class APIErrorResponse(BaseModel):
    code: str
    message: str
    hint: Optional[str] = None
    details: Optional[Any] = None

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
