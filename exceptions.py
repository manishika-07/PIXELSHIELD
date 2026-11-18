from typing import Optional, Any
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class PixelShieldException(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        hint: Optional[str] = None,
        details: Optional[Any] = None
    ):
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details

async def pixelshield_exception_handler(request: Request, exc: PixelShieldException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "hint": exc.hint,
            "details": exc.details
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred while processing the request.",
            "hint": "Please retry with a valid photo or switch to on-device mode if offline."
        }
    )
