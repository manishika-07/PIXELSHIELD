import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.app.config import settings
from backend.app.database import engine, Base
from backend.app.core.logging import logger
from backend.app.core.exceptions import PixelShieldException, pixelshield_exception_handler, generic_exception_handler
from backend.app.services.shield_engine import check_models_ready, load_ml_models
from backend.app.api.v1.shield import router as shield_router

# Rate limiter based on remote IP
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing Photo Shield Backend...")
    # Create DB tables
    Base.metadata.create_all(bind=engine)
    # Ensure evidence vault directory
    os.makedirs(settings.EVIDENCE_VAULT_DIR, exist_ok=True)
    # Preload models in background
    logger.info("Checking ML models readiness...")
    load_ml_models()
    yield
    # Shutdown actions
    logger.info("Photo Shield backend shutting down.")

app = FastAPI(
    title="Photo Shield API",
    version="1.0.0",
    description="Proactive anti-deepfake protection, misuse watching, and incident response platform for Indian users.",
    lifespan=lifespan
)

# Attach SlowAPI State & Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(PixelShieldException, pixelshield_exception_handler)
if not settings.DEBUG:
    app.add_exception_handler(Exception, generic_exception_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID & Logging Middleware
@app.middleware("http")
async def request_id_and_logging_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id
    start_time = time.time()
    
    response = await call_next(request)
    
    elapsed = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    
    # Exclude health probe logs to keep output clean
    if request.url.path not in ["/health", "/ready"]:
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed}ms)")
        
    return response

# System Probes
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "service": "photo-shield-backend",
        "version": "1.0.0",
        "server_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@app.get("/ready", tags=["System"])
def readiness_check():
    models_status = check_models_ready()
    return {
        "status": "ready",
        "models": models_status
    }

# Mount API Routers
app.include_router(shield_router, prefix="/api/v1")

# Serve Frontend if directory exists
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.exists(frontend_dir):
    @app.get("/", tags=["Frontend"])
    def serve_frontend_root():
        pixi_path = os.path.join(frontend_dir, "pixi.html")
        if os.path.exists(pixi_path):
            return FileResponse(pixi_path)
        return {"message": "Photo Shield API Active. Frontend not found at root."}
    
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
