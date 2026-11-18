import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "*"
    ]
    
    # Security & Storage
    SECRET_KEY: str = "pixelshield_super_secret_hmac_key_for_tamper_proof_evidence_2026"
    DATABASE_URL: str = "sqlite:///./pixelshield.db"
    EVIDENCE_VAULT_DIR: str = "./data/vault"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024 # 10 MB
    MAX_IMAGE_PIXELS: int = 16 * 1024 * 1024 # Decompression bomb guard
    RATE_LIMIT_PER_MINUTE: int = 60
    JOB_TTL_SECONDS: int = 600 # 10 minutes
    
    # ML Engine
    DEVICE: str = "auto"
    RESIZE_MAX_LONG_SIDE: int = 1024
    DEFAULT_PGD_STEPS: int = 60
    DEFAULT_EPSILON: float = 8.0 / 255.0
    FACE_MODEL: str = "vggface2"
    VAE_MODEL: str = "stabilityai/sd-vae-ft-mse"
    
    # Search Providers
    DEFAULT_SEARCH_PROVIDER: str = "mock"
    GOOGLE_VISION_API_KEY: str = ""
    TINEYE_API_KEY: str = ""
    WEBHOOK_ALERT_URL: str = ""

settings = Settings()
