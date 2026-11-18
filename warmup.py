import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.core.logging import logger
from backend.app.services.shield_engine import load_ml_models, check_models_ready

def warmup():
    logger.info("Starting Photo Shield model weight warmup...")
    ok = load_ml_models(timeout_sec=30.0)
    status = check_models_ready()
    logger.info(f"Warmup status: {status}")
    if ok:
        print("Model warmup completed successfully.")
    else:
        print("Warmup completed with offline fallback architecture ready.")

if __name__ == "__main__":
    warmup()
