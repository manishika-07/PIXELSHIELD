"""
Photo Shield backend launcher.
Run from project root: python run_server.py
"""
import os
import sys

# Set thread counts BEFORE any torch/numpy import
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Ensure repo root is on sys.path so `backend.app...` imports resolve
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)  # ensure cwd = project root regardless of how script is called

# Import the app BEFORE passing to uvicorn so our sys.path is used
from backend.app.main import app  # noqa: E402

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,                # pass the app object directly (no string import)
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,       # reload=True causes double-import of torch → OOM
    )
