import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import app

# Vercel serverless function entrypoint
# The 'app' object is exposed as WSGI handler
