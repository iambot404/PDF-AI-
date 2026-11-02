import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "mini-rag-secret-key-2026")
    
    # Uploads
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_FILE_SIZE_MB", 25)) * 1024 * 1024  # 25 MB max
    ALLOWED_EXTENSIONS = {"pdf"}
    MAX_PAGES = int(os.getenv("MAX_PAGES", 150))
    
    # Gemini AI configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    PRIMARY_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    
    # Fallback models in priority order
    _models_env = os.getenv("GEMINI_MODELS", "")
    FALLBACK_MODELS = [m.strip() for m in _models_env.split(",") if m.strip()] if _models_env else [
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash"
    ]
    
    # Chunking configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))  # in characters/words
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    
    # Retrieval configuration
    TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", 4))
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Rate Limiting & Reliability
    MAX_RPM = int(os.getenv("MAX_RPM", 30))  # Requests per minute
    RETRY_MAX_ATTEMPTS = 3
    RETRY_BASE_DELAY = 1.0  # seconds
    
    # Fact Checking Thresholds
    FACT_CHECK_STRICT_MODE = os.getenv("FACT_CHECK_STRICT_MODE", "true").lower() == "true"
    MIN_GROUNDEDNESS_SCORE = float(os.getenv("MIN_GROUNDEDNESS_SCORE", "0.65"))

    # Supabase optional
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
