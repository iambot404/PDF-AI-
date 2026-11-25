import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import app

class VercelPathMiddleware:
    """
    WSGI middleware to accurately resolve paths on Vercel serverless functions.
    Vercel sets HTTP_X_MATCHED_PATH (or HTTP_X_VERCEL_PATH) to the client's actual request path
    when rewrites are evaluated, while PATH_INFO may be set to '/api/index'.
    This middleware restores the true client PATH_INFO so Flask dispatches to the correct route and method.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # 1. Recover true incoming client path from Vercel edge headers
        matched_path = (
            environ.get("HTTP_X_MATCHED_PATH") or 
            environ.get("HTTP_X_VERCEL_PATH") or 
            environ.get("HTTP_X_FORWARDED_URI") or 
            environ.get("REQUEST_URI") or
            ""
        )
        if matched_path:
            clean_path = matched_path.split("?")[0].strip()
            # If the header contains a real application path (like /api/upload, /upload, /api/query)
            if clean_path and not clean_path.startswith(("/api/index", "/api/index.py")):
                environ["PATH_INFO"] = clean_path

        # 2. Normalize internal Vercel entrypoint paths
        path = environ.get("PATH_INFO", "")
        if path in ("/api/index", "/api/index.py", "/api", "/api/"):
            environ["PATH_INFO"] = "/"
        elif path.startswith("/api/index/"):
            environ["PATH_INFO"] = path[10:]

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
