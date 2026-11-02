import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import app

# WSGI middleware to normalize Vercel serverless request path
class VercelPathMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path in ("/api/index", "/api/index.py", "/api", "/api/"):
            environ["PATH_INFO"] = "/"
        elif path.startswith("/api/index/"):
            environ["PATH_INFO"] = path[10:]
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
