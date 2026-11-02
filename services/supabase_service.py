import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class SupabaseService:
    """
    Manages optional persistence to Supabase for document metadata and query history.
    Gracefully falls back to local in-memory storage if credentials are not configured.
    """

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_ANON_KEY")
        self.is_configured = bool(self.url and self.key)
        self.client = None

        if self.is_configured:
            try:
                from supabase import create_client
                self.client = create_client(self.url, self.key)
                logger.info("Supabase client initialized successfully.")
            except ImportError:
                logger.warning("supabase-py not installed; continuing with local storage fallback.")
                self.is_configured = False
            except Exception as e:
                logger.warning(f"Failed to connect to Supabase: {e}; falling back to local storage.")
                self.is_configured = False

        # In-memory document session store fallback
        self._local_docs: Dict[str, Dict[str, Any]] = {}
        self._local_history: List[Dict[str, Any]] = []

    def save_document_meta(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Saves document metadata."""
        self._local_docs[doc_id] = metadata
        if self.is_configured and self.client:
            try:
                self.client.table("documents").upsert({
                    "id": doc_id,
                    "filename": metadata.get("filename"),
                    "total_pages": metadata.get("total_pages"),
                    "total_chunks": metadata.get("total_chunks"),
                    "created_at": "now()"
                }).execute()
                return True
            except Exception as e:
                logger.warning(f"Supabase upsert failed: {e}")
        return True

    def log_query_event(self, event_data: Dict[str, Any]) -> bool:
        """Logs question and answer fact-check events."""
        self._local_history.append(event_data)
        if self.is_configured and self.client:
            try:
                self.client.table("qa_logs").insert(event_data).execute()
                return True
            except Exception as e:
                logger.debug(f"Supabase log failed: {e}")
        return True

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._local_docs.get(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self._local_docs.values())

    def delete_document(self, doc_id: str) -> bool:
        if doc_id in self._local_docs:
            del self._local_docs[doc_id]
        if self.is_configured and self.client:
            try:
                self.client.table("documents").delete().eq("id", doc_id).execute()
            except Exception:
                pass
        return True
