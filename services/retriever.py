import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.info("FAISS not installed; using optimized pure-NumPy vector search.")

class FAISSRetriever:
    """
    Manages vector indexes and metadata linking for semantic similarity retrieval.
    Uses FAISS IndexFlatIP when available, and automatically falls back to an
    ultra-fast NumPy dot-product engine on serverless platforms (e.g. Vercel)
    to keep function bundle sizes tiny (< 50MB).
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.use_faiss = HAS_FAISS
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.doc_index_map: Dict[str, List[int]] = {}

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self._vectors = np.empty((0, dimension), dtype=np.float32)

    def index_document(self, doc_id: str, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> int:
        """
        Adds document embeddings and their metadata to the index.
        Returns the number of chunks indexed.
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return 0

        # Adjust dimension if needed
        emb_dim = embeddings.shape[1]
        if emb_dim != self.dimension:
            self.dimension = emb_dim
            if self.use_faiss:
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self._vectors = np.empty((0, self.dimension), dtype=np.float32)
            self.chunks_metadata = []
            self.doc_index_map = {}

        start_idx = len(self.chunks_metadata)

        if self.use_faiss:
            self.index.add(embeddings)
        else:
            self._vectors = np.vstack([self._vectors, embeddings])

        if doc_id not in self.doc_index_map:
            self.doc_index_map[doc_id] = []

        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            chunk_copy = dict(chunk)
            chunk_copy["vector_id"] = idx
            self.chunks_metadata.append(chunk_copy)
            self.doc_index_map[doc_id].append(idx)

        return len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        doc_id: Optional[str] = None,
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Searches for the top_k most similar chunks using cosine similarity.
        Filters by doc_id if provided.
        """
        total = self.total_chunks
        if total == 0 or len(self.chunks_metadata) == 0:
            return []

        fetch_k = min(total, top_k * 4 if doc_id else top_k)

        if self.use_faiss:
            raw_scores, raw_indices = self.index.search(query_vector, fetch_k)
            scores = raw_scores[0]
            indices = raw_indices[0]
        else:
            # Pure NumPy inner product search on L2-normalized vectors
            sims = np.dot(self._vectors, query_vector.T).flatten()
            top_idx = np.argsort(sims)[::-1][:fetch_k]
            scores = sims[top_idx]
            indices = top_idx

        results: List[Dict[str, Any]] = []
        seen_texts = set()

        for score, idx in zip(scores, indices):
            if idx == -1 or idx >= len(self.chunks_metadata):
                continue

            chunk = self.chunks_metadata[idx]
            if doc_id and chunk.get("doc_id") != doc_id:
                continue

            text_snippet = chunk["text"][:100].lower()
            if text_snippet in seen_texts:
                continue
            seen_texts.add(text_snippet)

            result = dict(chunk)
            result["score"] = float(score)

            if result["score"] >= score_threshold:
                results.append(result)

            if len(results) >= top_k:
                break

        return results

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document from the index."""
        if doc_id not in self.doc_index_map:
            return False

        indices_to_remove = set(self.doc_index_map[doc_id])
        remaining_chunks = [c for i, c in enumerate(self.chunks_metadata) if i not in indices_to_remove]

        if not self.use_faiss:
            keep_mask = [i not in indices_to_remove for i in range(len(self.chunks_metadata))]
            self._vectors = self._vectors[keep_mask] if len(keep_mask) > 0 else np.empty((0, self.dimension), dtype=np.float32)
        else:
            # Reset FAISS
            self.index = faiss.IndexFlatIP(self.dimension)

        self.chunks_metadata = remaining_chunks
        del self.doc_index_map[doc_id]
        return True

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        return [c for c in self.chunks_metadata if c.get("doc_id") == doc_id]

    def clear_all(self):
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.chunks_metadata.clear()
        self.doc_index_map.clear()

    @property
    def total_chunks(self) -> int:
        if self.use_faiss:
            return self.index.ntotal
        return self._vectors.shape[0] if hasattr(self, "_vectors") else 0
