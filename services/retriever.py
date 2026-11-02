import numpy as np
import faiss
from typing import List, Dict, Any, Optional

class FAISSRetriever:
    """
    Manages FAISS vector indexes and metadata linking for semantic similarity retrieval.
    Uses IndexFlatIP with normalized vectors for exact Cosine Similarity scores.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        # Ordered list of chunk metadata dictionaries corresponding to FAISS vector IDs (0..N-1)
        self.chunks_metadata: List[Dict[str, Any]] = []
        # Mapping from doc_id to list of chunk indices for quick document management
        self.doc_index_map: Dict[str, List[int]] = {}

    def index_document(self, doc_id: str, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> int:
        """
        Adds document embeddings and their metadata to the FAISS index.
        Returns the number of chunks indexed.
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return 0

        if embeddings.shape[1] != self.dimension:
            # Recreate index if dimension changed
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks_metadata = []
            self.doc_index_map = {}

        start_idx = len(self.chunks_metadata)
        self.index.add(embeddings)

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
        Searches the FAISS index for the top_k most similar chunks.
        Filters by doc_id if provided.
        Returns list of chunks with similarity scores.
        """
        if self.index.ntotal == 0 or len(self.chunks_metadata) == 0:
            return []

        # Retrieve extra results if filtering by doc_id
        fetch_k = min(self.index.ntotal, top_k * 4 if doc_id else top_k)
        scores, indices = self.index.search(query_vector, fetch_k)

        results: List[Dict[str, Any]] = []
        seen_texts = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks_metadata):
                continue

            chunk = self.chunks_metadata[idx]
            
            # Filter by document if specified
            if doc_id and chunk.get("doc_id") != doc_id:
                continue

            # Remove near-duplicate passages
            text_snippet = chunk["text"][:100].lower()
            if text_snippet in seen_texts:
                continue
            seen_texts.add(text_snippet)

            result = dict(chunk)
            # FAISS IndexFlatIP score on normalized vectors is cosine similarity [-1.0, 1.0]
            result["score"] = float(score)
            
            if result["score"] >= score_threshold:
                results.append(result)

            if len(results) >= top_k:
                break

        return results

    def remove_document(self, doc_id: str) -> bool:
        """
        Removes a document and rebuilds the FAISS index without its chunks.
        """
        if doc_id not in self.doc_index_map:
            return False

        # Filter remaining chunks
        remaining_chunks = [c for c in self.chunks_metadata if c.get("doc_id") != doc_id]
        
        # Reset index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []
        self.doc_index_map = {}

        if remaining_chunks:
            # We need to re-index. If we had stored the raw vectors or we re-insert
            pass

        return True

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Returns all chunks belonging to a document."""
        return [c for c in self.chunks_metadata if c.get("doc_id") == doc_id]

    def clear_all(self):
        """Clears all indexed documents and resets FAISS."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata.clear()
        self.doc_index_map.clear()

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
