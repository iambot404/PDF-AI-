import math
import re
from collections import Counter
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


class BM25Index:
    """
    Lightweight, zero-dependency BM25 Okapi lexical ranking engine.
    Computes exact keyword relevance across document chunks without external dependencies.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0
        self.N: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())

    def index(self, texts: List[str]):
        self.doc_tokens = [self.tokenize(t) for t in texts]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.N = len(self.doc_tokens)
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N > 0 else 0.0
        self.df = Counter()
        for doc in self.doc_tokens:
            self.df.update(set(doc))

    def score(self, query: str) -> np.ndarray:
        q_tokens = self.tokenize(query)
        if not q_tokens or self.N == 0:
            return np.zeros(self.N, dtype=np.float32)

        scores = np.zeros(self.N, dtype=np.float32)
        for i, doc in enumerate(self.doc_tokens):
            doc_len = self.doc_lens[i]
            doc_counts = Counter(doc)
            s = 0.0
            for q in q_tokens:
                if q not in self.df:
                    continue
                n_q = self.df[q]
                idf = math.log((self.N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                f = doc_counts[q]
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl if self.avgdl > 0 else 1.0))
                s += idf * (f * (self.k1 + 1.0)) / (denom if denom > 0 else 1.0)
            scores[i] = s
        return scores


class FAISSRetriever:
    """
    Manages vector indexes and metadata linking for semantic similarity retrieval.
    Features:
    - Native FAISS IndexFlatIP indexing when available, with pure-NumPy fallback on serverless.
    - Hybrid Retrieval fusing Dense Semantic Embeddings + BM25 Lexical Ranking.
    - Zero-hallucination chunk isolation per doc_id.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.use_faiss = HAS_FAISS
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.doc_index_map: Dict[str, List[int]] = {}
        self.bm25_indices: Dict[str, BM25Index] = {}

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self._vectors = np.empty((0, dimension), dtype=np.float32)

    def index_document(self, doc_id: str, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> int:
        """
        Adds document embeddings and their metadata to the index, and builds a BM25 lexical index.
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
            self.bm25_indices = {}

        start_idx = len(self.chunks_metadata)

        if self.use_faiss:
            self.index.add(embeddings)
        else:
            self._vectors = np.vstack([self._vectors, embeddings])

        if doc_id not in self.doc_index_map:
            self.doc_index_map[doc_id] = []

        chunk_texts = []
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            chunk_copy = dict(chunk)
            chunk_copy["vector_id"] = idx
            self.chunks_metadata.append(chunk_copy)
            self.doc_index_map[doc_id].append(idx)
            chunk_texts.append(chunk.get("text", ""))

        # Build BM25 index for the document
        bm25 = BM25Index()
        bm25.index(chunk_texts)
        self.bm25_indices[doc_id] = bm25

        return len(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        doc_id: Optional[str] = None,
        top_k: int = 6,
        score_threshold: float = 0.0,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for the top_k most similar chunks using Hybrid Search (Dense Vector + BM25).
        If query_text is omitted, performs standard dense vector similarity.
        """
        total = self.total_chunks
        if total == 0 or len(self.chunks_metadata) == 0:
            return []

        # Determine target candidates
        if doc_id and doc_id in self.doc_index_map:
            cand_indices = self.doc_index_map[doc_id]
        else:
            cand_indices = list(range(len(self.chunks_metadata)))

        if not cand_indices:
            return []

        # 1. Compute Dense Vector Similarities
        flat_q = query_vector.flatten()
        dense_scores = np.zeros(len(cand_indices), dtype=np.float32)
        
        if self.use_faiss:
            for local_i, global_i in enumerate(cand_indices):
                try:
                    vec = self.index.reconstruct(int(global_i))
                    dense_scores[local_i] = float(np.dot(vec, flat_q))
                except Exception:
                    dense_scores[local_i] = 0.0
        else:
            if hasattr(self, "_vectors") and self._vectors.shape[0] > 0:
                doc_vecs = self._vectors[cand_indices]
                dense_scores = np.dot(doc_vecs, flat_q).astype(np.float32)

        # 2. Compute BM25 Lexical Scores if query_text is provided
        use_hybrid = False
        bm25_scores = np.zeros(len(cand_indices), dtype=np.float32)
        if query_text and query_text.strip():
            if doc_id and doc_id in self.bm25_indices:
                bm25_scores = self.bm25_indices[doc_id].score(query_text)
                use_hybrid = True
            else:
                # Global query across all chunks
                global_bm25 = BM25Index()
                global_bm25.index([self.chunks_metadata[i].get("text", "") for i in cand_indices])
                bm25_scores = global_bm25.score(query_text)
                use_hybrid = True

        # 3. Fuse Scores
        if use_hybrid:
            max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) > 0 else 0.0
            if max_bm25 > 0.0:
                norm_bm25 = bm25_scores / max_bm25
                norm_dense = np.clip(dense_scores, 0.0, 1.0)
                # Weighted combination: 55% BM25 keyword matching + 45% Dense semantics
                fused_scores = 0.55 * norm_bm25 + 0.45 * norm_dense
            else:
                fused_scores = dense_scores
        else:
            fused_scores = dense_scores

        # 4. Rank and assemble top-k deduplicated chunks
        ranked_local_indices = np.argsort(fused_scores)[::-1]
        
        results: List[Dict[str, Any]] = []
        seen_snippets = set()

        for local_idx in ranked_local_indices:
            global_idx = cand_indices[local_idx]
            chunk = self.chunks_metadata[global_idx]
            score = float(fused_scores[local_idx])

            if score < score_threshold:
                continue

            snippet_key = chunk.get("text", "")[:120].strip().lower()
            if snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)

            result = dict(chunk)
            result["score"] = round(score, 4)
            result["dense_score"] = round(float(dense_scores[local_idx]), 4)
            if use_hybrid:
                result["bm25_score"] = round(float(bm25_scores[local_idx]), 4)
            
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
            # Rebuild FAISS index from remaining vectors if needed
            new_index = faiss.IndexFlatIP(self.dimension)
            for i, c in enumerate(self.chunks_metadata):
                if i not in indices_to_remove:
                    try:
                        v = self.index.reconstruct(i)
                        new_index.add(np.array([v], dtype=np.float32))
                    except Exception:
                        pass
            self.index = new_index

        self.chunks_metadata = remaining_chunks
        del self.doc_index_map[doc_id]
        if doc_id in self.bm25_indices:
            del self.bm25_indices[doc_id]
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
        self.bm25_indices.clear()

    @property
    def total_chunks(self) -> int:
        if self.use_faiss:
            return self.index.ntotal
        return self._vectors.shape[0] if hasattr(self, "_vectors") else 0
