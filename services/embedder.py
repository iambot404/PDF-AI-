import numpy as np
from typing import List, Optional
import os

class EmbeddingService:
    """
    Generates dense vector embeddings using Sentence Transformers.
    Caches model in memory and normalizes vectors for fast cosine similarity.
    """

    _instance: Optional['EmbeddingService'] = None
    _model = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        if EmbeddingService._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Load SentenceTransformer model
                EmbeddingService._model = SentenceTransformer(self.model_name)
            except Exception as e:
                # If local sentence-transformers cannot be loaded, prepare fallback
                raise RuntimeError(f"Could not load SentenceTransformer model '{self.model_name}': {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts and returns L2-normalized float32 numpy array.
        L2 normalization allows FAISS IndexFlatIP (Inner Product) to compute exact Cosine Similarity.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        self._ensure_model_loaded()
        embeddings = EmbeddingService._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single user question query and returns shape (1, dim) float32 array.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        self._ensure_model_loaded()
        embedding = EmbeddingService._model.encode(
            [query.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        self._ensure_model_loaded()
        if hasattr(EmbeddingService._model, "get_embedding_dimension"):
            return EmbeddingService._model.get_embedding_dimension()
        return EmbeddingService._model.get_sentence_embedding_dimension()
