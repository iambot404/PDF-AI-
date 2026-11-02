import os
import json
import logging
import urllib.request
import urllib.error
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Generates semantic dense embeddings.
    By default uses Google Gemini Embeddings API (gemini-embedding-001) for
    ultra-fast, lightweight serverless deployment (no multi-gigabyte PyTorch bundle).
    Gracefully falls back to local SentenceTransformers if installed and configured.
    """

    _instance: Optional['EmbeddingService'] = None
    _st_model = None

    def __init__(self, model_name: Optional[str] = None, gemini_model: str = "gemini-embedding-001"):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = gemini_model
        self.st_model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.prefer_local = os.getenv("PREFER_LOCAL_EMBEDDINGS", "false").lower() == "true"
        self._dim = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts and returns L2-normalized float32 numpy array.
        L2 normalization allows inner product (dot product) to compute exact Cosine Similarity.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.prefer_local:
            try:
                return self._embed_via_sentence_transformers(texts)
            except Exception as e:
                logger.warning(f"Local sentence-transformers embedding failed: {e}. Falling back to Gemini API.")

        return self._embed_via_gemini(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single user question query and returns shape (1, dim) float32 array.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        if self.prefer_local:
            try:
                return self._embed_via_sentence_transformers([query.strip()])
            except Exception as e:
                logger.warning(f"Local sentence-transformers query embedding failed: {e}. Falling back to Gemini API.")

        return self._embed_via_gemini([query.strip()])

    def _embed_via_gemini(self, texts: List[str]) -> np.ndarray:
        """
        Batch embeds texts using the Google Gemini embedding REST API.
        Lightweight, lightning-fast (< 500ms), and adds 0 MB to serverless bundle.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required to generate embeddings.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:batchEmbedContents?key={self.api_key}"
        batch_size = 35
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            chunk_batch = texts[i:i + batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.gemini_model}",
                        "content": {"parts": [{"text": t}]}
                    }
                    for t in chunk_batch
                ]
            }
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                for emb in resp_data.get("embeddings", []):
                    all_vectors.append(emb.get("values", []))

        arr = np.array(all_vectors, dtype=np.float32)
        if arr.shape[0] > 0:
            self._dim = arr.shape[1]
            # L2 normalize so cosine similarity = dot product
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms

        return arr

    def _embed_via_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        if EmbeddingService._st_model is None:
            from sentence_transformers import SentenceTransformer
            EmbeddingService._st_model = SentenceTransformer(self.st_model_name)

        embeddings = EmbeddingService._st_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self._dim = embeddings.shape[1]
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        if self._dim is not None:
            return self._dim
        if self.prefer_local and EmbeddingService._st_model is not None:
            if hasattr(EmbeddingService._st_model, "get_embedding_dimension"):
                return EmbeddingService._st_model.get_embedding_dimension()
        return 3072  # gemini-embedding-001 default dimension
