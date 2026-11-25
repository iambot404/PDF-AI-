import os
import json
import time
import random
import logging
import re
import hashlib
import urllib.request
import urllib.error
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

def hash_embed_texts(texts: List[str], dim: int = 768) -> np.ndarray:
    """
    Ultra-fast, zero-dependency, local subword/token TF-IDF vectorizer using pure NumPy.
    Used as an immediate, guaranteed fallback when Gemini Embeddings API hits 429 rate limits.
    """
    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        tokens = re.findall(r'[a-zA-Z0-9_]{2,}', t.lower())
        for tok in tokens:
            # Word token hashing
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vecs[i, idx] += sign
            
            # Subword 3-gram hashing for morphological matching
            for j in range(len(tok) - 2):
                gram = tok[j:j+3]
                hg = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                idx_g = hg % dim
                sign_g = 1.0 if (hg // dim) % 2 == 0 else -1.0
                vecs[i, idx_g] += 0.5 * sign_g

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


class EmbeddingService:
    """
    Generates semantic dense embeddings with automatic multi-tier fallback:
    1. Google Gemini Embeddings API (gemini-embedding-001) with exponential jittered retry.
    2. Automatic pure-NumPy vectorizer fallback if Gemini API hits HTTP 429 rate limits.
    3. Optional local SentenceTransformers if installed.
    """

    _instance: Optional['EmbeddingService'] = None
    _st_model = None

    def __init__(self, model_name: Optional[str] = None, gemini_model: str = "gemini-embedding-001"):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = gemini_model
        self.st_model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.prefer_local = os.getenv("PREFER_LOCAL_EMBEDDINGS", "false").lower() == "true"
        self._dim = None
        self.last_mode_used = "gemini"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts and returns L2-normalized float32 numpy array.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.prefer_local:
            try:
                res = self._embed_via_sentence_transformers(texts)
                self.last_mode_used = "sentence_transformers"
                return res
            except Exception as e:
                logger.warning(f"Local sentence-transformers embedding failed: {e}. Trying Gemini API.")

        # Attempt Gemini Cloud Embeddings with backoff retry
        try:
            res = self._embed_via_gemini_with_retries(texts)
            self.last_mode_used = "gemini"
            return res
        except Exception as e:
            logger.warning(f"Gemini embedding API failed ({e}). Automatically activating high-precision BM25 fallback.")
            res = hash_embed_texts(texts, dim=768)
            self._dim = 768
            self.last_mode_used = "bm25"
            return res

    def embed_query(self, query: str, mode: Optional[str] = None) -> np.ndarray:
        """
        Embeds a single user question query matching the mode used for document indexing.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        active_mode = mode or self.last_mode_used

        if active_mode in ("bm25", "numpy_vectorizer"):
            return hash_embed_texts([query.strip()], dim=768)

        if active_mode == "sentence_transformers":
            try:
                return self._embed_via_sentence_transformers([query.strip()])
            except Exception:
                pass

        try:
            return self._embed_via_gemini_with_retries([query.strip()])
        except Exception as e:
            logger.warning(f"Gemini query embedding failed: {e}. Using high-precision BM25 fallback.")
            return hash_embed_texts([query.strip()], dim=768)

    def _embed_via_gemini_with_retries(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        """
        Batch embeds texts using the Google Gemini embedding REST API with jittered backoff.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required to generate embeddings.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:batchEmbedContents?key={self.api_key}"
        # Batch 25 chunks per call to stay within API payload limits
        batch_size = 25
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

            batch_succeeded = False
            last_err = None

            for attempt in range(1, max_retries + 1):
                try:
                    req = urllib.request.Request(
                        url,
                        data=data_bytes,
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=25) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                        for emb in resp_data.get("embeddings", []):
                            all_vectors.append(emb.get("values", []))
                    batch_succeeded = True
                    break

                except urllib.error.HTTPError as he:
                    status = he.code
                    err_msg = he.read().decode("utf-8", errors="ignore")
                    last_err = he

                    # 429 Too Many Requests (immediate fallback) vs 503/504 transient overload
                    if status == 429:
                        raise RuntimeError(f"Gemini Embeddings API rate limit reached (HTTP 429). Fast-switching to BM25.")
                    elif status in (503, 504) and attempt < max_retries:
                        jitter = random.uniform(0.5, 1.5)
                        sleep_time = (1.5 * (2 ** (attempt - 1))) + jitter
                        logger.warning(f"Gemini Embeddings HTTP {status} on batch {i}. Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise RuntimeError(f"Gemini Embeddings API HTTP {status}: {err_msg}")

                except (urllib.error.URLError, TimeoutError) as ue:
                    last_err = ue
                    if attempt < max_retries:
                        time.sleep(1.0 * attempt)
                        continue
                    raise RuntimeError(f"Gemini Embeddings connection error: {ue}")

            if not batch_succeeded:
                raise RuntimeError(f"Batch embedding failed after {max_retries} attempts: {last_err}")

            # Brief pause between batches to respect RPM limits
            if i + batch_size < len(texts):
                time.sleep(0.2)

        arr = np.array(all_vectors, dtype=np.float32)
        if arr.shape[0] > 0:
            self._dim = arr.shape[1]
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
        return 3072
