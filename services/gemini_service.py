import os
import time
import random
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

class GeminiServiceError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, is_transient: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.is_transient = is_transient

class GeminiService:
    """
    Manages communication with Google Gemini API with:
    - Jittered exponential backoff
    - Automatic model fallback (e.g. flash-lite -> flash -> pro)
    - Response caching to avoid redundant API requests
    - Grounded prompt templates restricting answers to retrieved document context
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        primary_model: str = "gemini-flash-lite-latest",
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No Gemini/Google API key found. Please set GEMINI_API_KEY in .env.")

        self.primary_model = primary_model
        self.fallback_models = fallback_models or [
            "gemini-flash-lite-latest",
            "gemini-2.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash"
        ]
        # Ensure unique order
        all_models = [self.primary_model] + [m for m in self.fallback_models if m != self.primary_model]
        self.models_to_try = all_models
        
        self.max_retries = max_retries
        self.base_delay = base_delay

        # In-memory LRU-style cache: hash(doc_id + query) -> response_dict
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_cache_size = 200

    def _get_cache_key(self, doc_id: str, question: str, chunks: List[Dict[str, Any]]) -> str:
        # Build deterministic fingerprint from doc_id, normalized question, and chunk IDs
        norm_q = question.strip().lower()
        chunk_ids = ",".join(sorted([c.get("chunk_id", "") for c in chunks]))
        raw = f"{doc_id}:{norm_q}:{chunk_ids}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate_grounded_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        doc_id: str = "doc",
        filename: str = "document.pdf"
    ) -> Dict[str, Any]:
        """
        Generates a context-grounded answer using the retrieved passages.
        Returns dictionary containing answer text, citations, used model, latency, and cache status.
        """
        # Check cache
        cache_key = self._get_cache_key(doc_id, question, retrieved_chunks)
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cached"] = True
            return cached

        # If no chunks were retrieved or empty context
        if not retrieved_chunks:
            return {
                "answer": f"I could not find any relevant sections in '{filename}' to answer your question.",
                "citations": [],
                "model_used": "none",
                "cached": False,
                "latency_ms": 0,
                "is_grounded": False,
                "warning": "No relevant context passages were retrieved."
            }

        # Build context blocks with clear citation markers
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            page_num = chunk.get("page_number", 1)
            chunk_id = chunk.get("chunk_id", f"c{i}")
            text = chunk.get("text", "").strip()
            context_blocks.append(f"--- [PASSAGE #{i+1} | Page {page_num} | ID: {chunk_id}] ---\n{text}")

        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "You are a strict, precise document QA assistant. "
            "You must answer the user's question using ONLY the provided document excerpts. "
            "STRICT FACT-CHECKING RULES:\n"
            "1. Ground every statement directly in the provided passages.\n"
            "2. Whenever you state a fact, cite the specific page number in brackets, e.g. [Page 4].\n"
            "3. If the answer cannot be determined or is missing from the excerpts, state clearly: "
            "'Based on the provided document excerpts, there is insufficient information to answer this question.'\n"
            "4. NEVER invent facts, extrapolate unsupported details, or hallucinate citations.\n"
            "5. Keep the answer clear, structured, and factual."
        )

        prompt = f"{system_instruction}\n\nDOCUMENT PASSAGES:\n{context_str}\n\nUSER QUESTION: {question}\n\nGROUNDED ANSWER:"

        # Attempt call with model fallback and jittered exponential retry
        last_error = None
        for model in self.models_to_try:
            try:
                start_time = time.time()
                response_text = self._call_gemini_with_retries(model, prompt)
                latency_ms = int((time.time() - start_time) * 1000)

                # Collect citations from retrieved chunks
                citations = []
                for c in retrieved_chunks:
                    citations.append({
                        "chunk_id": c.get("chunk_id"),
                        "page_number": c.get("page_number"),
                        "filename": c.get("filename", filename),
                        "snippet": c.get("text", "")[:280] + ("..." if len(c.get("text", "")) > 280 else ""),
                        "score": round(c.get("score", 0.0), 3)
                    })

                result = {
                    "answer": response_text.strip(),
                    "citations": citations,
                    "model_used": model,
                    "cached": False,
                    "latency_ms": latency_ms,
                    "is_grounded": True
                }

                # Save to cache
                if len(self._cache) >= self._max_cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = result

                return result

            except GeminiServiceError as ge:
                logger.warning(f"Model {model} failed: {ge}. Trying fallback...")
                last_error = ge
                continue
            except Exception as e:
                logger.error(f"Unexpected error with model {model}: {e}")
                last_error = e
                continue

        # If all models failed, raise clean error
        raise GeminiServiceError(f"All Gemini models exhausted. Last error: {last_error}")

    def _call_gemini_with_retries(self, model: str, prompt: str) -> str:
        """
        Calls Gemini generateContent endpoint with jittered exponential backoff.
        """
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,  # Low temperature for factual precision
                "maxOutputTokens": 1024,
                "topP": 0.8
            }
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    candidates = resp_data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
                    return "No answer generated by the model."

            except urllib.error.HTTPError as he:
                status = he.code
                err_body = he.read().decode("utf-8", errors="ignore")
                
                # Check for rate limit (429) or server overload (503, 504)
                if status in (429, 503, 504):
                    if attempt < self.max_retries:
                        jitter = random.uniform(0.5, 1.5)
                        sleep_time = (self.base_delay * (2 ** (attempt - 1))) + jitter
                        logger.warning(f"HTTP {status} on attempt {attempt}/{self.max_retries}. Backing off {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise GeminiServiceError(f"Rate limit or service overload (HTTP {status}): {err_body}", status_code=status, is_transient=True)
                elif status in (400, 401, 403, 404):
                    # Non-transient errors: do not retry
                    raise GeminiServiceError(f"Gemini API Client Error (HTTP {status}): {err_body}", status_code=status, is_transient=False)
                else:
                    if attempt < self.max_retries:
                        time.sleep(1.0)
                        continue
                    raise GeminiServiceError(f"HTTP Error {status}: {err_body}", status_code=status)

            except (urllib.error.URLError, TimeoutError) as ue:
                if attempt < self.max_retries:
                    time.sleep(self.base_delay * attempt)
                    continue
                raise GeminiServiceError(f"Network timeout/error connecting to Gemini: {ue}", is_transient=True)

        raise GeminiServiceError("Exceeded maximum retry attempts.")
