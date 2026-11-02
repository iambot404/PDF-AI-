# DocuMind AI — Mini RAG Document Assistant with Fact-Checking Engine

A production-grade, context-grounded PDF Question-Answering system built with Python, Flask, PyMuPDF, Sentence Transformers, FAISS, and Google Gemini API.

Includes an intelligent **Fact-Checking & Citation Audit Engine** that scores answer groundedness and detects hallucinations claim-by-claim.

---

## Key Features

1. **Robust PDF Extraction (PyMuPDF)**
   - Page-by-page text extraction with character metrics.
   - Detects password-protected, encrypted, corrupted, and empty files.
   - Automatically identifies scanned / image-only PDFs that lack a text layer.

2. **Precision Text Chunking**
   - Sliding-window text chunker respecting sentence and word boundaries.
   - Preserves exact page numbers and document IDs across every chunk.

3. **Local Vector Similarity Search (FAISS + MiniLM)**
   - Vector embeddings generated using `all-MiniLM-L6-v2` (`SentenceTransformers`).
   - L2 normalized embeddings indexed in FAISS (`IndexFlatIP`) for exact cosine similarity search.
   - Retrieval with configurable `top_k` and deduplication of redundant context snippets.

4. **Context-Grounded Gemini Generation**
   - Answer synthesis strictly constrained to retrieved document excerpts.
   - Jittered exponential backoff retry for transient API rate limits (HTTP 429, 503, 504).
   - Dynamic multi-model fallback (`gemini-flash-lite-latest` -> `gemini-2.5-flash` -> `gemini-3.1-flash-lite` -> `gemini-3.8-flash`).
   - LRU deterministic query caching to prevent duplicate API costs.

5. **Fact-Checking & Citation Verification Engine**
   - Decomposes every generated response into discrete claims.
   - Audits each claim against retrieved PDF passages, assigning confidence and support levels (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`).
   - Verifies explicit page citations (e.g. `[Page 3]`) against actual pages in the retrieved chunks.
   - Calculates overall Groundedness Score (0% – 100%) and displays a visual audit badge.
   - Built-in interactive "Fact-Check Inspector" drawer to review verbatim evidence passages.
   - Honest refusal detection: if information is missing from the PDF, explicitly informs the user instead of hallucinating.

6. **Modern Vector-Art UI**
   - Dark glassmorphic theme with cyan, blue, and violet ambient glows.
   - Interactive drag-and-drop PDF upload zone.
   - Real-time document statistics (pages, chunks, character count, preview).
   - FAISS vector chunk inspector modal.
   - Groundedness score gauge and citation pill buttons.

7. **Vercel Serverless Ready**
   - Includes `api/index.py` entrypoint and `vercel.json` deployment rules.

---

## Architecture Flow

```
PDF Document 
    │
    ▼
[PyMuPDF Extractor] ───► Page-by-page text & metadata
    │
    ▼
[Sliding Chunker] ─────► Chunks with page numbers & unique IDs
    │
    ▼
[SentenceTransformers] ─► L2-Normalized Vectors
    │
    ▼
[FAISS IndexFlatIP] ───► Semantic Cosine Retrieval
    │
    ▼ (Top-K Chunks)
[Gemini LLM] ──────────► Grounded Answer with [Page X] Citations
    │
    ▼
[Fact-Check Engine] ───► Claim Verification, Citation Audit & Groundedness Score
    │
    ▼
[Modern UI Dashboard]
```

---

## Installation & Local Setup

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key (stored in `.env`)

### 2. Virtual Environment Setup
```bash
# Clone or open project directory
cd "PDF-based AI assistant"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and insert your Gemini API Key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
GEMINI_MODELS=gemini-flash-lite-latest,gemini-2.5-flash,gemini-3.1-flash-lite,gemini-3.8-flash

CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K_CHUNKS=4
MAX_PAGES=150
MAX_FILE_SIZE_MB=25

FACT_CHECK_STRICT_MODE=true
MIN_GROUNDEDNESS_SCORE=0.65
```

### 4. Running the Application
```bash
python app.py
```
Open your browser to `http://localhost:5000`.

---

## Running Automated Tests

Run the test suite with pytest:
```bash
pytest tests/ -v
```

Tests cover:
- `tests/test_pdf_extractor.py`: PDF text extraction, empty/scanned detection, page limits.
- `tests/test_chunker.py`: Boundary splitting, overlapping chunks, metadata retention.
- `tests/test_retriever.py`: FAISS vector addition, cosine similarity search, chunk retrieval.
- `tests/test_fact_checker.py`: Claim grounding, citation validation, hallucination detection, honest refusal.

---

## Deployment to Vercel

The project is structured with `api/index.py` and `vercel.json` for Vercel deployment:
1. Push project to GitHub (ensure `.env` and `venv` are gitignored).
2. Import project in Vercel.
3. Add `GEMINI_API_KEY` to Vercel Environment Variables.
4. Deploy!

*(Note: In serverless environments with strict memory constraints, external vector persistence such as Supabase / pgvector can be activated by providing `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `.env`.)*
