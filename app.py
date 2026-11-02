import os
import uuid
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from core.config import Config
from services import (
    PDFExtractor,
    PDFExtractionError,
    TextChunker,
    EmbeddingService,
    FAISSRetriever,
    GeminiService,
    GeminiServiceError,
    FactChecker,
    SupabaseService
)

# Base directory for resolving static and template paths
BASE_DIR = Path(__file__).resolve().parent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
        static_url_path="/static"
    )
    app.config.from_object(config_class)
    CORS(app)

    # Initialize services
    chunker = TextChunker(chunk_size=config_class.CHUNK_SIZE, chunk_overlap=config_class.CHUNK_OVERLAP)
    retriever = FAISSRetriever()
    fact_checker = FactChecker(min_grounded_threshold=config_class.MIN_GROUNDEDNESS_SCORE)
    supabase_service = SupabaseService(url=config_class.SUPABASE_URL, key=config_class.SUPABASE_ANON_KEY)
    
    # Lazy loaded embedder & gemini services to handle startup gracefully
    _embedder = None
    _gemini = None

    def get_embedder():
        nonlocal _embedder
        if _embedder is None:
            _embedder = EmbeddingService(model_name=config_class.EMBEDDING_MODEL_NAME)
        return _embedder

    def get_gemini():
        nonlocal _gemini
        if _gemini is None:
            _gemini = GeminiService(
                api_key=config_class.GEMINI_API_KEY,
                primary_model=config_class.PRIMARY_MODEL,
                fallback_models=config_class.FALLBACK_MODELS,
                max_retries=config_class.RETRY_MAX_ATTEMPTS,
                base_delay=config_class.RETRY_BASE_DELAY
            )
        return _gemini

    # In-memory document storage for session
    documents_store = {}

    @app.route("/")
    @app.route("/api")
    @app.route("/api/index")
    def index():
        return render_template(
            "index.html",
            model_name=config_class.PRIMARY_MODEL,
            max_pages=config_class.MAX_PAGES
        )

    @app.route("/static/<path:filename>")
    def serve_static(filename):
        return send_from_directory(str(BASE_DIR / "static"), filename)

    @app.route("/api/upload", methods=["POST"])
    @app.route("/upload", methods=["POST"])
    def upload_pdf():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        raw_filename = file.filename
        if not raw_filename.lower().endswith(".pdf"):
            return jsonify({"error": "Invalid file type. Only PDF documents are supported."}), 400

        safe_filename = secure_filename(raw_filename) or f"document_{uuid.uuid4().hex[:6]}.pdf"

        try:
            file_bytes = file.read()
            if len(file_bytes) == 0:
                return jsonify({"error": "Uploaded file is 0 bytes."}), 400

            # 1. Extract text page-by-page with PyMuPDF
            doc_data = PDFExtractor.extract_from_bytes(file_bytes, filename=safe_filename, max_pages=config_class.MAX_PAGES)
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"

            # 2. Split into overlapping chunks with page metadata
            chunks = chunker.chunk_document(doc_data, doc_id)

            if len(chunks) == 0:
                if doc_data.get("is_scanned"):
                    return jsonify({
                        "error": "This PDF appears to be a scanned image with no extractable text layer. OCR is required to process it."
                    }), 400
                return jsonify({"error": "Could not extract any readable text chunks from this document."}), 400

            # 3. Generate embeddings
            embedder = get_embedder()
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_texts(chunk_texts)

            # 4. Index in vector store
            retriever.index_document(doc_id, chunks, embeddings)

            # 5. Store document metadata
            doc_meta = {
                "doc_id": doc_id,
                "filename": safe_filename,
                "total_pages": doc_data["total_pages"],
                "total_chunks": len(chunks),
                "total_characters": doc_data["total_characters"],
                "is_scanned": doc_data["is_scanned"],
                "preview": doc_data["preview"],
                "chunks": chunks
            }
            documents_store[doc_id] = doc_meta
            supabase_service.save_document_meta(doc_id, doc_meta)

            logger.info(f"Indexed document {safe_filename} ({doc_id}) with {len(chunks)} chunks.")

            return jsonify({
                "success": True,
                "doc_id": doc_id,
                "filename": safe_filename,
                "total_pages": doc_data["total_pages"],
                "total_chunks": len(chunks),
                "total_characters": doc_data["total_characters"],
                "is_scanned": doc_data["is_scanned"],
                "preview": doc_data["preview"]
            })

        except PDFExtractionError as pe:
            return jsonify({"error": str(pe)}), 400
        except Exception as e:
            logger.error(f"Error processing upload: {e}", exc_info=True)
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

    @app.route("/api/query", methods=["POST"])
    @app.route("/query", methods=["POST"])
    def query_document():
        data = request.get_json() or {}
        doc_id = data.get("doc_id")
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        if not doc_id or doc_id not in documents_store:
            return jsonify({"error": "Document not found or session expired. Please re-upload the PDF."}), 404

        doc_meta = documents_store[doc_id]

        try:
            # 1. Embed query
            embedder = get_embedder()
            query_vector = embedder.embed_query(question)

            # 2. Retrieve top-k relevant chunks
            retrieved_chunks = retriever.search(
                query_vector,
                doc_id=doc_id,
                top_k=config_class.TOP_K_CHUNKS,
                score_threshold=0.0
            )

            # 3. Call Gemini with context grounding & retry
            gemini = get_gemini()
            answer_payload = gemini.generate_grounded_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
                doc_id=doc_id,
                filename=doc_meta["filename"]
            )

            # 4. Strict Fact-Checking Audit against retrieved passages
            fact_check_result = fact_checker.audit_answer(
                answer=answer_payload["answer"],
                retrieved_chunks=retrieved_chunks
            )

            # 5. Log QA event
            supabase_service.log_query_event({
                "doc_id": doc_id,
                "question": question,
                "groundedness_score": fact_check_result["groundedness_score"],
                "status": fact_check_result["status"],
                "model_used": answer_payload["model_used"]
            })

            return jsonify({
                "success": True,
                "question": question,
                "answer": answer_payload["answer"],
                "citations": answer_payload["citations"],
                "fact_check": fact_check_result,
                "model_used": answer_payload["model_used"],
                "cached": answer_payload["cached"],
                "latency_ms": answer_payload["latency_ms"]
            })

        except GeminiServiceError as ge:
            return jsonify({"error": f"Gemini AI Service Error: {str(ge)}"}), 502
        except Exception as e:
            logger.error(f"Error during query: {e}", exc_info=True)
            return jsonify({"error": f"Failed to answer question: {str(e)}"}), 500

    @app.route("/api/document/<doc_id>/chunks", methods=["GET"])
    @app.route("/document/<doc_id>/chunks", methods=["GET"])
    def get_chunks(doc_id):
        if doc_id not in documents_store:
            return jsonify({"error": "Document not found"}), 404

        chunks = documents_store[doc_id].get("chunks", [])
        return jsonify({
            "doc_id": doc_id,
            "filename": documents_store[doc_id]["filename"],
            "chunks": chunks
        })

    @app.route("/api/document/<doc_id>", methods=["DELETE"])
    @app.route("/document/<doc_id>", methods=["DELETE"])
    def delete_document(doc_id):
        if doc_id in documents_store:
            del documents_store[doc_id]
        retriever.remove_document(doc_id)
        supabase_service.delete_document(doc_id)
        return jsonify({"success": True, "message": "Document deleted and index cleared."})

    @app.route("/api/health", methods=["GET"])
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "primary_model": config_class.PRIMARY_MODEL,
            "total_indexed_chunks": retriever.total_chunks,
            "documents_count": len(documents_store)
        })

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
