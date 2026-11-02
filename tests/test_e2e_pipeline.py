import pytest
import pymupdf as fitz
from services.pdf_extractor import PDFExtractor
from services.chunker import TextChunker
from services.embedder import EmbeddingService
from services.retriever import FAISSRetriever
from services.gemini_service import GeminiService
from services.fact_checker import FactChecker
from core.config import Config

def test_full_rag_and_fact_check_pipeline():
    # 1. Create in-memory sample test PDF
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "DocuMind is an AI assistant developed in 2026. It features 99% accuracy in grounded retrieval.")
    p2 = doc.new_page()
    p2.insert_text((50, 72), "The system costs $15 per month for individual users and $49 per month for team licenses. It supports up to 150 pages per PDF.")
    pdf_bytes = doc.write()
    doc.close()

    # 2. Extract
    doc_data = PDFExtractor.extract_from_bytes(pdf_bytes, filename="pricing.pdf")
    assert doc_data["total_pages"] == 2
    assert not doc_data["is_scanned"]

    # 3. Chunk
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(doc_data, doc_id="doc_test_1")
    assert len(chunks) >= 2

    # 4. Embed
    embedder = EmbeddingService()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    assert embeddings.shape[0] == len(chunks)

    # 5. Index in FAISS
    retriever = FAISSRetriever(dimension=embedder.dimension)
    retriever.index_document("doc_test_1", chunks, embeddings)
    assert retriever.total_chunks == len(chunks)

    # 6. Retrieve
    query = "How much does the system cost for team licenses?"
    query_vec = embedder.embed_query(query)
    top_chunks = retriever.search(query_vec, top_k=2)
    assert len(top_chunks) > 0
    # Page 2 contains pricing information
    assert any(c["page_number"] == 2 for c in top_chunks)

    # 7. Generate Answer with Gemini
    gemini = GeminiService(api_key=Config.GEMINI_API_KEY, primary_model=Config.PRIMARY_MODEL)
    answer_data = gemini.generate_grounded_answer(query, top_chunks, doc_id="doc_test_1")
    assert "49" in answer_data["answer"]
    assert len(answer_data["citations"]) > 0

    # 8. Fact Check
    checker = FactChecker()
    audit = checker.audit_answer(answer_data["answer"], top_chunks)
    assert audit["status"] in ("VERIFIED", "PARTIALLY_VERIFIED")
    assert audit["groundedness_score"] >= 0.5
    assert len(audit["claims"]) > 0

    print("\n--- Pipeline Verified Successfully! ---")
    print("Answer:", answer_data["answer"])
    print("Fact Check Badge:", audit["badge"])
    print("Groundedness Score:", audit["groundedness_score"])
