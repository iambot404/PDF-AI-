import pytest
import pymupdf as fitz
from services.pdf_extractor import PDFExtractor, PDFExtractionError

def create_sample_pdf_bytes(pages_text: list) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def test_extract_valid_pdf():
    sample_text = ["DocuMind is an AI assistant.", "It uses PyMuPDF and FAISS for RAG."]
    pdf_bytes = create_sample_pdf_bytes(sample_text)

    result = PDFExtractor.extract_from_bytes(pdf_bytes, filename="test.pdf")

    assert result["filename"] == "test.pdf"
    assert result["total_pages"] == 2
    assert len(result["pages"]) == 2
    assert result["pages"][0]["page_number"] == 1
    assert "DocuMind" in result["pages"][0]["text"]
    assert result["pages"][1]["page_number"] == 2
    assert "FAISS" in result["pages"][1]["text"]
    assert not result["is_scanned"]

def test_extract_empty_pages_detected_as_scanned():
    # Empty page PDF
    pdf_bytes = create_sample_pdf_bytes(["", ""])
    result = PDFExtractor.extract_from_bytes(pdf_bytes, filename="empty.pdf")

    assert result["total_pages"] == 2
    assert result["is_scanned"] is True

def test_extract_max_pages_limit_exceeded():
    pdf_bytes = create_sample_pdf_bytes(["P1", "P2", "P3", "P4"])
    with pytest.raises(PDFExtractionError, match="maximum allowed page count"):
        PDFExtractor.extract_from_bytes(pdf_bytes, max_pages=2)
