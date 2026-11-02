import os
from typing import Dict, Any, List
import pymupdf as fitz  # PyMuPDF

class PDFExtractionError(Exception):
    """Custom exception for PDF extraction issues."""
    pass

class PDFExtractor:
    """
    Extracts text page-by-page from PDF files using PyMuPDF.
    Gracefully handles encrypted, empty, scanned, or malformed documents.
    """

    @staticmethod
    def extract_from_path(file_path: str, max_pages: int = 150) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF file on disk.
        """
        if not os.path.exists(file_path):
            raise PDFExtractionError(f"File not found: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise PDFExtractionError(f"Failed to open PDF document. It may be corrupt or invalid: {str(e)}")

        return PDFExtractor._process_document(doc, os.path.basename(file_path), os.path.getsize(file_path), max_pages)

    @staticmethod
    def extract_from_bytes(file_bytes: bytes, filename: str = "document.pdf", max_pages: int = 150) -> Dict[str, Any]:
        """
        Extract text and metadata directly from in-memory PDF bytes.
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise PDFExtractionError(f"Failed to parse PDF bytes: {str(e)}")

        return PDFExtractor._process_document(doc, filename, len(file_bytes), max_pages)

    @staticmethod
    def _process_document(doc: fitz.Document, filename: str, file_size: int, max_pages: int) -> Dict[str, Any]:
        if doc.is_encrypted:
            doc.close()
            raise PDFExtractionError("The uploaded PDF is password-protected or encrypted. Please decrypt it first.")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise PDFExtractionError("The uploaded PDF contains 0 pages.")

        if total_pages > max_pages:
            doc.close()
            raise PDFExtractionError(
                f"Document exceeds maximum allowed page count ({total_pages} pages > limit of {max_pages})."
            )

        pages: List[Dict[str, Any]] = []
        total_text_length = 0

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()
            char_count = len(page_text)
            total_text_length += char_count

            pages.append({
                "page_number": page_idx + 1,
                "text": page_text,
                "char_count": char_count
            })

        doc.close()

        # Check for scanned PDF (pages exist but text is essentially empty)
        is_scanned = False
        if total_pages > 0 and (total_text_length / total_pages) < 25:
            is_scanned = True

        # Generate a preview of the first readable content
        preview = ""
        for p in pages:
            if p["text"]:
                preview = p["text"][:350] + ("..." if len(p["text"]) > 350 else "")
                break

        return {
            "filename": filename,
            "file_size": file_size,
            "total_pages": total_pages,
            "total_characters": total_text_length,
            "is_scanned": is_scanned,
            "pages": pages,
            "preview": preview or "(No extractable text found. Document might be scanned or image-only.)"
        }
