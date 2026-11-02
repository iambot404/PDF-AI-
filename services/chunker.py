import re
from typing import List, Dict, Any

class TextChunker:
    """
    Splits page-by-page document text into overlapping chunks,
    preserving exact page references and generating unique IDs.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 4)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_data: Dict[str, Any], doc_id: str) -> List[Dict[str, Any]]:
        """
        Takes extracted document data (with page list) and creates chunks.
        """
        chunks: List[Dict[str, Any]] = []
        filename = doc_data.get("filename", "unknown.pdf")
        pages = doc_data.get("pages", [])

        global_chunk_idx = 0

        for page_info in pages:
            page_num = page_info.get("page_number", 1)
            raw_text = page_info.get("text", "")
            
            # Normalize whitespace
            clean_text = re.sub(r'[ \t]+', ' ', raw_text).strip()
            if not clean_text:
                continue

            # If the page text is shorter than or equal to chunk_size, keep as a single chunk
            if len(clean_text) <= self.chunk_size:
                chunk_id = f"{doc_id}_p{page_num}_c{global_chunk_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": page_num,
                    "text": clean_text,
                    "char_count": len(clean_text)
                })
                global_chunk_idx += 1
                continue

            # Sliding window chunking with word boundary respect
            start = 0
            text_len = len(clean_text)
            
            while start < text_len:
                end = start + self.chunk_size
                if end >= text_len:
                    end = text_len
                else:
                    # Look for nearest sentence end (.!?) or whitespace to avoid cutting words
                    split_candidate = re.search(r'[.!?\n]\s', clean_text[max(start, end - 60):min(text_len, end + 40)])
                    if split_candidate:
                        end = max(start, end - 60) + split_candidate.end()
                    else:
                        space_idx = clean_text.rfind(' ', start, end)
                        if space_idx > start + (self.chunk_size // 2):
                            end = space_idx

                chunk_text = clean_text[start:end].strip()
                if chunk_text:
                    chunk_id = f"{doc_id}_p{page_num}_c{global_chunk_idx}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "filename": filename,
                        "page_number": page_num,
                        "text": chunk_text,
                        "char_count": len(chunk_text)
                    })
                    global_chunk_idx += 1

                if end >= text_len:
                    break

                # Advance start by chunk_size - chunk_overlap
                step = max(1, (end - start) - self.chunk_overlap)
                start += step

        return chunks
