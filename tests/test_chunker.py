from services.chunker import TextChunker

def test_chunker_short_text():
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    doc_data = {
        "filename": "test.pdf",
        "pages": [
            {"page_number": 1, "text": "This is a short text that easily fits in one chunk."}
        ]
    }
    chunks = chunker.chunk_document(doc_data, doc_id="doc_123")

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["doc_id"] == "doc_123"
    assert "short text" in chunks[0]["text"]

def test_chunker_sliding_window_overlap():
    chunker = TextChunker(chunk_size=100, chunk_overlap=30)
    # Long text with multiple sentences
    long_text = "Sentence one is here. " * 15
    doc_data = {
        "filename": "long.pdf",
        "pages": [
            {"page_number": 3, "text": long_text}
        ]
    }
    chunks = chunker.chunk_document(doc_data, doc_id="doc_long")

    assert len(chunks) > 1
    for c in chunks:
        assert c["page_number"] == 3
        assert len(c["text"]) > 0
        assert c["chunk_id"].startswith("doc_long_p3_c")
