import numpy as np
from services.retriever import FAISSRetriever

def test_faiss_retriever_indexing_and_search():
    dim = 4
    retriever = FAISSRetriever(dimension=dim)

    # 3 normalized vectors
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)

    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "page_number": 1, "text": "First chunk on topic A"},
        {"chunk_id": "c2", "doc_id": "d1", "page_number": 2, "text": "Second chunk on topic B"},
        {"chunk_id": "c3", "doc_id": "d2", "page_number": 1, "text": "Third chunk on topic C"}
    ]

    indexed_count = retriever.index_document("d1", chunks[:2], vecs[:2])
    assert indexed_count == 2
    assert retriever.total_chunks == 2

    # Query vector closest to chunk 1
    query = np.array([[0.95, 0.05, 0.0, 0.0]], dtype=np.float32)
    results = retriever.search(query, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] > 0.9


def test_hybrid_retriever_bm25_ranking():
    retriever = FAISSRetriever(dimension=4)
    chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "page_number": 2, "text": "He had gone eighty-four days without taking a fish in the Gulf Stream."},
        {"chunk_id": "c2", "doc_id": "d1", "page_number": 10, "text": "The boy went to buy sardines and fresh bait for tomorrow."},
        {"chunk_id": "c3", "doc_id": "d1", "page_number": 25, "text": "The marlin pulled steadily through the dark cold water."}
    ]
    # Dummy uniform vectors
    vecs = np.zeros((3, 4), dtype=np.float32)
    retriever.index_document("d1", chunks, vecs)

    # Search with keyword text
    query_vec = np.zeros((1, 4), dtype=np.float32)
    results = retriever.search(query_vec, doc_id="d1", top_k=1, query_text="How many days without catching a fish?")
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["page_number"] == 2
    assert "bm25_score" in results[0]

