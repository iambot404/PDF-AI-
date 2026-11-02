from services.fact_checker import FactChecker

def test_fact_checker_grounded_claims():
    checker = FactChecker()
    retrieved_chunks = [
        {
            "page_number": 2,
            "chunk_id": "c1",
            "text": "The company reported annual revenue of $45 million in 2024, representing 15% growth."
        }
    ]

    answer = "The company generated $45 million in revenue in 2024 with 15% growth [Page 2]."
    audit = checker.audit_answer(answer, retrieved_chunks)

    assert audit["status"] == "VERIFIED"
    assert audit["groundedness_score"] >= 0.8
    assert audit["citation_audit"]["all_citations_valid"] is True
    assert 2 in audit["citation_audit"]["valid_citations"]

def test_fact_checker_unverified_claims():
    checker = FactChecker()
    retrieved_chunks = [
        {
            "page_number": 1,
            "chunk_id": "c1",
            "text": "This document covers Python programming and syntax."
        }
    ]

    # Hallucinated answer claiming rocket science details not in context
    answer = "The rocket was launched into lunar orbit using liquid hydrogen propellant."
    audit = checker.audit_answer(answer, retrieved_chunks)

    assert audit["status"] == "UNVERIFIED"
    assert audit["groundedness_score"] < 0.5
    assert audit["claims"][0]["status"] == "UNSUPPORTED"

def test_fact_checker_honest_refusal():
    checker = FactChecker()
    retrieved_chunks = [
        {"page_number": 1, "chunk_id": "c1", "text": "Basic machine learning overview."}
    ]

    answer = "Based on the provided document excerpts, there is insufficient information to answer this question."
    audit = checker.audit_answer(answer, retrieved_chunks)

    assert audit["status"] == "HONEST_REFUSAL"
    assert audit["groundedness_score"] == 1.0
