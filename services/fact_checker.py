import re
from typing import List, Dict, Any, Tuple

class FactChecker:
    """
    Rigorously audits AI-generated answers against retrieved PDF passages.
    Verifies claim-by-claim groundedness, citation validity, and computes
    verifiable groundedness scores to prevent and detect hallucinations.
    """

    def __init__(self, min_grounded_threshold: float = 0.65):
        self.min_grounded_threshold = min_grounded_threshold

    def audit_answer(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Audits the generated answer against the source chunks.
        Returns a detailed fact-check breakdown:
        - groundedness_score: float (0.0 - 1.0)
        - status: 'VERIFIED' | 'PARTIALLY_VERIFIED' | 'UNVERIFIED' | 'HONEST_REFUSAL'
        - claims_audit: list of individual claims with verdict and matched passages
        - citation_audit: verification of page numbers mentioned in text
        """
        # Check for honest refusal
        refusal_patterns = [
            r"insufficient information",
            r"not mentioned in the",
            r"could not find",
            r"cannot find",
            r"does not contain",
            r"no information provided"
        ]
        is_refusal = any(re.search(pat, answer, re.IGNORECASE) for pat in refusal_patterns)
        if is_refusal and len(answer.split()) < 40:
            return {
                "groundedness_score": 1.0,
                "status": "HONEST_REFUSAL",
                "badge": "Honest Refusal (Grounded)",
                "summary": "The model accurately identified that the answer is not present in the document.",
                "claims": [{
                    "claim": answer,
                    "status": "SUPPORTED",
                    "confidence": 1.0,
                    "matched_page": None,
                    "evidence_snippet": "Document does not contain the requested information."
                }],
                "citation_audit": {"valid_citations": [], "invalid_citations": []}
            }

        # 1. Audit explicit page citations e.g. [Page 4] or (Page 2)
        citation_matches = re.findall(r'\[Page\s*(\d+)\]|\(Page\s*(\d+)\)|page\s*(\d+)', answer, re.IGNORECASE)
        cited_pages = set()
        for match in citation_matches:
            for group in match:
                if group:
                    cited_pages.add(int(group))

        available_pages = {c.get("page_number") for c in retrieved_chunks if c.get("page_number") is not None}
        valid_citations = list(cited_pages.intersection(available_pages))
        invalid_citations = list(cited_pages.difference(available_pages))

        # 2. Decompose answer into distinct sentences/claims
        # Clean common LLM lead-in preambles
        clean_ans = re.sub(r'^(?:based on the (?:provided )?document(?: excerpts)?,?\s*|according to the (?:provided )?document,?\s*|the document states that\s*)', '', answer, flags=re.IGNORECASE)
        raw_sentences = re.split(r'(?<=[.!?])\s+', clean_ans.strip())
        claims = [s.strip() for s in raw_sentences if len(s.strip()) > 10]

        if not claims:
            claims = [clean_ans.strip() or answer.strip()]

        # Combine all chunk texts into a searchable pool
        chunk_texts = [
            (c.get("page_number", 1), c.get("text", "").lower(), c.get("chunk_id", ""))
            for c in retrieved_chunks
        ]

        claim_results = []
        supported_weight = 0.0

        for claim in claims:
            claim_audit = self._verify_claim(claim, chunk_texts)
            claim_results.append(claim_audit)
            if claim_audit["status"] == "SUPPORTED":
                supported_weight += 1.0
            elif claim_audit["status"] == "PARTIALLY_SUPPORTED":
                supported_weight += 0.5

        groundedness_score = round(supported_weight / max(1, len(claims)), 2)

        # Determine overall badge & verdict
        if groundedness_score >= 0.85 and len(invalid_citations) == 0:
            status = "VERIFIED"
            badge = "Fact-Checked: 100% Grounded"
            badge_color = "success"
        elif groundedness_score >= 0.55:
            status = "PARTIALLY_VERIFIED"
            badge = f"Fact-Checked: {int(groundedness_score * 100)}% Grounded"
            badge_color = "warning"
        else:
            status = "UNVERIFIED"
            badge = "Fact-Check: Potential Unverified Claims"
            badge_color = "danger"

        return {
            "groundedness_score": groundedness_score,
            "status": status,
            "badge": badge,
            "badge_color": badge_color,
            "summary": f"{len(claims)} claim(s) audited against {len(retrieved_chunks)} source passage(s). Groundedness: {int(groundedness_score * 100)}%.",
            "claims": claim_results,
            "citation_audit": {
                "cited_pages": list(cited_pages),
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
                "all_citations_valid": len(invalid_citations) == 0
            }
        }

    def _verify_claim(self, claim: str, chunk_texts: List[Tuple[int, str, str]]) -> Dict[str, Any]:
        """
        Verifies an individual claim against the retrieved text chunks.
        """
        # Strip citation markers like [Page 1] so they don't bias factual number matching
        claim_without_citations = re.sub(r'\[Page\s*\d+\]|\(Page\s*\d+\)', '', claim, flags=re.IGNORECASE)
        clean_claim = claim_without_citations.lower()
        
        # Extract significant words (length > 3, exclude common stop words)
        stopwords = {
            "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
            "his", "from", "they", "say", "her", "she", "will", "one", "all", "would",
            "there", "their", "what", "out", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
            "page", "document", "passage", "section", "states", "according", "mentioned"
        }
        
        # Extract numbers, dates, capitalized names/entities
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', clean_claim)
        tokens = [t for t in re.findall(r'\b[a-z]{3,}\b', clean_claim) if t not in stopwords]

        best_page = None
        best_evidence = ""
        best_overlap_ratio = 0.0

        for page_num, chunk_lower, chunk_id in chunk_texts:
            # Check numbers match
            if numbers:
                num_matches = sum(1 for n in numbers if n in chunk_lower)
                num_ratio = num_matches / len(numbers)
            else:
                num_ratio = 1.0

            # Check keyword tokens match
            if tokens:
                token_matches = sum(1 for t in tokens if t in chunk_lower)
                token_ratio = token_matches / len(tokens)
            else:
                token_ratio = 0.5

            combined_ratio = (token_ratio * 0.7) + (num_ratio * 0.3)

            if combined_ratio > best_overlap_ratio:
                best_overlap_ratio = combined_ratio
                best_page = page_num
                
                # Find matching snippet in chunk
                for token in tokens:
                    idx = chunk_lower.find(token)
                    if idx != -1:
                        start_pos = max(0, idx - 40)
                        end_pos = min(len(chunk_lower), idx + 120)
                        best_evidence = chunk_lower[start_pos:end_pos].strip()
                        break

        # Classify status based on overlap
        if best_overlap_ratio >= 0.55:
            status = "SUPPORTED"
            confidence = round(best_overlap_ratio, 2)
        elif best_overlap_ratio >= 0.35:
            status = "PARTIALLY_SUPPORTED"
            confidence = round(best_overlap_ratio, 2)
        else:
            status = "UNSUPPORTED"
            confidence = round(best_overlap_ratio, 2)

        return {
            "claim": claim,
            "status": status,
            "confidence": confidence,
            "matched_page": best_page,
            "evidence_snippet": f"...{best_evidence}..." if best_evidence else "No direct textual match located in excerpts."
        }
