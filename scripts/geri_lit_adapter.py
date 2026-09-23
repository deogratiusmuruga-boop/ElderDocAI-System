#!/usr/bin/env python3
"""ElderDocAI GeriLit Evidence Adapter - Phase 3 Task 10B.

Converts GeriLitRetriever.retrieve() output into the evidence representation
expected by the existing ElderDocAI pipeline (prepare_evidence / reliability
evaluation / prompt builder).

Mappings (Task 10A + Task 2 correction):
    dense_score       -> similarity_score  (dense cosine SEMANTIC relevance,
                                             used by reliability 'relevance')
    rerank_score      -> retrieval_score   (CrossEncoder logit; RANKING ONLY,
                                             never consumed by reliability)
    text              -> text
    chunk_id          -> chunk_id
    source_document   -> source_document
    document_category -> "PMC"
    authority_score   -> deterministic injected value (below)

AUTHORITY VALUE (documented, integration-level only)
-----------------------------------------------------
GeriLit/PMC sources are NOT classified by the existing organization-based
authority_mapping (WHO/NIA/NIH/CDC/.gov), and no clinically meaningful
authority claim is made here. The adapter injects a CONSERVATIVE DETERMINISTIC
constant, AUTHORITY_PMC = 0.85, explicitly labelled as an integration value
("PMC-OA scholarly publication") rather than a clinical-quality score. This
avoids silently relying on prepare_evidence()'s default authority of 1.0 and
avoids pretending the existing authority mapping can rate biomedical evidence.

All useful GeriLit provenance fields are preserved unchanged in each output
item. The original GeriLit result objects are never modified (new dicts are
built).
"""
from typing import Any, Dict, List

# Conservative, deterministic integration authority (documented above).
# Deliberately < 1.0 and mirrors the spirit of the legacy "Other trusted
# medical" bucket (0.85) without claiming WHO/NIA-level authority.
AUTHORITY_PMC = 0.85
DOCUMENT_CATEGORY_PMC = "PMC"

# Provenance fields to carry through verbatim when present.
PROVENANCE_FIELDS = (
    "row_id",
    "chunk_id",
    "pmcid",
    "pmid",
    "doi",
    "title",
    "journal",
    "publication_year",
    "region",
    "section_id",
    "section_title",
    "content_type",
    "evidence_type",
    "source_locator",
    "text",
    "dense_score",
    "sparse_score",
    "hybrid_score",
    "rerank_score",
    "source_document",
)


def _dense_cosine(item):
    """Dense-cosine semantic relevance in [-1, 1].

    Task 2: CrossEncoder rerank logits are unbounded ranking signals and are
    never interpreted as cosine similarity. When the dense BGE cosine is
    unavailable, an already-computed similarity field is the fallback; any
    floating-point drift outside [-1, 1] is clamped explicitly.
    """
    raw = None
    if "dense_score" in item and item.get("dense_score") is not None:
        raw = float(item["dense_score"])
    elif ("similarity_score" in item
          and item.get("similarity_score") is not None):
        raw = float(item["similarity_score"])
    if raw is None:
        return 0.0
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return raw


def to_legacy_evidence(geri_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adapt GeriLit retriever output items to the legacy evidence shape.

    Score semantics (Task 2 - reliability relevance normalization):
      - ``similarity_score`` = dense cosine semantic relevance [-1, 1]
        (reliability 'relevance' input; NEVER a CrossEncoder logit).
      - ``retrieval_score``  = CrossEncoder rerank logit (ranking only;
        NOT consumed by reliability).
      The input ranking order (already reranked by the retriever) is preserved.

    Returns a NEW list of NEW dicts; input items are not mutated.
    """
    output = []
    for item in geri_items or []:
        adapted: Dict[str, Any] = {}
        for field in PROVENANCE_FIELDS:
            if field in item:
                adapted[field] = item[field]
        # canonical legacy keys
        adapted["chunk_id"] = item.get("chunk_id")
        adapted["source_document"] = item.get("source_document") or (
            item.get("pmcid") or "Unknown")
        adapted["text"] = item.get("text", "")
        adapted["document_category"] = DOCUMENT_CATEGORY_PMC
        adapted["category"] = DOCUMENT_CATEGORY_PMC
        adapted["authority_score"] = AUTHORITY_PMC
        adapted["similarity_score"] = _dense_cosine(item)
        adapted["retrieval_score"] = item.get(
            "rerank_score",
            item.get("hybrid_score", item.get("dense_score", 0.0)))
        output.append(adapted)
    return output


if __name__ == "__main__":
    sample = [{
        "row_id": 0, "chunk_id": "PMC1__BODY__000__PROSE__00001",
        "pmcid": "PMC1", "pmid": "1", "doi": "10.x/x",
        "title": "T", "journal": "J", "publication_year": 2020,
        "region": "BODY", "section_id": "s", "section_title": "Intro",
        "content_type": "PROSE", "evidence_type": "RCT",
        "source_locator": "PMC1:BODY:s:0..1", "text": "Evidence text.",
        "dense_score": 0.8, "sparse_score": 0.2, "hybrid_score": 0.5,
        "rerank_score": 0.9, "source_document": "PMC1 (J, 2020)",
    }]
    adapted = to_legacy_evidence(sample)
    print(adapted[0]["chunk_id"], adapted[0]["authority_score"],
          adapted[0]["similarity_score"], adapted[0]["document_category"])