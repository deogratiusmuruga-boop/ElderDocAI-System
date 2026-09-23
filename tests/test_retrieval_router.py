"""Tests for scripts.retrieval_router and scripts.geri_lit_adapter (Task 10B).

Unit tests use mocks/stubs. No live Ollama, no model downloads, no
modification of frozen GeriLit indexes.
"""
import os
import sys
import unittest
from unittest import mock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import scripts.retrieval_router as router  # noqa: E402
from scripts.geri_lit_adapter import (  # noqa: E402
    AUTHORITY_PMC,
    DOCUMENT_CATEGORY_PMC,
    to_legacy_evidence,
)


class FakeChunk(dict):
    pass


SAMPLE_GERI_ITEM = {
    "row_id": 7,
    "chunk_id": "PMC123__BODY__000__PROSE__00002",
    "pmcid": "PMC123",
    "pmid": "456",
    "doi": "10.123/abc",
    "title": "Some title",
    "journal": "Some Journal",
    "publication_year": 2021,
    "region": "BODY",
    "section_id": "sec1",
    "section_title": "Methods",
    "content_type": "PROSE",
    "evidence_type": "RCT",
    "source_locator": "PMC123:BODY:sec1:2..3",
    "text": "This is the evidence sentence about older adults.",
    "dense_score": 0.8,
    "sparse_score": 0.2,
    "hybrid_score": 0.6,
    "rerank_score": 0.93,
    "source_document": "PMC123 (Some Journal, 2021)",
}


class RetrievalRouterTests(unittest.TestCase):

    def setUp(self):
        router.reset_geri_lit_singleton()
        self._old = os.environ.pop(router.BACKEND_ENV, None)

    def tearDown(self):
        if self._old is not None:
            os.environ[router.BACKEND_ENV] = self._old
        router.reset_geri_lit_singleton()

    def test_default_backend_legacy_when_env_absent(self):
        # A: env absent -> legacy
        self.assertEqual(router._enabled_backend(None), "legacy")

    def test_explicit_legacy_selects_legacy_retriever(self):
        # B: explicit legacy -> legacy retriever called
        with mock.patch("scripts.hybrid_retriever.hybrid_search",
                        return_value=[FakeChunk(chunk_id="legacy-1")]) as m:
            out = router.retrieve("query", backend="legacy", fallback=False)
            m.assert_called_once_with("query")
            self.assertEqual(out[0]["chunk_id"], "legacy-1")

    def test_geri_lit_selection_calls_geri_lit(self):
        # C: geri_lit -> GeriLit retriever called and adapted
        with mock.patch.object(router, "_geri_lit_retrieve",
                               return_value=[{"chunk_id": "GL-G1"}]) as m:
            out = router.retrieve("query", backend="geri_lit", fallback=False)
            m.assert_called_once()
            self.assertEqual(out[0]["chunk_id"], "GL-G1")

    def test_unknown_backend_falls_back_safely(self):
        # F: invalid backend -> deterministic safe default (legacy)
        with mock.patch("scripts.hybrid_retriever.hybrid_search",
                        return_value=[{"chunk_id": "legacy-1"}]) as m:
            out = router.retrieve("query", backend="bogus-backend",
                                  fallback=False)
            m.assert_called_once()
            self.assertTrue(out)

    def test_geri_lit_failure_falls_back_to_legacy(self):
        # G: geri_lit failure -> fallback to legacy (no fabricated evidence)
        with mock.patch.object(router, "_geri_lit",
                               side_effect=RuntimeError("boom")) as gl, \
             mock.patch("scripts.hybrid_retriever.hybrid_search",
                        return_value=[{"chunk_id": "legacy-1"}]) as hs:
            out = router.retrieve("query", backend="geri_lit", fallback=True)
            gl.assert_called_once()
            hs.assert_called_once()
            self.assertEqual(out[0]["chunk_id"], "legacy-1")

    def test_geri_lit_failure_no_fallback_returns_empty(self):
        # Gb: geri_lit failure without fallback -> safe empty list
        with mock.patch.object(router, "_geri_lit",
                               side_effect=RuntimeError("boom")):
            out = router.retrieve("query", backend="geri_lit", fallback=False)
            self.assertEqual(out, [])

    def test_both_backend_uses_legacy_only(self):
        # both -> legacy only (Task 10B compatibility; no fusion)
        with mock.patch("scripts.hybrid_retriever.hybrid_search",
                        return_value=[{"chunk_id": "legacy-1"}]) as m:
            out = router.retrieve("query", backend="both", fallback=False)
            m.assert_called_once()
            self.assertEqual(out[0]["chunk_id"], "legacy-1")

    def test_legacy_retriever_still_callable(self):
        # H: existing legacy retrieval remains callable directly
        import scripts.hybrid_retriever  # noqa: F401
        self.assertTrue(callable(router._legacy_retrieve))

    def test_adapter_mapping(self):
        ev = to_legacy_evidence([SAMPLE_GERI_ITEM])
        a = ev[0]
        # Task 2: similarity_score = dense cosine (semantic relevance);
        # rerank logit is kept separately as the ranking-only retrieval score.
        self.assertEqual(a["similarity_score"], 0.8)          # dense->sim
        self.assertEqual(a["retrieval_score"], 0.93)          # rerank->retrieval
        self.assertEqual(a["text"], SAMPLE_GERI_ITEM["text"])
        self.assertEqual(a["chunk_id"], SAMPLE_GERI_ITEM["chunk_id"])
        self.assertEqual(a["source_document"],
                         SAMPLE_GERI_ITEM["source_document"])
        self.assertEqual(a["document_category"], DOCUMENT_CATEGORY_PMC)
        self.assertEqual(a["category"], DOCUMENT_CATEGORY_PMC)
        self.assertAlmostEqual(a["authority_score"], AUTHORITY_PMC)
        self.assertIsNotNone(a["authority_score"])             # explicit

    def test_adapter_preserves_provenance(self):
        ev = to_legacy_evidence([SAMPLE_GERI_ITEM])
        a = ev[0]
        for f in ("row_id", "pmcid", "pmid", "doi", "title", "journal",
                  "publication_year", "region", "section_id",
                  "section_title", "content_type", "evidence_type",
                  "source_locator", "dense_score", "sparse_score",
                  "hybrid_score", "rerank_score"):
            self.assertIn(f, a)
            self.assertEqual(a[f], SAMPLE_GERI_ITEM[f])

    def test_adapter_does_not_mutate_input(self):
        src = dict(SAMPLE_GERI_ITEM)
        _ = to_legacy_evidence([src])
        self.assertEqual(src, SAMPLE_GERI_ITEM)


if __name__ == "__main__":
    unittest.main(verbosity=2)