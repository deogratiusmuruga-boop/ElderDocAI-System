"""Task 1/2 - Programmatic Reliability Gate + relevance semantics tests.

Mocks retrieval and the LLM only (no models loaded). Reliability scoring and
thresholds use the real production functions (fixtures self-checked).

Score semantics (Task 2): ``rerank_score`` = CrossEncoder ranking logit;
``dense_score`` = dense cosine semantic relevance. ``prepare_evidence`` must
feed the reliability 'relevance' factor the dense cosine and never a logit;
retrieval/reranking order is preserved.
"""
import unittest
from unittest import mock

import scripts.carebuddy_service as service
import scripts.geri_lit_adapter as adapter
import scripts.rag_chat as rag_chat
import scripts.retrieval_router as router
from scripts.adaptive_decision_controller import make_reliability_decision
from scripts.reliability_evaluation import evaluate_reliability

QUERY = "How can falls be prevented in older adults?"
SUPPORTING_TEXT = (
    "Falls prevention programs for older adults include balance and "
    "strength training."
)
NON_SUPPORTING_TEXT = "Weather forecast for tomorrow is clear and sunny."
MOCK_ANSWER = "Mocked grounded answer."


def chunk(chunk_id, text, authority_score, rerank_score,
          source_document="tips-take-medicines-safely.pdf",
          dense_score=None):
    out = {
        "chunk_id": chunk_id,
        "text": text,
        "source_document": source_document,
        "authority_score": authority_score,
        "rerank_score": rerank_score,
    }
    if dense_score is not None:
        out["dense_score"] = dense_score
    return out


def evidence_decision(chunks):
    items = rag_chat.prepare_evidence(chunks)
    return make_reliability_decision(
        evaluate_reliability(query=QUERY, evidence_items=items))["decision"]


class ReliabilityGateTests(unittest.TestCase):

    def setUp(self):
        self.chat_patch = mock.patch.object(
            rag_chat.ollama, "chat",
            return_value={"message": {"content": MOCK_ANSWER}})
        self.mock_chat = self.chat_patch.start()
        self.addCleanup(self.chat_patch.stop)

    def _gen(self, side_effect):
        with mock.patch("scripts.rag_chat.retrieve_evidence",
                        side_effect=side_effect) as mr, \
             mock.patch.object(rag_chat, "build_grounded_prompt",
                               wraps=rag_chat.build_grounded_prompt) as bgp:
            out = rag_chat.generate_answer(
                QUERY, return_evidence=True, return_evaluation=True)
        return out, mr, bgp

    def test_accept_generates_answer(self):
        ok = chunk("c1", SUPPORTING_TEXT, 1.0, 0.9, dense_score=0.9)
        self.assertEqual(evidence_decision([ok]), "ACCEPT")
        out, mr, _ = self._gen([[ok]])
        self.assertEqual(out["decision"]["decision"], "ACCEPT")
        self.assertEqual(out["answer"], MOCK_ANSWER)
        self.assertEqual(out["refused"], False)
        self.assertEqual(out["retrieval_attempts"], 1)
        self.assertEqual(mr.call_count, 1)
        self.assertEqual(self.mock_chat.call_count, 1)

    def test_refine_occurs_before_generation(self):
        weak = [
            chunk("c1", SUPPORTING_TEXT, 1.0, 0.2, dense_score=0.2),
            chunk("c2", SUPPORTING_TEXT, 1.0, 0.2, dense_score=0.2),
            chunk("c3", NON_SUPPORTING_TEXT, 1.0, 0.2, dense_score=0.2),
        ]
        self.assertEqual(evidence_decision(weak), "REFINE")
        out, mr, bgp = self._gen([weak])
        self.assertGreaterEqual(out["refinement_attempts"], 1)
        self.assertEqual(out["answer"], MOCK_ANSWER)
        self.assertEqual(mr.call_count, 1)
        self.assertEqual(self.mock_chat.call_count, 1)
        final_args = bgp.call_args_list[-1]
        self.assertEqual(len(final_args.kwargs["evidence_items"]), 2)

    def test_refine_budget_is_limited(self):
        # single supporting item at the cosine boundary (-1): stays REFINE
        weak = [chunk("c1", SUPPORTING_TEXT, 1.0, -8.0, dense_score=-1.0)]
        self.assertEqual(evidence_decision(weak), "REFINE")
        out, mr, _ = self._gen([weak])
        self.assertEqual(out["refinement_attempts"], 1)
        self.assertEqual(out["answer"], MOCK_ANSWER)
        self.assertEqual(self.mock_chat.call_count, 1)

    def test_reretrieve_performs_second_retrieval(self):
        weak = [chunk("c1", SUPPORTING_TEXT, 0.5, -20.0, dense_score=-1.0)]
        strong = [chunk("c1", SUPPORTING_TEXT, 1.0, 0.9, dense_score=0.9)]
        self.assertEqual(evidence_decision(weak), "RE-RETRIEVE")
        out, mr, _ = self._gen([weak, strong])
        self.assertEqual(mr.call_count, 2)
        self.assertEqual(out["retrieval_attempts"], 2)
        self.assertEqual(out["decision"]["decision"], "ACCEPT")
        self.assertEqual(out["answer"], MOCK_ANSWER)
        self.assertEqual(self.mock_chat.call_count, 1)

    def test_reretrieve_failure_does_not_generate(self):
        weak = [chunk("c1", SUPPORTING_TEXT, 0.5, -20.0, dense_score=-1.0)]
        out, mr, _ = self._gen([weak, weak])
        self.assertEqual(mr.call_count, 2)
        self.assertEqual(out["retrieval_attempts"], 2)
        self.assertTrue(out["refused"])
        self.assertEqual(out["decision"]["decision"], "RE-RETRIEVE")
        self.assertEqual(self.mock_chat.call_count, 0)
        self.assertEqual(out["answer"], rag_chat.REJECTION_RESPONSE)

    def test_reject_no_llm_generation(self):
        bad = [chunk("c1", SUPPORTING_TEXT, 0.1, -20.0, dense_score=-1.0)]
        self.assertEqual(evidence_decision(bad), "REJECT")
        out, mr, _ = self._gen([bad])
        self.assertEqual(out["decision"]["decision"], "REJECT")
        self.assertTrue(out["refused"])
        self.assertEqual(mr.call_count, 1)
        self.assertEqual(self.mock_chat.call_count, 0)
        self.assertEqual(out["answer"], rag_chat.REJECTION_RESPONSE)

    def test_empty_evidence_safe_behavior(self):
        out, mr, _ = self._gen([[]])
        self.assertEqual(
            out["answer"],
            "I couldn't find that information in the knowledge base.")
        self.assertTrue(out["refused"])
        self.assertEqual(self.mock_chat.call_count, 0)
        self.assertEqual(mr.call_count, 1)
        self.assertEqual(out["decision"]["decision"], "REJECT")


class ServiceIntegrationTests(unittest.TestCase):
    """carebuddy_service consumes the gate's single evaluation."""

    def setUp(self):
        self.chat_patch = mock.patch.object(
            rag_chat.ollama, "chat",
            return_value={"message": {"content": MOCK_ANSWER}})
        self.mock_chat = self.chat_patch.start()
        self.addCleanup(self.chat_patch.stop)

    def test_service_response_contract_and_single_evaluation(self):
        strong = [chunk("c1", SUPPORTING_TEXT, 1.0, 0.9, dense_score=0.9)]
        with mock.patch("scripts.rag_chat.retrieve_evidence",
                        return_value=strong), \
             mock.patch("scripts.rag_chat.evaluate_reliability",
                        wraps=evaluate_reliability) as ev:
            result = service.answer_question(
                question=QUERY, user_profile=None,
                conversation_history=[], response_language="en")

        for key in ("answer", "sources", "reliability", "decision",
                    "care_context", "profile_used"):
            self.assertIn(key, result)
        self.assertEqual(result["decision"]["decision"], "ACCEPT")
        self.assertIn("overall_reliability", result["reliability"])
        self.assertEqual(ev.call_count, 1)

    def test_service_returns_rejection_for_reject(self):
        bad = [chunk("c1", SUPPORTING_TEXT, 0.1, -20.0, dense_score=-1.0)]
        with mock.patch("scripts.rag_chat.retrieve_evidence",
                        return_value=bad):
            result = service.answer_question(
                question=QUERY, user_profile=None,
                conversation_history=[], response_language="en")
        self.assertEqual(result["decision"]["decision"], "REJECT")
        self.assertEqual(result["answer"], rag_chat.REJECTION_RESPONSE)
        # rejected evidence is still surfaced as sources (no LLM generation)
        self.assertEqual(result["sources"], ["tips-take-medicines-safely.pdf"])
        self.assertEqual(self.mock_chat.call_count, 0)
class RetrievalDefaultTests(unittest.TestCase):
    """The production retrieval default must remain legacy."""

    def test_router_default_backend_is_legacy(self):
        self.assertEqual(router.DEFAULT_BACKEND, "legacy")
        self.assertEqual(router._enabled_backend(None), "legacy")

    def test_rag_chat_uses_the_router_retrieve(self):
        import scripts.retrieval_router as rr
        self.assertIs(rag_chat.retrieve_evidence, rr.retrieve)


class RelevanceSemanticsTests(unittest.TestCase):
    """Task 2 - reliability relevance must use dense cosine, never a logit."""

    def test_a_cross_encoder_logits_never_fill_relevance(self):
        chunks = [
            chunk("P", SUPPORTING_TEXT, 1.0, 6.5, dense_score=-0.4),
            chunk("N", SUPPORTING_TEXT, 1.0, -12.0, dense_score=0.7),
        ]
        items = rag_chat.prepare_evidence(chunks)
        sims = [i["similarity_score"] for i in items]
        self.assertEqual(sims, [-0.4, 0.7])          # dense cosine, not logits
        self.assertEqual([i["retrieval_score"] for i in items],
                         [6.5, -12.0])               # logits kept for ranking
        rel = evaluate_reliability(query=QUERY, evidence_items=items)
        expected = (((-0.4 + 1) / 2) + ((0.7 + 1) / 2)) / 2.0
        self.assertAlmostEqual(rel["relevance"], expected)

    def test_b_cosine_normalization_map(self):
        from scripts.reliability_evaluation import (
            _normalized_cosine_similarity as f)
        for raw, expected in [(-1.0, 0.0), (-0.5, 0.25), (0.0, 0.5),
                              (0.5, 0.75), (1.0, 1.0)]:
            self.assertAlmostEqual(f(raw), expected)

    def test_c_ranking_order_preserved_with_separated_scores(self):
        ranked = [
            chunk("A", SUPPORTING_TEXT, 1.0, 3.0, dense_score=0.8),
            chunk("B", SUPPORTING_TEXT, 1.0, -4.5, dense_score=0.6),
            chunk("C", SUPPORTING_TEXT, 1.0, -9.0, dense_score=0.4),
        ]
        items = rag_chat.prepare_evidence(ranked)
        # retrieval order is preserved verbatim
        self.assertEqual([i["chunk_id"] for i in items], ["A", "B", "C"])
        # similarity == dense cosine (not the logits)
        self.assertEqual([i["similarity_score"] for i in items],
                         [0.8, 0.6, 0.4])
        # ranking score (logit) is kept but not used as relevance
        self.assertEqual([i["retrieval_score"] for i in items],
                         [3.0, -4.5, -9.0])

    def test_c_before_after_ranking_identity(self):
        chunks = [
            chunk("A", SUPPORTING_TEXT, 1.0, 3.0, dense_score=0.8),
            chunk("B", SUPPORTING_TEXT, 1.0, -4.5, dense_score=0.6),
            chunk("C", SUPPORTING_TEXT, 1.0, -9.0, dense_score=0.4),
        ]
        old_similarity = [c["rerank_score"] for c in chunks]   # old semantics
        items = rag_chat.prepare_evidence(chunks)
        new_similarity = [i["similarity_score"] for i in items]
        # ranking by rerank unchanged; the relevance input changed semantics
        self.assertEqual([i["chunk_id"] for i in items],
                         [c["chunk_id"] for c in chunks])
        self.assertNotEqual(new_similarity, old_similarity)

    def test_d_reliability_receives_dense_cosine(self):
        chunks = [chunk("X", SUPPORTING_TEXT, 1.0, 9.0, dense_score=0.2)]
        items = rag_chat.prepare_evidence(chunks)
        rel = evaluate_reliability(query=QUERY, evidence_items=items)
        self.assertAlmostEqual(rel["relevance"], (0.2 + 1) / 2.0)

    def test_float_drift_clamped_safely(self):
        hi = rag_chat.prepare_evidence(
            [chunk("HI", SUPPORTING_TEXT, 1.0, -5.0, dense_score=1.0000001)])
        lo = rag_chat.prepare_evidence(
            [chunk("LO", SUPPORTING_TEXT, 1.0, -5.0, dense_score=-1.0000001)])
        self.assertEqual(hi[0]["similarity_score"], 1.0)
        self.assertEqual(lo[0]["similarity_score"], -1.0)

    def test_adapter_separates_scores(self):
        item = {
            "row_id": 0, "chunk_id": "PMC1__BODY__000__PROSE__00001",
            "pmcid": "PMC1", "pmid": "1", "doi": "10.x/x",
            "title": "T", "journal": "J", "publication_year": 2020,
            "region": "BODY", "section_id": "s", "section_title": "Intro",
            "content_type": "PROSE", "evidence_type": "RCT",
            "source_locator": "PMC1:BODY:s:0..1", "text": "Evidence text.",
            "dense_score": 0.8, "sparse_score": 0.2, "hybrid_score": 0.5,
            "rerank_score": 0.93, "source_document": "PMC1 (J, 2020)",
        }
        (adapted,) = adapter.to_legacy_evidence([item])
        self.assertEqual(adapted["similarity_score"], 0.8)
        self.assertEqual(adapted["retrieval_score"], 0.93)


if __name__ == "__main__":
    unittest.main(verbosity=2)