"""
ElderDocAI / CareBuddy RAG Generation Module

Pipeline:

    User Query
        ↓
    Adaptive Context
        ↓
    Hybrid Retrieval
        ↓
    Evidence Preparation
        ↓
    Reliability Evaluation
        ↓
    Adaptive Decision
        ↓
    Grounded Prompt
        ↓
    Llama 3.2
        ↓
    Evidence-Grounded Response

Adaptive context contains:
    - patient profile
    - current care state
    - temporal transition
    - changed dimensions
    - adaptive assistance

Important:
    Adaptive context describes documented care activity and temporal
    changes. It is not a diagnosis and does not predict medical risk.
"""

import json
from pathlib import Path

import ollama

from scripts.retrieval_router import retrieve as retrieve_evidence
from scripts.build_grounded_prompt import build_grounded_prompt
from scripts.reliability_evaluation import evaluate_reliability
from scripts.adaptive_decision_controller import make_reliability_decision


# ============================================================
# Configuration
# ============================================================

LLM_MODEL = "llama3.2:latest"


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ADAPTIVE_CONTEXT_FILE = (
    BASE_DIR
    / "datasets"
    / "synthea"
    / "elderdocai"
    / "processed"
    / "adaptive_context.json"
)

ASSISTANCE_PLAN_FILE = (
    BASE_DIR
    / "datasets"
    / "synthea"
    / "elderdocai"
    / "processed"
    / "assistance_plans.json"
)

ASSISTANCE_DECISION_FILE = (
    BASE_DIR
    / "datasets"
    / "synthea"
    / "elderdocai"
    / "processed"
    / "assistance_decisions.json"
)


# ============================================================
# Adaptive Context Loading
# ============================================================

print("Loading ElderDocAI adaptive context...")

try:

    with open(
        ADAPTIVE_CONTEXT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        adaptive_context_records = json.load(f)

except FileNotFoundError:

    print(
        "WARNING: adaptive_context.json was not found."
    )

    adaptive_context_records = []


print(
    f"Adaptive context records loaded: "
    f"{len(adaptive_context_records)}"
)


# ============================================================
# Build Patient Index
# ============================================================

adaptive_context_by_patient = {}


for record in adaptive_context_records:

    patient_id = record.get("patient_id")

    if not patient_id:
        continue

    adaptive_context_by_patient.setdefault(
        str(patient_id),
        []
    ).append(record)


# Keep chronological ordering
for patient_id in adaptive_context_by_patient:

    adaptive_context_by_patient[patient_id].sort(
        key=lambda x: (
            x.get("window_start", ""),
            x.get("window_end", "")
        )
    )


print(
    f"Patients with adaptive context: "
    f"{len(adaptive_context_by_patient)}"
)


# ============================================================
# Assistance Plan + Decision Loading
# ============================================================
# The validated assistance-plan and assistance-decision layers describe
# the response behavior (strategy, actions, safety constraints). They are
# loaded alongside adaptive context and joined by
# (patient_id, window_start, window_end) so the RAG layer honors them.

print("Loading ElderDocAI assistance decisions and plans...")

try:
    with open(
        ASSISTANCE_DECISION_FILE,
        "r",
        encoding="utf-8"
    ) as f_dec:

        assistance_decision_records = json.load(f_dec)

except FileNotFoundError:

    print(
        "WARNING: assistance_decisions.json was not found."
    )

    assistance_decision_records = []

try:
    with open(
        ASSISTANCE_PLAN_FILE,
        "r",
        encoding="utf-8"
    ) as f_plan:

        assistance_plan_records = json.load(f_plan)

except FileNotFoundError:

    print(
        "WARNING: assistance_plans.json was not found."
    )

    assistance_plan_records = []

# ------------------------------------------------------------------
# Window-keyed indexes for plans and decisions
# ------------------------------------------------------------------
assistance_plan_by_window = {}     # (pid, window_start, window_end) -> plan
assistance_decision_by_window = {} # (pid, window_start, window_end) -> decision

for record in assistance_plan_records:

    patient_id = record.get("patient_id")
    window_start = record.get("window_start")
    window_end = record.get("window_end")

    if not patient_id or window_start is None or window_end is None:
        continue

    assistance_plan_by_window[
        (str(patient_id), window_start, window_end)
    ] = record

for record in assistance_decision_records:

    patient_id = record.get("patient_id")
    window_start = record.get("window_start")
    window_end = record.get("window_end")

    if not patient_id or window_start is None or window_end is None:
        continue

    assistance_decision_by_window[
        (str(patient_id), window_start, window_end)
    ] = record

print(
    f"Assistance plans loaded: "
    f"{len(assistance_plan_by_window)}"
)

print(
    f"Assistance decisions loaded: "
    f"{len(assistance_decision_by_window)}"
)


# ============================================================
# Adaptive Context Lookup
# ============================================================

def get_adaptive_context(
    patient_id=None,
    context_date=None
):
    """
    Retrieve the appropriate adaptive-context window.

    Selection priority:

    1. Exact window containing context_date
    2. Latest non-DATA_GAP window
    3. Latest available window
    """

    if not patient_id:
        return None

    records = adaptive_context_by_patient.get(
        str(patient_id),
        []
    )

    if not records:
        return None

    # --------------------------------------------------------
    # Exact date-based lookup
    # --------------------------------------------------------

    if context_date:

        context_date = str(context_date)

        for record in records:

            start = record.get(
                "window_start",
                ""
            )

            end = record.get(
                "window_end",
                ""
            )

            if start <= context_date <= end:
                return record

    # --------------------------------------------------------
    # Otherwise use latest window containing data
    # --------------------------------------------------------

    usable = [
        record
        for record in records
        if record.get("context_status") != "DATA_GAP"
    ]

    if usable:
        return usable[-1]

    # --------------------------------------------------------
    # Last resort
    # --------------------------------------------------------

    return records[-1]


# ============================================================
# Assistance Plan + Decision Lookup
# ============================================================

def get_assistance_plan(
    patient_id=None,
    window_start=None,
    window_end=None
):
    """
    Retrieve the validated assistance-plan record for an exact
    (patient_id, window_start, window_end) window.

    Returns None when no matching plan exists.
    """
    if not patient_id or window_start is None or window_end is None:
        return None

    return assistance_plan_by_window.get(
        (str(patient_id), window_start, window_end)
    )


def get_assistance_decision(
    patient_id=None,
    window_start=None,
    window_end=None
):
    """
    Retrieve the validated assistance-decision record for an exact
    (patient_id, window_start, window_end) window.

    Returns None when no matching decision exists.
    """
    if not patient_id or window_start is None or window_end is None:
        return None

    return assistance_decision_by_window.get(
        (str(patient_id), window_start, window_end)
    )


def prepare_assistance_plan(plan):
    """
    Convert an internal assistance-plan record into the compact shape
    consumed by the grounded-prompt builder.

    Returns an empty dict when no plan is available so the prompt
    builder can fall back gracefully.
    """
    if not plan:
        return {}

    return {
        "assistance_strategy": plan.get(
            "assistance_strategy"
        ),
        "priority": plan.get(
            "priority"
        ),
        "actions": plan.get(
            "actions",
            []
        ),
        "safety_constraints": plan.get(
            "safety_constraints",
            []
        ),
    }


# ============================================================
# Extract Patient ID
# ============================================================

def extract_patient_id(user_profile):
    """
    Extract patient identifier from either:

    - a dictionary
    - a Pydantic model
    - another object exposing the relevant attribute

    This is important because the FastAPI layer passes a
    Pydantic UserProfile object, while direct Python tests
    may pass a dictionary.
    """

    if not user_profile:
        return None

    possible_fields = [
        "patient_id",
        "user_id",
        "id"
    ]

    for field in possible_fields:

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(user_profile, dict):

            value = user_profile.get(field)

        # ----------------------------------------------------
        # Pydantic model / object
        # ----------------------------------------------------

        else:

            value = getattr(
                user_profile,
                field,
                None
            )

        if value is not None:
            return str(value)

    return None


# ============================================================
# Prepare Adaptive Context
# ============================================================

def prepare_adaptive_context(context):

    if not context:

        return {
            "available": False
        }

    care_state = (
        context.get("care_state")
        or {}
    )

    transition = (
        context.get("transition")
        or {}
    )

    assistance = (
        context.get("adaptive_assistance")
        or {}
    )

    return {

        "available": True,

        "context_status":
            context.get(
                "context_status"
            ),

        "window_start":
            context.get(
                "window_start"
            ),

        "window_end":
            context.get(
                "window_end"
            ),

        "care_state": {

            "state":
                care_state.get(
                    "state"
                ),

            "overall_score":
                care_state.get(
                    "overall_score"
                ),

            "has_documented_activity":
                care_state.get(
                    "has_documented_activity"
                ),

            "event_summary":
                care_state.get(
                    "event_summary"
                )

        },

        "changed_dimensions":
            context.get(
                "changed_dimensions",
                []
            ),

        "transition": {

            "type":
                transition.get(
                    "type"
                ),

            "direction":
                transition.get(
                    "direction"
                ),

            "magnitude":
                transition.get(
                    "magnitude"
                ),

            "score_delta":
                transition.get(
                    "score_delta"
                ),

            "previous_state":
                transition.get(
                    "previous_state"
                ),

            "current_state":
                transition.get(
                    "current_state"
                ),

            "supporting_evidence":
                transition.get(
                    "supporting_evidence"
                )

        },

        "adaptive_assistance": {

            "mode":
                assistance.get(
                    "mode"
                ),

            "priority":
                assistance.get(
                    "priority"
                ),

            "reason_codes":
                assistance.get(
                    "reason_codes",
                    []
                ),

            "reasons":
                assistance.get(
                    "reasons",
                    []
                )

        },

        "interpretation":
            context.get(
                "interpretation",
                (
                    "Adaptive assistance is selected from "
                    "documented care-state and transition "
                    "information. It does not represent a "
                    "diagnosis or medical risk prediction."
                )
            )

    }


# ============================================================
# Evidence Preparation
# ============================================================

def _semantic_similarity_of(chunk):
    """Dense-cosine semantic relevance for the reliability evaluator.

    Returns a cosine-similarity value in [-1, 1] (the reliability 'relevance'
    factor normalizes this with (score + 1) / 2).

    Score-semantics distinction (Task 2):
      - CrossEncoder rerank logits are UNBOUNDED ranking signals and are
        NEVER interpreted as cosine similarity here.
      - Dense BGE cosine similarity (FAISS inner product on L2-normalized
        embeddings) is the semantic relevance value.
      - If no dense cosine is available, an already-computed similarity field
        (native cosine) is used as a fallback.
      - Tiny floating-point drift outside [-1, 1] is clamped explicitly.
    """
    raw = None
    if "dense_score" in chunk and chunk.get("dense_score") is not None:
        raw = float(chunk["dense_score"])
    elif ("similarity_score" in chunk
          and chunk.get("similarity_score") is not None):
        raw = float(chunk["similarity_score"])
    if raw is None:
        return 0.0
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return raw


def prepare_evidence(chunks):
    """
    Convert hybrid-retrieval results into the evidence format required by
    the reliability evaluator and prompt builder.

    Score semantics (Task 2 - reliability relevance normalization):
      - ``similarity_score`` = DENSE cosine semantic similarity ([-1, 1]);
        consumed by the reliability 'relevance' factor. It is NEVER a
        CrossEncoder logit.
      - ``retrieval_score``  = CrossEncoder rerank logit (or the strongest
        available ranking signal); RANKING ONLY and NOT consumed by
        reliability.
      The retrieval/reranking ORDER of ``chunks`` is preserved verbatim.
    """

    evidence_items = []

    for chunk in chunks:

        evidence_items.append(

            {
                "chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "source_document":
                    chunk.get(
                        "source_document",
                        "Unknown"
                    ),

                "document_category":
                    chunk.get(
                        "category",
                        chunk.get(
                            "document_category",
                            "Unknown"
                        )
                    ),

                "authority_score":
                    chunk.get(
                        "authority_score",
                        1.0
                    ),

                "similarity_score":
                    _semantic_similarity_of(
                        chunk
                    ),

                "retrieval_score":
                    chunk.get(
                        "rerank_score",
                        chunk.get(
                            "hybrid_score"
                        )
                    ),

                "text":
                    chunk.get(
                        "text",
                        ""
                    )
            }
        )

    return evidence_items


# ============================================================
# Reliability Report Printer
# ============================================================

def print_reliability_report(
    reliability,
    decision
):

    print("\n")
    print("-" * 70)
    print("RELIABILITY REPORT")
    print("-" * 70)

    print(
        f"  Authority:           "
        f"{reliability['authority']:.2f}"
    )

    print(
        f"  Relevance:           "
        f"{reliability['relevance']:.2f}"
    )

    print(
        f"  Support:             "
        f"{reliability['support']:.2f}"
    )

    print(
        f"  Coverage:            "
        f"{reliability['coverage']:.2f}"
    )

    print(
        f"  Consistency:         "
        f"{reliability['consistency']:.2f}"
    )

    print(
        f"  Overall Reliability: "
        f"{reliability['overall_reliability']:.2f}"
    )

    print(
        f"  Decision:            "
        f"{decision['decision']}"
    )

    print(
        f"  Reason:              "
        f"{decision['reason']}"
    )

    print("-" * 70)


# ============================================================
# Generate Answer
# ============================================================

MAX_REFINE_ATTEMPTS = 1
MAX_RETRIEVE_ATTEMPTS = 1
MIN_SUPPORT_TERMS = 2          # mirrors the reliability support factor

REJECTION_RESPONSE = (
    "I couldn't find information reliable enough to answer that "
    "question. Please ask again with a little more detail, or ask "
    "a doctor or a trusted caregiver."
)

EMPTY_EVIDENCE_RESPONSE = (
    "I couldn't find that information "
    "in the knowledge base."
)



# ============================================================
# Generation System Prompt (hoisted unchanged)
# ============================================================

GENERATION_SYSTEM_PROMPT = """
You are CareBuddy,
an evidence-grounded elderly-care assistant.

You have two information sources:

1. ADAPTIVE CONTEXT
   - Describes documented care activity,
     temporal changes, and selected assistance strategy.
   - It does NOT diagnose disease.
   - It does NOT predict medical risk.
   - It must not be treated as a medical diagnosis.

2. RETRIEVED KNOWLEDGE
   - Provides evidence from the CareBuddy knowledge base.
   - Answers must be grounded in this evidence.

STRICT RULES:

1. Use only the provided adaptive context and retrieved evidence.

2. Do not use outside medical knowledge.

3. Do not guess.

4. Do not invent facts about the patient.

5. Do not diagnose.

6. Do not predict disease progression,
   medical risk, hospitalization, or mortality.

7. If the retrieved evidence is insufficient
   to answer the user's question, say:

   I couldn't find that information in the knowledge base.

8. Use the adaptive context to personalize
   the structure and relevance of the response,
   but never turn care-state labels into diagnoses.

9. Keep answers short, clear, and factual.

10. When adaptive assistance is available,
    follow its assistance mode and priority
    while remaining grounded in retrieved evidence.

11. Do not include a Sources section in the
    user-facing answer.

12. Do not mention retrieval, evidence scores,
    reliability scores, internal decisions,
    care-state computation, or system architecture
    unless explicitly asked.

Return only the answer itself.
"""
def _refine_evidence(evidence_items, query):
    """Controlled evidence-preparation refinement: keep only items sharing
    at least MIN_SUPPORT_TERMS content terms with the query (the same
    threshold used by the reliability support factor). Ordering and identity
    of kept items are preserved. Retrieval and scoring are unchanged."""
    from scripts.reliability_evaluation import _content_terms

    query_terms = _content_terms(query or "")
    if not query_terms:
        return list(evidence_items)

    kept = [
        item
        for item in evidence_items
        if len(query_terms & _content_terms((item.get("text") or "")))
        >= MIN_SUPPORT_TERMS
    ]
    return kept
# ============================================================
# Gated Answer Flow
# ============================================================

def _run_gated_generation(
    query,
    user_profile=None,
    conversation_context="",
    response_language="en",
    assistance_plan=None,
    system_message="",
):
    """Reliability-gated answer flow.

    ACCEPT -> grounded generation on the accepted evidence.
    REFINE -> one controlled evidence refinement, re-evaluate, then generate
              from the once-refined evidence (LLM called only if evidence left).
    RE-RETRIEVE -> one controlled additional retrieval, then re-evaluate.
    REJECT -> no LLM call; controlled refusal.

    Returns dict: answer, evidence_items, reliability, decision,
    retrieval_attempts, refinement_attempts, refused, message.
    """
    retrieval_attempts = 0
    refinement_attempts = 0
    refused = False
    why_refused = ""

    retrieved_chunks = retrieve_evidence(query)
    retrieval_attempts += 1

    if not retrieved_chunks:
        reliability = evaluate_reliability(query=query, evidence_items=[])
        decision = make_reliability_decision(reliability)
        return {
            "answer": EMPTY_EVIDENCE_RESPONSE,
            "evidence_items": [],
            "reliability": reliability,
            "decision": decision,
            "retrieval_attempts": retrieval_attempts,
            "refinement_attempts": refinement_attempts,
            "refused": True,
            "message": "EMPTY_EVIDENCE",
        }

    evidence_items = prepare_evidence(retrieved_chunks)
    reliability = evaluate_reliability(
        query=query,
        evidence_items=evidence_items,
    )
    decision = make_reliability_decision(reliability)

    guard = 0
    max_guard = 8
    while guard < max_guard:
        guard += 1
        label = decision["decision"]

        if label == "ACCEPT":
            break

        if label == "REJECT":
            refused = True
            why_refused = "REJECT"
            break

        if label == "REFINE" and refinement_attempts < MAX_REFINE_ATTEMPTS:
            refinement_attempts += 1
            refined = _refine_evidence(evidence_items, query)
            if not refined:
                refused = True
                why_refused = "REFINE_EMPTY"
                evidence_items = []
            else:
                evidence_items = refined
            reliability = evaluate_reliability(
                query=query,
                evidence_items=evidence_items,
            )
            decision = make_reliability_decision(reliability)
            continue

        if (
            label == "RE-RETRIEVE"
            and retrieval_attempts <= MAX_RETRIEVE_ATTEMPTS
        ):
            retrieval_attempts += 1
            re_chunks = retrieve_evidence(query)
            if not re_chunks:
                refused = True
                why_refused = "RERETRIEVE_EMPTY"
                break
            evidence_items = prepare_evidence(re_chunks)
            reliability = evaluate_reliability(
                query=query,
                evidence_items=evidence_items,
            )
            decision = make_reliability_decision(reliability)
            continue

        # REFINE with refinement budget exhausted: generate from the
        # once-refined evidence, provided evidence remains.
        if label == "REFINE" and evidence_items:
            break

        # RE-RETRIEVE budget exhausted (or unrecognized label): never
        # silently generate from unacceptable evidence.
        refused = True
        why_refused = label or "UNKNOWN"
        break
# ------------------------------------------------ refusal boundary
    if refused:
        if not evidence_items:
            answer_text = EMPTY_EVIDENCE_RESPONSE
        else:
            answer_text = REJECTION_RESPONSE
        return {
            "answer": answer_text,
            "evidence_items": evidence_items,
            "reliability": reliability,
            "decision": decision,
            "retrieval_attempts": retrieval_attempts,
            "refinement_attempts": refinement_attempts,
            "refused": True,
            "message": why_refused,
        }

    # ------------------------------------------------ grounded generation
    prompt = build_grounded_prompt(
        query=query,
        evidence_items=evidence_items,
        reliability=reliability,
        decision=decision,
        user_profile=user_profile,
        conversation_context=conversation_context,
        response_language=response_language,
        assistance_plan=assistance_plan or {},
    )

    response = ollama.chat(
        model=LLM_MODEL,
        options={
            "temperature": 0,
            "top_p": 0.1,
            "top_k": 10,
        },
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer_text = (
        response["message"]["content"]
        .strip()
    )

    return {
        "answer": answer_text,
        "evidence_items": evidence_items,
        "reliability": reliability,
        "decision": decision,
        "retrieval_attempts": retrieval_attempts,
        "refinement_attempts": refinement_attempts,
        "refused": False,
        "message": "generated",
    }
def generate_answer(
    query,
    user_profile=None,
    conversation_context="",
    response_language="en",
    return_evidence=False,
    return_evaluation=False
):

    print("\n" + "=" * 70)
    print("ELDERDOCAI / CAREBUDDY RAG")
    print("=" * 70)

    # ========================================================
    # Patient / Adaptive Context
    # ========================================================

    patient_id = extract_patient_id(
        user_profile
    )

    adaptive_context = get_adaptive_context(
        patient_id=patient_id
    )

    adaptive_context_for_prompt = (
        prepare_adaptive_context(
            adaptive_context
        )
    )

    assistance_plan = None
    assistance_plan_for_prompt = None

    if adaptive_context:

        assistance_plan = get_assistance_plan(
            patient_id=patient_id,
            window_start=adaptive_context.get("window_start"),
            window_end=adaptive_context.get("window_end")
        )

        assistance_plan_for_prompt = (
            prepare_assistance_plan(
                assistance_plan
            )
        )

    print("\nAdaptive Context:")

    if adaptive_context:
        print("  patient_id:", patient_id)
        print("  window:", adaptive_context.get("window_start"),
              "to", adaptive_context.get("window_end"))
        care_state = adaptive_context.get("care_state") or {}
        print("  care_state:", care_state.get("state"))
        print("  score:", care_state.get("overall_score"))
        assistance = adaptive_context.get("adaptive_assistance") or {}
        print("  assistance:", assistance.get("mode"))
        print("  priority:", assistance.get("priority"))
        if assistance_plan_for_prompt:
            print("  assistance_strategy:",
                  assistance_plan_for_prompt.get("assistance_strategy"))
            print("  assistance actions:",
                  [a.get("action")
                   for a in assistance_plan_for_prompt.get("actions", [])])
    else:
        print("  No adaptive context available.")

    # ========================================================
    # Reliability-Gated Execution
    # ========================================================

    print("\nRetrieving evidence...\n")

    result = _run_gated_generation(
        query,
        user_profile=user_profile,
        conversation_context=conversation_context,
        response_language=response_language,
        assistance_plan=assistance_plan_for_prompt,
        system_message=GENERATION_SYSTEM_PROMPT,
    )

    answer = result["answer"]
    evidence_items = result["evidence_items"]
    reliability = result["reliability"]
    decision = result["decision"]
    gate_retrieval_attempts = result["retrieval_attempts"]
    gate_refinement_attempts = result["refinement_attempts"]
    gate_refused = result["refused"]

    # ========================================================
    # Reliability Report
    # ========================================================

    print_reliability_report(
        reliability,
        decision
    )

    # ========================================================
    # Extract Answer (post-processing, unchanged)
    # ========================================================

    if "\nSources:" in answer:

        answer = answer.split(
            "\nSources:",
            1
        )[0].strip()

    elif answer.startswith("Sources:"):

        answer = ""

    if answer.startswith("Answer:"):

        answer = answer[
            len("Answer:"):
        ].strip()

    # ========================================================
    # Return
    # ========================================================

    if return_evaluation:

        return {
            "answer": answer,
            "evidence_items": evidence_items,
            "reliability": reliability,
            "decision": decision,
            "retrieval_attempts": gate_retrieval_attempts,
            "refinement_attempts": gate_refinement_attempts,
            "refused": gate_refused,
        }

    if return_evidence:

        return (
            answer,
            evidence_items
        )

    return answer
