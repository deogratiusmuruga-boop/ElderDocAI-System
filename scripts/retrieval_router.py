#!/usr/bin/env python3
"""ElderDocAI Retrieval Router - Phase 3 Task 10B.

Minimal backend-selection layer between rag_chat.py and the retrieval
implementations.

Backends (env ELDERDOCAI_RETRIEVAL_BACKEND):
  legacy   -> scripts.hybrid_retriever.hybrid_search()         [DEFAULT]
  geri_lit -> GeriLitRetriever.retrieve() + geri_lit_adapter
  both     -> legacy-only (compatibility behavior defined by Task 10A;
              true multi-source fusion is NOT implemented in Task 10B)

Behavior guarantees:
  - deterministic default (legacy)
  - legacy path is byte-identical unless geri_lit is explicitly selected
  - no existing retrieval logic modified
  - frozen GeriLit retriever/artifacts read-only
  - safe failure / fallback to legacy
  - model reuse via a module-level singleton (no reload per query)
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BACKEND_ENV = "ELDERDOCAI_RETRIEVAL_BACKEND"
DEFAULT_BACKEND = "legacy"
SUPPORTED = ("legacy", "geri_lit", "both")

# module-level singleton to avoid reloading models/indexes per query
_geri_lit_retriever_singleton = None
_logged = False


def _log(event, message):
    print("[retrieval_router] %s: %s" % (event, message))


def _enabled_backend(backend=None):
    """Resolve the effective backend deterministically.

    - explicit argument wins
    - else env var
    - else DEFAULT_BACKEND
    - unknown values => DEFAULT_BACKEND (safe)
    """
    if backend is None:
        backend = os.getenv(BACKEND_ENV, DEFAULT_BACKEND)
    backend = str(backend).strip().lower()
    if backend not in SUPPORTED:
        _log("warn", "unknown backend %r; using default %r"
             % (backend, DEFAULT_BACKEND))
        return DEFAULT_BACKEND
    return backend


def _legacy_retrieve(query, **kwargs):
    """Execute the original legacy hybrid retriever."""
    from scripts.hybrid_retriever import hybrid_search
    return hybrid_search(query)


def _geri_lit(singleton=True):
    """Get (and optionally cache) the GeriLit retriever instance."""
    global _geri_lit_retriever_singleton
    if singleton and _geri_lit_retriever_singleton is not None:
        return _geri_lit_retriever_singleton
    # ensure offline, then add the geri_lit package dir to sys.path
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    geri_dir = str(BASE_DIR / "datasets" / "elderdocai_geriLit")
    if geri_dir not in sys.path:
        sys.path.insert(0, geri_dir)
    from geri_lit_retriever import GeriLitRetriever  # noqa: E402
    instance = GeriLitRetriever(base_dir=geri_dir, offline=True)
    if singleton:
        _geri_lit_retriever_singleton = instance
    return instance


def _geri_lit_retrieve(query, fallback=True, **kwargs):
    """Run GeriLit retrieval and adapt to the legacy evidence shape.

    On failure: log and (optionally) fall back to legacy retrieval, matching
    Task 10A backward-compatibility requirements. Never fabricate evidence.
    """
    global _logged
    try:
        retriever = _geri_lit()
        raw = retriever.retrieve(query)
        from scripts.geri_lit_adapter import to_legacy_evidence
        return to_legacy_evidence(raw)
    except Exception as exc:  # noqa: BLE001 - safe integration boundary
        _log("error", "geri_lit retrieval failed: %s" % repr(exc))
        if fallback:
            _log("fallback", "using legacy retrieval")
            return _legacy_retrieve(query)
        return []


def retrieve(query, backend=None, fallback=True, **kwargs):
    """Select and execute the retrieval backend for a query.

    Returns a list of chunk-like dicts in the legacy evidence shape
    (GeriLit output is adapted by geri_lit_adapter). Empty list is a valid
    safe result (downstream REJECT/empty-evidence path).
    """
    eff = _enabled_backend(backend)
    if eff == "legacy":
        return _legacy_retrieve(query, **kwargs)
    if eff == "geri_lit":
        return _geri_lit_retrieve(query, fallback=fallback, **kwargs)
    if eff == "both":
        # Task 10B compatibility: "both" uses legacy-only (no fusion).
        _log("both", "multi-source fusion not implemented; using legacy")
        return _legacy_retrieve(query, **kwargs)
    return _legacy_retrieve(query, **kwargs)


def reset_geri_lit_singleton():
    """Clear the singleton (used by tests)."""
    global _geri_lit_retriever_singleton
    _geri_lit_retriever_singleton = None


if __name__ == "__main__":
    import sys as _sys
    q = _sys.argv[1] if len(_sys.argv) > 1 else (
        "What factors are associated with falls in older adults?")
    print("backend env =", os.getenv(BACKEND_ENV, "(unset)"))
    res = retrieve(q)
    print("results:", len(res))
    for r in res[:3]:
        print(" -", r.get("chunk_id"), "|", r.get("source_document"),
              "| sim=%.4f" % float(r.get("similarity_score", 0.0)))