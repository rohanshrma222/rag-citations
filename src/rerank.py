"""Phase 6: Cross-encoder reranking — rescore hybrid candidates, keep the true top-k.

The retrievers in Phase 5 never see a query and a chunk together. A bi-encoder
embeds each independently, so similarity is whatever survives being compressed into
one vector per side; BM25 only counts term overlap. A cross-encoder instead runs
the query and the passage through one transformer as a single input, so attention
spans both, and it can judge whether the passage actually answers the question
rather than whether it merely looks topically similar.

The cost is that it cannot be precomputed: every (query, chunk) pair needs a forward
pass at query time. That is why it reranks ~20 candidates rather than the corpus,
and why Phase 5 produces exactly that many.

Scores are raw logits, not probabilities — unbounded, roughly -11..+11 for this
model, and only comparable *within* one query's candidate set. Never threshold them
against a fixed number across different queries without calibrating first.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import search
import store

MODEL_NAME = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = 5            # what generation in Phase 7 will actually see
BATCH_SIZE = 32      # pairs per forward pass
MAX_LENGTH = 512     # model limit; Phase 2 chunks (~230 tokens) fit comfortably

_model = None        # loaded once, lazily — construction costs ~30s cold


def get_model():
    """Load the cross-encoder once per process."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int | None = TOP_K,
           model=None) -> list[dict]:
    """Rescore candidates with the cross-encoder and return them best-first.

    Each returned chunk gains `rerank_score` (a raw logit) and `prior_rank`, its
    1-indexed position before reranking, so rank movement stays inspectable.
    Returns all candidates when top_k is None.
    """
    if not candidates:
        return []

    model = model or get_model()
    pairs = [(query, chunk["text"]) for chunk in candidates]
    scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

    scored = [
        {**chunk, "rerank_score": float(score), "prior_rank": i}
        for i, (chunk, score) in enumerate(zip(candidates, scores), start=1)
    ]
    # Tie-break on the prior rank so equal scores preserve upstream ordering.
    scored.sort(key=lambda hit: (-hit["rerank_score"], hit["prior_rank"]))
    return scored if top_k is None else scored[:top_k]


def search_and_rerank(conn, query: str, top_k: int = TOP_K,
                      candidates: int = search.CANDIDATES) -> list[dict]:
    """The full Phase 5 + 6 retrieval path: hybrid search, then rerank."""
    fused = search.hybrid_search(conn, query, k=candidates, candidates=candidates)
    return rerank(query, fused, top_k=top_k)


# --------------------------------------------------------------------------- #
# inspection
# --------------------------------------------------------------------------- #

def _preview(hit: dict, width: int = 52) -> str:
    return " ".join(hit["text"].split())[:width]


def compare(conn, query: str, top_k: int = TOP_K,
            candidates: int = search.CANDIDATES) -> dict:
    """Show hybrid ordering against reranked ordering, with timings."""
    # Load the model BEFORE timing anything: a cold CrossEncoder costs ~10-30s to
    # construct, which otherwise lands inside the rerank measurement and reports a
    # per-pair cost roughly 50x the real one.
    model = get_model()

    t0 = time.perf_counter()
    fused = search.hybrid_search(conn, query, k=candidates, candidates=candidates)
    t_retrieve = time.perf_counter() - t0

    t0 = time.perf_counter()
    reranked = rerank(query, fused, top_k=None, model=model)
    t_rerank = time.perf_counter() - t0

    print(f"\n{'=' * 78}\nQUERY: {query!r}")
    print(f"  retrieve {t_retrieve * 1000:.0f} ms | "
          f"rerank {len(fused)} candidates {t_rerank * 1000:.0f} ms "
          f"({t_rerank / max(len(fused), 1) * 1000:.1f} ms/pair)")

    print(f"\n  hybrid top-{top_k} (RRF)")
    for i, hit in enumerate(fused[:top_k], 1):
        print(f"    {i}. {hit['rrf_score']:.5f}  p{hit['page_number']:<3} {_preview(hit)}")

    print(f"\n  reranked top-{top_k} (cross-encoder logit)")
    for i, hit in enumerate(reranked[:top_k], 1):
        move = hit["prior_rank"] - i
        arrow = f"was #{hit['prior_rank']}" + (f", {move:+d}" if move else ", same")
        print(f"    {i}. {hit['rerank_score']:+8.3f}  p{hit['page_number']:<3} "
              f"{_preview(hit)}  [{arrow}]")

    kept = {hit["rowid"] for hit in fused[:top_k]} & {hit["rowid"] for hit in reranked[:top_k]}
    new_top = reranked[0]["prior_rank"]
    print(f"\n  top-{top_k} overlap: {len(kept)}/{top_k} | "
          f"new #1 was hybrid #{new_top}")
    return {"hybrid": fused, "reranked": reranked,
            "retrieve_s": t_retrieve, "rerank_s": t_rerank,
            "promoted": new_top}


def movers(conn, queries: list[str], candidates: int = search.CANDIDATES) -> None:
    """Find queries where reranking actually changes the answer.

    The roadmap's verification asks for a case where hybrid top-1 is *not* the
    best chunk and reranking fixes it. Rather than guess, score a batch of
    queries and report how far the new #1 was promoted from.
    """
    model = get_model()
    rows = []
    for query in queries:
        fused = search.hybrid_search(conn, query, k=candidates, candidates=candidates)
        reranked = rerank(query, fused, top_k=None, model=model)
        if not reranked:
            continue
        rows.append({
            "query": query,
            "promoted_from": reranked[0]["prior_rank"],
            "top_page": reranked[0]["page_number"],
            "was_page": fused[0]["page_number"],
            "score": reranked[0]["rerank_score"],
        })

    rows.sort(key=lambda row: -row["promoted_from"])
    print(f"\n  {'promoted':>8}  {'page':>9}  score     query")
    for row in rows:
        moved = "" if row["promoted_from"] == 1 else "  <-- rerank changed the answer"
        print(f"  #{row['promoted_from']:>2} -> #1  p{row['was_page']}->p{row['top_page']:<5} "
              f"{row['score']:+7.2f}  {row['query'][:42]!r}{moved}")

    changed = sum(1 for row in rows if row["promoted_from"] != 1)
    print(f"\n  reranking changed the top answer for {changed}/{len(rows)} queries")


def latency(conn, query: str = "What is multi-head attention?") -> None:
    """Measure the rerank step against candidate count — the roadmap's budget check."""
    model = get_model()
    pool = search.hybrid_search(conn, query, k=50, candidates=50)
    print(f"\n  candidate pool available: {len(pool)}")
    print(f"  {'candidates':>10}  {'total':>9}  {'per pair':>9}")
    for n in (5, 10, 20, 30, 50):
        subset = pool[:n]
        if len(subset) < n:
            continue
        t0 = time.perf_counter()
        rerank(query, subset, top_k=None, model=model)
        elapsed = time.perf_counter() - t0
        print(f"  {n:>10}  {elapsed * 1000:>7.0f}ms  {elapsed / n * 1000:>7.1f}ms")
    print("  budget: the roadmap wants well under 1s for ~20 candidates")


def selftest() -> bool:
    """Ordering, metadata and batching logic, with a stub model. No download."""
    results = []

    class StubModel:
        """Returns a fixed score per passage text, and records batch sizes."""
        def __init__(self, table):
            self.table = table
            self.calls = []

        def predict(self, pairs, batch_size=None, show_progress_bar=False):
            self.calls.append({"n": len(pairs), "batch_size": batch_size})
            return [self.table[text] for _, text in pairs]

    # Candidates deliberately in the WRONG order: the best passage is last.
    candidates = [
        {"rowid": 1, "text": "irrelevant filler about optimisers", "page_number": 9},
        {"rowid": 2, "text": "partially relevant mention of attention", "page_number": 4},
        {"rowid": 3, "text": "the actual answer about multi-head attention", "page_number": 5},
    ]
    stub = StubModel({
        "irrelevant filler about optimisers": -8.0,
        "partially relevant mention of attention": 1.5,
        "the actual answer about multi-head attention": 9.0,
    })

    out = rerank("q", candidates, top_k=None, model=stub)

    results.append(("reorders worst-first input", [hit["rowid"] for hit in out] == [3, 2, 1]))
    results.append(("attaches rerank_score", out[0]["rerank_score"] == 9.0))
    results.append(("records prior_rank", out[0]["prior_rank"] == 3 and out[2]["prior_rank"] == 1))
    results.append(("preserves chunk metadata", out[0]["page_number"] == 5))
    results.append(("scores sorted desc",
                    [hit["rerank_score"] for hit in out]
                    == sorted((hit["rerank_score"] for hit in out), reverse=True)))
    results.append(("one forward pass", len(stub.calls) == 1))
    results.append(("passes batch_size", stub.calls[0]["batch_size"] == BATCH_SIZE))

    # top_k truncates after sorting, not before
    top1 = rerank("q", candidates, top_k=1, model=stub)
    results.append(("top_k truncates post-sort", len(top1) == 1 and top1[0]["rowid"] == 3))

    # empty input must not touch the model
    stub.calls.clear()
    results.append(("empty input", rerank("q", [], model=stub) == []))
    results.append(("empty skips model", stub.calls == []))

    # ties fall back to upstream order
    tied = [{"rowid": 7, "text": "a", "page_number": 1},
            {"rowid": 4, "text": "b", "page_number": 2}]
    tie_out = rerank("q", tied, top_k=None, model=StubModel({"a": 2.0, "b": 2.0}))
    results.append(("ties keep prior order", [hit["rowid"] for hit in tie_out] == [7, 4]))

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


# A spread of query shapes: factual lookup, paraphrase, rare token, and a couple
# whose best chunk is unlikely to be hybrid #1.
PROBES = [
    "What is multi-head attention?",
    "how does the model know word order without recurrence?",
    "WMT 2014 English-German BLEU score",
    "why divide by the square root of the key dimension?",
    "what optimiser and learning rate schedule were used?",
    "how many GPUs were used for training and for how long?",
    "what is the computational complexity of self-attention per layer?",
    "what regularisation techniques were applied?",
    "how are the encoder and decoder stacks connected?",
    "what does label smoothing do to perplexity?",
]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if "--selftest" in sys.argv:
        print("Phase 6 self-test (stub model, no download)\n")
        sys.exit(0 if selftest() else 1)

    conn = store.connect()
    store.init_schema(conn)
    if store.stats(conn)["chunks"] == 0:
        print("index is empty — run `python src/ingest.py` first")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--latency" in sys.argv:
        latency(conn)
    elif "--movers" in sys.argv:
        print("Does reranking change the answer?")
        movers(conn, PROBES)
    elif args:
        compare(conn, " ".join(args))
    else:
        for query in PROBES[:4]:
            compare(conn, query)
    conn.close()
