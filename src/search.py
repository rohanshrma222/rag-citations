"""Phase 5: Hybrid search — vector KNN + BM25 keyword, merged with Reciprocal Rank Fusion.

Two retrievers with complementary failure modes:

  vector   embeds the query and finds semantically near chunks. Survives
           paraphrase, but is weak on rare literal tokens (names, numbers,
           dataset labels) that carry little semantic signal.
  keyword  BM25 over FTS5. Nails rare literal tokens, but scores a term that
           appears in most documents near zero — on a single-topic corpus it is
           near-useless for that corpus's own central vocabulary.

Their scores are not comparable: vector distance is 0..2 (lower is better), BM25 is
an unbounded negative (lower is better, and its scale shifts with corpus
statistics). Normalising them against each other would mean inventing a conversion
that does not exist, so fusion happens on *ranks* instead — Reciprocal Rank Fusion:

    score(doc) = sum over lists of  1 / (RRF_K + rank(doc))

A document ranked highly by both retrievers beats one ranked highly by only one,
and no score scale is ever compared to another.

Query text cannot be passed to FTS5 MATCH unescaped. FTS5 parses bareword syntax,
so a hyphen reads as a column reference: "multi-head" raises
`OperationalError: no such column: head`. Since hyphenated terms are everywhere in
technical PDFs, `fts_query()` rewrites free text into a safe OR of quoted tokens.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import store
from embed import embed_query

RRF_K = 60          # the standard RRF constant; damps the top of each list
CANDIDATES = 20     # per retriever, before fusion — Phase 6 reranks this many

# Tokens worth sending to BM25: words and numbers. Everything else is punctuation
# that FTS5 would either choke on or ignore.
_TOKEN = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------------- #
# query preparation
# --------------------------------------------------------------------------- #

def fts_query(text: str) -> str:
    """Rewrite free text into a safe FTS5 MATCH expression.

    Each token is double-quoted (making it a literal string rather than FTS5
    syntax) and joined with OR, so recall is driven by BM25's ranking rather than
    by requiring every term to be present. Returns "" if nothing usable remains,
    which callers must treat as "no keyword hits" rather than passing it to MATCH.

        'What is multi-head attention?'  ->  '"What" OR "is" OR "multi" OR "head" OR "attention"'
    """
    tokens = _TOKEN.findall(text)
    return " OR ".join(f'"{t}"' for t in tokens)


# --------------------------------------------------------------------------- #
# the two retrievers
# --------------------------------------------------------------------------- #

def vector_search(conn, query: str, k: int = CANDIDATES) -> list[dict]:
    """Embed the query and return the k nearest chunks, best first."""
    return store.vector_search(conn, embed_query(query), k=k)


def keyword_search(conn, query: str, k: int = CANDIDATES) -> list[dict]:
    """BM25 search over FTS5, best first. Safe for arbitrary query text."""
    match = fts_query(query)
    if not match:
        return []
    return store.keyword_search(conn, match, k=k)


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #

def reciprocal_rank_fusion(ranked_lists: dict[str, list[dict]],
                           k: int = RRF_K,
                           key: str = "rowid") -> list[dict]:
    """Merge named ranked lists by Reciprocal Rank Fusion.

    `ranked_lists` maps a retriever name to its results, best first. Returns one
    merged list sorted by fused score, each entry carrying `rrf_score` and a
    `ranks` dict recording where each retriever placed it (1-indexed, absent if
    that retriever did not return it).
    """
    scores: dict = {}
    for name, results in ranked_lists.items():
        for rank, hit in enumerate(results, start=1):
            entry = scores.setdefault(
                hit[key], {**hit, "rrf_score": 0.0, "ranks": {}}
            )
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["ranks"][name] = rank

    merged = list(scores.values())
    # Tie-break on rowid so equal scores order deterministically run to run.
    merged.sort(key=lambda hit: (-hit["rrf_score"], hit[key]))
    return merged


def hybrid_search(conn, query: str, k: int = CANDIDATES,
                  candidates: int = CANDIDATES) -> list[dict]:
    """Run both retrievers and return the top k fused results."""
    lists = {
        "vector": vector_search(conn, query, k=candidates),
        "keyword": keyword_search(conn, query, k=candidates),
    }
    return reciprocal_rank_fusion(lists)[:k]


# --------------------------------------------------------------------------- #
# comparison harness
# --------------------------------------------------------------------------- #

def _preview(hit: dict, width: int = 62) -> str:
    return " ".join(hit["text"].split())[:width]


def compare(conn, query: str, k: int = 5) -> dict:
    """Print vector-only, keyword-only and hybrid top-k side by side."""
    vec = vector_search(conn, query, k=CANDIDATES)
    kw = keyword_search(conn, query, k=CANDIDATES)
    fused = reciprocal_rank_fusion({"vector": vec, "keyword": kw})

    print(f"\n{'=' * 78}\nQUERY: {query!r}")
    print(f"MATCH: {fts_query(query)[:70]}...")

    print(f"\n  vector-only (distance, lower is better)")
    for i, hit in enumerate(vec[:k], 1):
        print(f"    {i}. {hit['distance']:.4f}  p{hit['page_number']:<3} {_preview(hit)}")
    if not vec:
        print("    (none)")

    print(f"\n  keyword-only (bm25, lower is better)")
    for i, hit in enumerate(kw[:k], 1):
        print(f"    {i}. {hit['score']:7.3f}  p{hit['page_number']:<3} {_preview(hit)}")
    if not kw:
        print("    (none)")

    print(f"\n  hybrid (RRF)")
    for i, hit in enumerate(fused[:k], 1):
        where = " ".join(f"{n}#{r}" for n, r in sorted(hit["ranks"].items()))
        print(f"    {i}. {hit['rrf_score']:.5f}  p{hit['page_number']:<3} "
              f"{_preview(hit, 44)}  [{where}]")

    both = sum(1 for hit in fused[:k] if len(hit["ranks"]) == 2)
    print(f"\n  {both}/{min(k, len(fused))} of hybrid top-{k} were found by both retrievers")

    return {"vector": vec, "keyword": kw, "hybrid": fused}


def overlap_report(conn, queries: list[str],
                   sizes: tuple[int, ...] = (3, 5, 10, 20)) -> None:
    """How much the two retrievers disagree, per candidate-pool size.

    RRF can only reorder things if the two lists actually differ. If the pools
    overlap almost completely, hybrid search is an expensive way to reproduce
    whichever retriever you already had — so read this before concluding that
    fusion helped or did not. Low Jaccard means the retrievers are finding
    different chunks and fusion has real work to do.
    """
    total = store.stats(conn)["chunks"]
    for size in sizes:
        print()
        print(f"  candidates per retriever = {size} ({size / total:.0%} of {total} chunks)")
        for query in queries:
            vec = {hit["rowid"] for hit in vector_search(conn, query, k=size)}
            kw = {hit["rowid"] for hit in keyword_search(conn, query, k=size)}
            union = vec | kw
            jaccard = len(vec & kw) / len(union) if union else 0.0
            print(f"    overlap {len(vec & kw):>2}/{size:<2}  jaccard {jaccard:.2f}   "
                  f"{query[:44]!r}")


def selftest() -> bool:
    """RRF arithmetic and FTS5 escaping. No database, no Ollama."""
    results = []

    # --- escaping: the bug that motivated fts_query ---
    results.append(("hyphen split", fts_query("multi-head") == '"multi" OR "head"'))
    results.append(("punctuation dropped", fts_query("cost / benefit") == '"cost" OR "benefit"'))
    results.append(("question mark dropped", fts_query("what is attention?")
                    == '"what" OR "is" OR "attention"'))
    results.append(("numbers kept", fts_query("WMT 2014") == '"WMT" OR "2014"'))
    results.append(("empty is empty", fts_query("???") == ""))
    # A user's literal "OR"/"NEAR" must become a quoted token, not an operator,
    # and an embedded quote must not escape its own quoting.
    results.append(("operators neutralised",
                    fts_query('say OR NEAR x') == '"say" OR "OR" OR "NEAR" OR "x"'))
    results.append(("embedded quote neutralised", fts_query('a"b') == '"a" OR "b"'))
    results.append(("column syntax neutralised", fts_query('text:foo') == '"text" OR "foo"'))

    # --- RRF arithmetic against hand-computed values ---
    a = [{"rowid": 1}, {"rowid": 2}, {"rowid": 3}]
    b = [{"rowid": 3}, {"rowid": 2}, {"rowid": 9}]
    fused = reciprocal_rank_fusion({"x": a, "y": b})

    expect_2 = 1 / 62 + 1 / 62          # rank 2 in both lists
    expect_1 = 1 / 61                   # rank 1 in x only
    expect_3 = 1 / 63 + 1 / 61          # rank 3 in x, rank 1 in y
    by_id = {hit["rowid"]: hit for hit in fused}

    results.append(("rrf one list", abs(by_id[1]["rrf_score"] - expect_1) < 1e-12))
    results.append(("rrf both lists", abs(by_id[2]["rrf_score"] - expect_2) < 1e-12))
    results.append(("rrf mixed ranks", abs(by_id[3]["rrf_score"] - expect_3) < 1e-12))
    results.append(("rrf union", {hit["rowid"] for hit in fused} == {1, 2, 3, 9}))
    results.append(("rrf sorted desc",
                    [hit["rrf_score"] for hit in fused]
                    == sorted((hit["rrf_score"] for hit in fused), reverse=True)))
    results.append(("rrf top is doc 3", fused[0]["rowid"] == 3))
    results.append(("ranks recorded", by_id[2]["ranks"] == {"x": 2, "y": 2}
                    and by_id[1]["ranks"] == {"x": 1}))

    # agreement beats a single strong placement: doc 3 (ranks 3 and 1) outranks
    # doc 1 (rank 1 in one list only)
    results.append(("agreement wins", by_id[3]["rrf_score"] > by_id[1]["rrf_score"]))

    # determinism on ties
    tied = reciprocal_rank_fusion({"x": [{"rowid": 7}], "y": [{"rowid": 4}]})
    results.append(("tie broken by rowid", [hit["rowid"] for hit in tied] == [4, 7]))

    results.append(("empty input", reciprocal_rank_fusion({}) == []))
    results.append(("one empty list",
                    len(reciprocal_rank_fusion({"x": a, "y": []})) == 3))

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


# Queries chosen to exercise the two retrievers' opposite failure modes.
PROBES = [
    ("rare literal token — BM25 should carry it", "WMT 2014 English-German"),
    ("paraphrase, no shared vocabulary — vector should carry it",
     "how does the model know word order without recurrence?"),
    ("central term, near-zero IDF — vector should carry it", "attention"),
    ("hyphenated term — crashes FTS5 unescaped", "multi-head attention"),
]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if "--selftest" in sys.argv:
        print("Phase 5 self-test (no DB, no Ollama)\n")
        sys.exit(0 if selftest() else 1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    conn = store.connect()
    store.init_schema(conn)
    if store.stats(conn)["chunks"] == 0:
        print("index is empty — run `python src/ingest.py` first")
        sys.exit(1)

    if "--overlap" in sys.argv:
        print("Retriever disagreement — does fusion have anything to do?")
        overlap_report(conn, [q for _, q in PROBES])
        print()
        print("  Low jaccard = retrievers disagree = RRF is doing real work.")
    elif args:
        compare(conn, " ".join(args))
    else:
        for note, query in PROBES:
            print(f"\n### {note}")
            compare(conn, query)
    conn.close()
