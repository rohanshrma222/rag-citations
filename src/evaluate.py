"""Phase 10: Evaluation — measure the pipeline instead of trusting it.

A hand-built test set of questions whose answers live on pages I can point to, and
a harness that reports two different things:

  retrieval recall   was the correct page among the top-k chunks retrieved?
                     Cheap (no LLM), so this is what you tune chunk size, k and
                     reranker choice against.
  citation accuracy  did the final answer actually cite that page? A pipeline can
                     retrieve the right chunk and still cite the wrong one.

Every item carries an `evidence` substring that must appear in the indexed text of
its expected page. `validate_dataset()` checks that before any measurement runs, so
a ground-truth page number that is simply wrong fails loudly rather than quietly
making the system look broken. Hand-written ground truth is the weakest link in any
eval, and this is the cheapest possible guard on it.

Recall is reported per retrieval configuration, which is what finally settles the
questions left open earlier: whether hybrid beats either retriever alone (Phase 5
could not show it on 59 chunks), and whether the cross-encoder helps on academic
prose (Phase 6 found it demoting a correct chunk).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate
import graph
import rerank
import search
import store
from embed import OllamaError, embed_query

# --------------------------------------------------------------------------- #
# the test set — hand built, with evidence for every ground-truth page
# --------------------------------------------------------------------------- #
# (question, expected_pages, evidence_substring)
# `expected_pages` may list more than one page when the fact genuinely appears in
# several places; a hit on any of them counts.

DATASET = [
    ("How many layers are in the encoder stack?", [3], "N = 6"),
    ("What is the dimension of the model's embeddings?", [3], "dmodel = 512"),
    ("Why is the dot product divided by the square root of the key dimension?",
     [4], "extremely small gradients"),
    ("What are the two most commonly used attention functions?",
     [4], "dot-product"),
    ("How many attention heads does the base model use?", [5], "h = 8"),
    ("What happens to the dimension of each head as the number of heads grows?",
     [5], "dk = dv"),
    ("How does the model inject information about token position?",
     [6], "positional encodings"),
    ("What functions are used to compute positional encodings?",
     [6], "sine and cosine"),
    ("What is the per-layer computational complexity of self-attention?",
     [6], "Complexity per Layer"),
    ("How many sentence pairs were in the English-German training data?",
     [7], "4.5 million"),
    ("What optimizer was used to train the model?", [7], "Adam"),
    ("How many warmup steps were used in the learning rate schedule?",
     [7], "4000"),
    ("What hardware was the model trained on?", [7], "P100"),
    ("How long did the base model take to train?", [7], "12 hours"),
    ("What BLEU score did the big model achieve on English-to-German?",
     [8], "28.4"),
    ("What dropout rate was applied to sublayer outputs?", [8], "residual dropout"),
    ("What beam size was used for decoding?", [8], "beam size of 4"),
    # Section 6.3 begins on p9 and its results table is on p10, so both pages
    # legitimately answer this. Scoring it as p10-only marked a correct citation
    # of p9 as a failure.
    ("What task was used to test whether the Transformer generalizes?",
     [9, 10], "WSJ 23"),
    ("What is intra-attention?", [2], "intra-attention"),
    # p3 states the decoder masks subsequent positions; p5 (3.2.3) explains the
    # mechanism -- "prevent leftward information flow ... masking out (setting to
    # -inf) all values ... which correspond to illegal connections" -- and is
    # arguably the better citation. Both count.
    ("How is the decoder prevented from attending to future positions?",
     [3, 5], "masking"),
]

# Questions the document does not answer. Used to measure the refusal rate and the
# Phase 9 gate, not retrieval recall.
UNANSWERABLE = [
    "What is the capital of Australia?",
    "Who won the 2022 FIFA World Cup?",
    "What is the best recipe for sourdough bread?",
    "What were the authors' salaries while writing this paper?",
    "How do I configure a Kubernetes ingress controller?",
]


# --------------------------------------------------------------------------- #
# ground-truth validation
# --------------------------------------------------------------------------- #

def validate_dataset(conn, dataset=DATASET) -> tuple[bool, list[str]]:
    """Check each expected page actually contains its evidence string.

    Searches the indexed chunk text rather than the PDF, because the chunks are
    what retrieval can possibly return. A fact that survives parsing but is lost
    in chunking would otherwise look like a retrieval failure.
    """
    problems = []
    for question, pages, evidence in dataset:
        found_on = [
            row["page_number"]
            for row in conn.execute(
                "SELECT DISTINCT page_number FROM chunks WHERE text LIKE ?",
                (f"%{evidence}%",))
        ]
        if not found_on:
            problems.append(f"evidence {evidence!r} not found in ANY chunk "
                            f"-- {question[:48]!r}")
        elif not set(pages) & set(found_on):
            problems.append(f"evidence {evidence!r} is on pages {sorted(found_on)}, "
                            f"not {pages} -- {question[:48]!r}")
    return not problems, problems


# --------------------------------------------------------------------------- #
# retrieval recall
# --------------------------------------------------------------------------- #

CONFIGS = ("vector", "keyword", "hybrid", "hybrid+rerank")


_VECTOR_CACHE: dict[str, list[float]] = {}


def query_vector(question: str) -> list[float]:
    """Embed a question once per process.

    A sweep runs each question through 4 configurations at 5 values of k. Without
    this, that is 20 identical embedding round-trips per question and embedding
    dominates the runtime of the whole evaluation.
    """
    if question not in _VECTOR_CACHE:
        _VECTOR_CACHE[question] = embed_query(question)
    return _VECTOR_CACHE[question]


def retrieved_pages(conn, question: str, config: str, k: int,
                    candidates: int, model=None) -> list[int]:
    """Pages of the top-k chunks under one retrieval configuration."""
    if config == "vector":
        hits = search.vector_search(conn, question, k=k,
                                    query_vector=query_vector(question))
    elif config == "keyword":
        hits = search.keyword_search(conn, question, k=k)
    elif config == "hybrid":
        hits = search.hybrid_search(conn, question, k=k, candidates=candidates,
                                    query_vector=query_vector(question))
    elif config == "hybrid+rerank":
        fused = search.hybrid_search(conn, question, k=candidates,
                                     candidates=candidates,
                                     query_vector=query_vector(question))
        hits = rerank.rerank(question, fused, top_k=k, model=model)
    else:
        raise ValueError(f"unknown config {config!r}")
    return [hit["page_number"] for hit in hits]


def recall_at_k(conn, dataset=DATASET, k: int = 5,
                candidates: int = search.CANDIDATES,
                configs=CONFIGS) -> dict:
    """Recall@k per configuration, plus the per-question detail."""
    model = rerank.get_model() if "hybrid+rerank" in configs else None
    results = {config: {"hits": 0, "misses": [], "ranks": []} for config in configs}

    for question, pages, _ in dataset:
        wanted = set(pages)
        for config in configs:
            got = retrieved_pages(conn, question, config, k, candidates, model)
            if wanted & set(got):
                results[config]["hits"] += 1
                rank = next(i for i, page in enumerate(got, 1) if page in wanted)
                results[config]["ranks"].append(rank)
            else:
                results[config]["misses"].append((question, pages, got))

    total = len(dataset)
    for config in configs:
        entry = results[config]
        entry["total"] = total
        entry["recall"] = entry["hits"] / total if total else 0.0
        ranks = entry["ranks"]
        # Mean reciprocal rank: rewards putting the right page first, not merely
        # somewhere in the top k.
        entry["mrr"] = sum(1 / r for r in ranks) / total if total else 0.0
    return results


# --------------------------------------------------------------------------- #
# end-to-end citation accuracy
# --------------------------------------------------------------------------- #

def citation_accuracy(conn, dataset=DATASET, app=None, verify: bool = False) -> dict:
    """Did the final answer cite the expected page?

    Runs the Phase 9 graph, so the gate is part of what is measured: a false
    refusal on an answerable question counts as a miss and is reported separately.
    """
    app = app or graph.build_graph(conn)
    rows, correct, refused, invalid_markers = [], 0, 0, 0

    for question, pages, _ in dataset:
        state = graph.ask(conn, question, verify=verify, app=app)
        cited = state.get("pages", [])
        hit = bool(set(pages) & set(cited))
        if hit:
            correct += 1
        if state.get("refused"):
            refused += 1
        invalid_markers += len(state.get("invalid", []))
        rows.append({"question": question, "expected": pages, "cited": cited,
                     "hit": hit, "refused": bool(state.get("refused")),
                     "top_score": state.get("top_score"),
                     "gated": "no_answer" in state.get("trace", [])})

    total = len(dataset)
    return {"rows": rows, "correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0,
            "false_refusals": refused, "invalid_markers": invalid_markers}


def refusal_rate(conn, questions=UNANSWERABLE, app=None) -> dict:
    """Does the pipeline decline questions the document does not answer?"""
    app = app or graph.build_graph(conn)
    rows, declined = [], 0
    for question in questions:
        state = graph.ask(conn, question, app=app)
        ok = bool(state.get("refused"))
        declined += ok
        rows.append({"question": question, "refused": ok,
                     "gated": "no_answer" in state.get("trace", []),
                     "top_score": state.get("top_score"),
                     "answer": state.get("answer", "")[:60]})
    return {"rows": rows, "declined": declined, "total": len(questions),
            "rate": declined / len(questions) if questions else 0.0}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_recall(results: dict, k: int) -> None:
    print(f"\n  Retrieval recall@{k} ({results[next(iter(results))]['total']} questions)")
    print(f"  {'config':<16} {'recall':>12}  {'MRR':>5}")
    for config, entry in results.items():
        print(f"  {config:<16} {entry['hits']:>3}/{entry['total']:<3} "
              f"{entry['recall']:>5.0%}  {entry['mrr']:.3f}")

    for config, entry in results.items():
        if entry["misses"]:
            print(f"\n  {config} missed {len(entry['misses'])}:")
            for question, pages, got in entry["misses"]:
                print(f"    want p{pages}, got p{sorted(set(got))}  {question[:50]!r}")


def print_citations(report: dict) -> None:
    print(f"\n  Citation accuracy (end to end through the graph)")
    print(f"  {report['correct']}/{report['total']} = {report['accuracy']:.0%} "
          f"cited the expected page")
    print(f"  false refusals: {report['false_refusals']} | "
          f"invalid markers: {report['invalid_markers']}")
    misses = [row for row in report["rows"] if not row["hit"]]
    if misses:
        print(f"\n  missed {len(misses)}:")
        for row in misses:
            tag = " GATED" if row["gated"] else ""
            score = f"{row['top_score']:+.2f}" if row["top_score"] is not None else "n/a"
            print(f"    want p{row['expected']}, cited p{row['cited']} "
                  f"(score {score}){tag}  {row['question'][:44]!r}")


def print_refusals(report: dict) -> None:
    print(f"\n  Refusal rate on unanswerable questions")
    print(f"  {report['declined']}/{report['total']} = {report['rate']:.0%} declined")
    for row in report["rows"]:
        mark = "OK  " if row["refused"] else "LEAK"
        score = f"{row['top_score']:+.2f}" if row["top_score"] is not None else "n/a"
        print(f"    [{mark}] score {score} {'gated' if row['gated'] else '     '} "
              f"{row['question'][:40]!r}")
        if not row["refused"]:
            print(f"           answered: {row['answer']!r}")


def sweep_k(conn, dataset=DATASET, ks=(1, 3, 5, 10, 20)) -> None:
    """Recall at several k, which is what tells you where to set top_k."""
    model = rerank.get_model()
    print(f"\n  Recall at increasing k")
    header = "  " + f"{'k':>3}" + "".join(f"{c:>16}" for c in CONFIGS)
    print(header)
    for k in ks:
        row = f"  {k:>3}"
        for config in CONFIGS:
            hits = 0
            for question, pages, _ in dataset:
                got = retrieved_pages(conn, question, config, k,
                                      search.CANDIDATES, model)
                hits += bool(set(pages) & set(got))
            row += f"{hits:>10}/{len(dataset):<5}"
        print(row)


def selftest() -> bool:
    """Metric arithmetic and dataset shape. No database, no models."""
    results = []

    # --- dataset shape ---
    results.append(("dataset has 20 items", len(DATASET) == 20))
    results.append(("all items are triples", all(len(item) == 3 for item in DATASET)))
    results.append(("pages are non-empty int lists",
                    all(item[1] and all(isinstance(p, int) for p in item[1])
                        for item in DATASET)))
    results.append(("evidence strings non-empty",
                    all(item[2].strip() for item in DATASET)))
    results.append(("questions unique",
                    len({item[0] for item in DATASET}) == len(DATASET)))
    results.append(("unanswerable set non-empty", len(UNANSWERABLE) >= 3))
    results.append(("answerable and unanswerable disjoint",
                    not ({item[0] for item in DATASET} & set(UNANSWERABLE))))

    # --- recall arithmetic, with retrieval stubbed ---
    g = globals()
    tiny = [("q1", [3], "e1"), ("q2", [7], "e2"), ("q3", [9], "e3")]
    # q1 hits at rank 1, q2 hits at rank 3, q3 misses entirely
    pages_by_question = {"q1": [3, 5, 6], "q2": [1, 2, 7], "q3": [1, 2, 4]}

    real = g["retrieved_pages"]
    g["retrieved_pages"] = lambda conn, q, config, k, candidates, model=None: \
        pages_by_question[q]
    try:
        out = recall_at_k(None, dataset=tiny, k=3, configs=("hybrid",))["hybrid"]
        results.append(("counts hits", out["hits"] == 2))
        results.append(("recall fraction", abs(out["recall"] - 2 / 3) < 1e-9))
        results.append(("records the miss", len(out["misses"]) == 1))
        results.append(("miss names the question", out["misses"][0][0] == "q3"))
        results.append(("ranks captured", sorted(out["ranks"]) == [1, 3]))
        # MRR = (1/1 + 1/3 + 0) / 3
        results.append(("mrr arithmetic",
                        abs(out["mrr"] - (1 + 1 / 3) / 3) < 1e-9))

        perfect = {"q1": [3], "q2": [7], "q3": [9]}
        g["retrieved_pages"] = lambda conn, q, config, k, candidates, model=None: \
            perfect[q]
        allhit = recall_at_k(None, dataset=tiny, k=1, configs=("hybrid",))["hybrid"]
        results.append(("perfect recall", allhit["recall"] == 1.0))
        results.append(("perfect mrr", abs(allhit["mrr"] - 1.0) < 1e-9))

        g["retrieved_pages"] = lambda conn, q, config, k, candidates, model=None: []
        nohit = recall_at_k(None, dataset=tiny, k=1, configs=("hybrid",))["hybrid"]
        results.append(("zero recall", nohit["recall"] == 0.0))
        results.append(("zero mrr", nohit["mrr"] == 0.0))
    finally:
        g["retrieved_pages"] = real

    # --- multi-page ground truth: a hit on any expected page counts ---
    multi = [("q", [4, 5], "e")]
    g["retrieved_pages"] = lambda conn, q, config, k, candidates, model=None: [5]
    try:
        out = recall_at_k(None, dataset=multi, k=1, configs=("hybrid",))["hybrid"]
        results.append(("any expected page counts", out["hits"] == 1))
    finally:
        g["retrieved_pages"] = real

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if "--selftest" in sys.argv:
        print("Phase 10 self-test (no DB, no models)\n")
        sys.exit(0 if selftest() else 1)

    conn = store.connect()
    store.init_schema(conn)
    if store.stats(conn)["chunks"] == 0:
        print("index is empty — run `python src/ingest.py` first")
        sys.exit(1)

    k = 5
    for arg in sys.argv[1:]:
        if arg.startswith("--k="):
            k = int(arg.split("=", 1)[1])

    try:
        print(f"Evaluating {len(DATASET)} answerable + {len(UNANSWERABLE)} "
              f"unanswerable questions | index: {store.stats(conn)}")

        print("\n[1] Ground-truth validation")
        ok, problems = validate_dataset(conn)
        if ok:
            print(f"  PASS: all {len(DATASET)} evidence strings found on their "
                  f"expected pages")
        else:
            print(f"  FAIL: {len(problems)} ground-truth problems")
            for problem in problems:
                print(f"    {problem}")
            print("\n  Refusing to report metrics against bad ground truth.")
            sys.exit(1)

        print("\n[2] Retrieval recall (no LLM)")
        started = time.perf_counter()
        print_recall(recall_at_k(conn, k=k), k)
        print(f"\n  ({time.perf_counter() - started:.0f}s)")

        if "--sweep" in sys.argv:
            sweep_k(conn)

        if "--retrieval-only" in sys.argv:
            sys.exit(0)

        print("\n[3] End-to-end citation accuracy (calls the LLM)")
        started = time.perf_counter()
        print_citations(citation_accuracy(conn))
        print(f"\n  ({time.perf_counter() - started:.0f}s)")

        print("\n[4] Refusals")
        print_refusals(refusal_rate(conn))

    except OllamaError as exc:
        print(f"\nOllamaError: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        conn.close()
