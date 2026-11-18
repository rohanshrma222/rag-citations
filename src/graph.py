"""Phase 9: LangGraph orchestration — the pipeline as an explicit state graph.

Phases 5-8 already work as plain functions, and this file deliberately adds no
retrieval, ranking or prompting logic of its own. Every node is a thin wrapper that
reads state, calls the function that already existed, and writes state back. If an
answer is wrong, it is wrong in the phase that produced it, not here.

    START -> retrieve -> rerank -> [gate] -> generate -> verify -> END
                                      \\-> no_answer --------------> END

The gate is the one conditional edge, and the reason the graph is worth building at
all: when the best reranked candidate scores below NO_ANSWER_THRESHOLD, nothing in
the corpus plausibly answers the question, so the expensive generation call is
skipped entirely rather than asking an 8B model to decline politely.

The threshold is measured, not guessed. Over 12 answerable and 6 unanswerable
questions against the sample PDF:

    answerable     -5.95 .. +7.77
    unanswerable  -11.21 .. -9.48

which separates with a 3.54-point gap, so the default sits at its midpoint. Two
caveats worth keeping in mind: that is one document and 18 questions, and the
margin is thin — phrasing alone moved one question 9.26 points ("Why is the dot
product scaled by the square root of the key dimension?" scored +3.31, while "why
divide by the square root of the key dimension?" scored -5.95). Phase 10 is where
this gets re-derived properly.
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Annotated, Any, TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import END, START, StateGraph

import generate
import rerank
import search
import store
import verify as verify_mod
from embed import OllamaError

# Midpoint of the measured answerable/unanswerable gap. See the module docstring.
NO_ANSWER_THRESHOLD = float(os.environ.get("NO_ANSWER_THRESHOLD", "-7.5"))


class RagState(TypedDict, total=False):
    """State threaded through the graph.

    `trace` uses an additive reducer so each node appends its own name rather than
    overwriting the list — that is what makes the executed path inspectable
    afterwards, which is the roadmap's "confirm state flows correctly" check.
    """
    question: str
    verify: bool                      # run Phase 8 after generating?
    top_k: int
    candidates_k: int

    candidates: list[dict]            # Phase 5 output (fused)
    chunks: list[dict]                # Phase 6 output (reranked, top_k)
    top_score: float | None

    answer: str
    refused: bool
    citations: list[dict]
    invalid: list[int]
    pages: list[int]
    verification: dict[str, Any] | None

    route: str                        # which branch the gate chose
    trace: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------- #
# nodes — each wraps one already-tested phase
# --------------------------------------------------------------------------- #

def retrieve_node(state: RagState, conn) -> dict:
    """Phase 5: hybrid search."""
    k = state.get("candidates_k", search.CANDIDATES)
    candidates = search.hybrid_search(conn, state["question"], k=k, candidates=k)
    return {"candidates": candidates, "trace": ["retrieve"]}


def rerank_node(state: RagState) -> dict:
    """Phase 6: cross-encoder rerank of the fused candidates."""
    chunks = rerank.rerank(state["question"], state["candidates"],
                           top_k=state.get("top_k", generate.TOP_K))
    top_score = chunks[0]["rerank_score"] if chunks else None
    return {"chunks": chunks, "top_score": top_score, "trace": ["rerank"]}


def generate_node(state: RagState) -> dict:
    """Phase 7: grounded generation, with citations resolved to pages."""
    chunks = state["chunks"]
    result = generate.chat(generate.build_messages(state["question"], chunks))
    answer = result["text"]
    resolved = generate.resolve_citations(answer, chunks)
    return {
        "answer": answer,
        "refused": generate.NO_ANSWER in answer,
        "citations": resolved["citations"],
        "invalid": resolved["invalid"],
        "pages": generate.pages_cited(resolved),
        "trace": ["generate"],
    }


def no_answer_node(state: RagState) -> dict:
    """The gate's other branch: decline without calling the LLM at all."""
    return {
        "answer": generate.NO_ANSWER,
        "refused": True,
        "citations": [], "invalid": [], "pages": [],
        "trace": ["no_answer"],
    }


def verify_node(state: RagState) -> dict:
    """Phase 8: grounding check. Opt-in, since it roughly doubles latency."""
    if not state.get("verify"):
        return {"verification": None, "trace": ["verify:skipped"]}

    report = verify_mod.verify_answer({
        "answer": state.get("answer", ""),
        "chunks": state.get("chunks", []),
        "refused": state.get("refused", False),
    })
    return {"verification": report, "trace": ["verify"]}


# --------------------------------------------------------------------------- #
# the conditional edge
# --------------------------------------------------------------------------- #

def gate(state: RagState) -> str:
    """Route to generation only if retrieval found something plausible.

    Returns "generate" or "no_answer". An empty candidate list also declines —
    there is nothing to put in a prompt.
    """
    score = state.get("top_score")
    if score is None or score < NO_ANSWER_THRESHOLD:
        return "no_answer"
    return "generate"


# --------------------------------------------------------------------------- #
# graph construction
# --------------------------------------------------------------------------- #

def build_graph(conn):
    """Compile the pipeline. `conn` is closed over rather than kept in state.

    A sqlite3 connection is not serialisable, so putting it in state would break
    the moment a checkpointer is added.
    """
    graph = StateGraph(RagState)

    graph.add_node("retrieve", lambda state: retrieve_node(state, conn))
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_answer", no_answer_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", gate,
                                {"generate": "generate", "no_answer": "no_answer"})
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    graph.add_edge("no_answer", END)

    return graph.compile()


def ask(conn, question: str, verify: bool = False, top_k: int = generate.TOP_K,
        candidates_k: int = search.CANDIDATES, app=None) -> dict:
    """Run one question through the graph."""
    app = app or build_graph(conn)
    return app.invoke({
        "question": question,
        "verify": verify,
        "top_k": top_k,
        "candidates_k": candidates_k,
        "trace": [],
    })


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def report(state: dict) -> None:
    """Print one graph run, including the path taken through the nodes."""
    print(f"\n{'=' * 78}\nQ: {state['question']}")
    score = state.get("top_score")
    gated = "no_answer" in state.get("trace", [])
    print(f"  top_score {score:+.2f}" if score is not None else "  top_score n/a",
          f"| threshold {NO_ANSWER_THRESHOLD:+.1f}",
          f"| {'GATED (no LLM call)' if gated else 'passed gate'}")
    print(f"  path: {' -> '.join(state.get('trace', []))}")

    print(f"\n{state.get('answer', '')}\n")

    for citation in state.get("citations", []):
        print(f"  [{citation['marker']}] -> "
              f"{os.path.basename(citation['source'])} page {citation['page_number']}")
    if state.get("invalid"):
        print(f"  !! INVALID markers: {state['invalid']}")
    if state.get("pages"):
        print(f"  pages cited: {state['pages']}")

    report_v = state.get("verification")
    if report_v and not report_v.get("skipped"):
        ratio = report_v.get("grounded_ratio")
        print(f"  grounded {report_v['supported']}/{len(report_v['sentences'])}"
              f"{f' ({ratio:.0%})' if ratio is not None else ''}"
              f" | unsupported {report_v['unsupported']}"
              f" | uncited {report_v['uncited']}")
        for item in report_v.get("spurious", []):
            print(f"  ~~ spurious citation [{item['marker']}] -> p{item['page_number']}")
    elif report_v and report_v.get("skipped"):
        print(f"  verification skipped: {report_v['skipped']}")


def selftest() -> bool:
    """Graph wiring, routing and state flow, with every phase stubbed.

    No Ollama, no cross-encoder, no database. This tests that the graph is wired
    correctly — the phases themselves have their own selftests.
    """
    results = []

    def chunk(rowid, page, text):
        return {"rowid": rowid, "chunk_id": f"c{rowid}", "source": "a.pdf",
                "page_number": page, "text": text}

    fake_chunks = [chunk(1, 4, "Multi-head attention runs heads in parallel."),
                   chunk(2, 7, "Adam was used.")]

    calls: list[str] = []

    def fake_hybrid(conn, question, k=20, candidates=20):
        calls.append("hybrid")
        return list(fake_chunks)

    def make_fake_rerank(score):
        def fake_rerank(question, candidates, top_k=None, model=None):
            calls.append("rerank")
            return [{**c, "rerank_score": score - i, "prior_rank": i + 1}
                    for i, c in enumerate(candidates)][:top_k or len(candidates)]
        return fake_rerank

    def fake_chat(messages, model=None):
        calls.append("chat")
        return {"text": "Heads run in parallel [1].", "elapsed": 0.0,
                "eval_count": 7, "prompt_eval_count": 100}

    def fake_ask_verifier(source_text, claim, model=None):
        calls.append("verifier")
        return {"supported": True, "raw": "YES"}

    real = (search.hybrid_search, rerank.rerank, generate.chat, verify_mod._ask)
    search.hybrid_search = fake_hybrid
    generate.chat = fake_chat
    verify_mod._ask = fake_ask_verifier
    try:
        # --- happy path: score above threshold ---
        rerank.rerank = make_fake_rerank(5.0)
        calls.clear()
        state = ask(None, "What is multi-head attention?")

        results.append(("runs retrieve then rerank then generate",
                        state["trace"] == ["retrieve", "rerank", "generate",
                                           "verify:skipped"]))
        results.append(("answer reaches state", state["answer"].startswith("Heads run")))
        results.append(("citation resolved to page",
                        [c["page_number"] for c in state["citations"]] == [4]))
        results.append(("pages populated", state["pages"] == [4]))
        results.append(("top_score recorded", state["top_score"] == 5.0))
        results.append(("not refused", state["refused"] is False))
        results.append(("llm was called", "chat" in calls))
        results.append(("verifier not called when off", "verifier" not in calls))
        results.append(("verification is None when off", state["verification"] is None))

        # --- gated path: score below threshold skips generation ---
        rerank.rerank = make_fake_rerank(NO_ANSWER_THRESHOLD - 1.0)
        calls.clear()
        gated = ask(None, "What is the capital of Australia?")

        results.append(("gate routes to no_answer",
                        gated["trace"] == ["retrieve", "rerank", "no_answer"]))
        results.append(("gated answer is the refusal token",
                        gated["answer"] == generate.NO_ANSWER))
        results.append(("gated marks refused", gated["refused"] is True))
        results.append(("gated skips the LLM entirely", "chat" not in calls))
        results.append(("gated has no citations", gated["citations"] == []))
        results.append(("gated never reaches verify",
                        "verify" not in gated["trace"] and
                        "verify:skipped" not in gated["trace"]))

        # --- verification enabled ---
        rerank.rerank = make_fake_rerank(5.0)
        calls.clear()
        verified = ask(None, "What is multi-head attention?", verify=True)
        results.append(("verify node runs when on",
                        verified["trace"][-1] == "verify"))
        results.append(("verifier was called", "verifier" in calls))
        results.append(("verification report present",
                        verified["verification"]["supported"] == 1))

        # --- boundary: exactly at the threshold passes ---
        rerank.rerank = make_fake_rerank(NO_ANSWER_THRESHOLD)
        at_threshold = ask(None, "edge case")
        results.append(("score == threshold passes the gate",
                        "generate" in at_threshold["trace"]))

        # --- no candidates at all must decline, not crash ---
        def empty_rerank(question, candidates, top_k=None, model=None):
            return []
        rerank.rerank = empty_rerank
        empty = ask(None, "nothing matches")
        results.append(("empty candidates decline",
                        empty["answer"] == generate.NO_ANSWER))
        results.append(("empty candidates route to no_answer",
                        "no_answer" in empty["trace"]))

        # --- gate function in isolation ---
        results.append(("gate: above", gate({"top_score": 0.0}) == "generate"))
        results.append(("gate: below",
                        gate({"top_score": NO_ANSWER_THRESHOLD - 0.1}) == "no_answer"))
        results.append(("gate: None", gate({"top_score": None}) == "no_answer"))
        results.append(("gate: missing key", gate({}) == "no_answer"))

        # --- the trace reducer must append, not overwrite ---
        results.append(("trace reducer appends", len(verified["trace"]) == 4))
    finally:
        search.hybrid_search, rerank.rerank, generate.chat, verify_mod._ask = real

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if "--selftest" in sys.argv:
        print("Phase 9 self-test (all phases stubbed)\n")
        sys.exit(0 if selftest() else 1)

    conn = store.connect()
    store.init_schema(conn)
    if store.stats(conn)["chunks"] == 0:
        print("index is empty — run `python src/ingest.py` first")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_verify = "--verify" in sys.argv

    try:
        app = build_graph(conn)

        if "--ascii" in sys.argv:
            drawn = app.get_graph()
            try:
                print(drawn.draw_ascii())
            except ImportError:
                # draw_ascii needs grandalf, which is optional. The edge list says
                # the same thing without another dependency.
                print("nodes: " + ", ".join(n.id for n in drawn.nodes.values()))
                print("edges:")
                for edge in drawn.edges:
                    label = f"  [{edge.data}]" if edge.data else ""
                    dashed = " (conditional)" if edge.conditional else ""
                    print(f"  {edge.source} -> {edge.target}{label}{dashed}")
                print("\n(install grandalf for an ASCII diagram)")
        elif args:
            report(ask(conn, " ".join(args), verify=do_verify, app=app))
        else:
            for question in generate.ANSWERABLE[:3]:
                report(ask(conn, question, verify=do_verify, app=app))
            for question in generate.UNANSWERABLE:
                report(ask(conn, question, verify=do_verify, app=app))
    except OllamaError as exc:
        print(f"\nOllamaError: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        conn.close()
