"""Phase 7: Grounded generation — answer from retrieved chunks, with real citations.

The model never sees page numbers. It sees sources numbered [1]..[n] and is told to
cite those numbers; the numbers are then resolved back to (source, page) from the
chunk list we built the prompt from. That indirection is deliberate: asking a small
local model to emit "page 6" invites it to invent a plausible page, whereas emitting
"[2]" is a pointer into a list it was handed, and a pointer can be checked.

Two failure modes this file is built around:

  citation drift   the model cites [7] when only 5 sources were given, or cites a
                   source that does not support the claim. Out-of-range markers are
                   detected here; unsupported claims are Phase 8's job.
  confident guessing
                   small models would rather answer than decline. The prompt makes
                   declining an explicit instruction, and `answer_question` reports
                   whether retrieval scores were negative so a caller can tell
                   "the sources did not cover this" from "the model gave up".

temperature=0: for citation work determinism matters more than fluency, and it makes
prompt changes attributable to the prompt rather than to sampling noise.
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rerank
import search
import store
from embed import OLLAMA_URL, OllamaError, _post

CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1:8b")
TOP_K = 5                    # sources shown to the model
TEMPERATURE = 0.0
NUM_PREDICT = 400            # cap the answer length; citation answers are short
NO_ANSWER = "INSUFFICIENT_CONTEXT"

# An 8B model on CPU runs at single-digit tokens/second and must also process a
# ~1500-token prompt, so one answer can take minutes. The embedding default of
# 120s is far too short here.
CHAT_TIMEOUT = int(os.environ.get("CHAT_TIMEOUT", "900"))

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources provided.

Rules:
1. Every factual sentence must end with a citation marker like [1] or [2][3].
2. Cite ONLY the source numbers you were given. Never invent a number.
3. Use the sources' own facts. Do not add outside knowledge, even if you know it.
4. If the sources do not contain the answer, reply with exactly: {NO_ANSWER}
   Do not guess, and do not apologise — just that one word.
5. Be concise: two or three sentences unless more is genuinely needed."""

USER_TEMPLATE = """Sources:
{sources}

Question: {question}

Answer using only the sources above, citing them by number."""

# Matches [1] and the individual numbers inside [2][3] or [2, 3].
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #

def format_sources(chunks: list[dict]) -> str:
    """Number the chunks for the prompt. Numbering is 1-based and positional."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        text = " ".join(chunk["text"].split())
        blocks.append(f"[{i}] {text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """The chat messages sent to Ollama."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            sources=format_sources(chunks), question=question)},
    ]


# --------------------------------------------------------------------------- #
# the model call
# --------------------------------------------------------------------------- #

def chat(messages: list[dict], model: str = CHAT_MODEL) -> dict:
    """One /api/chat call. Returns {"text", "elapsed", "eval_count", ...}."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
    }
    started = time.perf_counter()
    status, body = _post("/api/chat", payload, timeout=CHAT_TIMEOUT)
    elapsed = time.perf_counter() - started

    if status == 404:
        raise OllamaError(f"Model '{model}' is not pulled. Run:  ollama pull {model}")
    if status != 200:
        raise OllamaError(f"/api/chat returned {status}: {body.get('error', body)}")

    return {
        "text": body.get("message", {}).get("content", "").strip(),
        "elapsed": elapsed,
        "eval_count": body.get("eval_count", 0),
        "prompt_eval_count": body.get("prompt_eval_count", 0),
    }


# --------------------------------------------------------------------------- #
# citation resolution
# --------------------------------------------------------------------------- #

def extract_citations(answer: str) -> list[int]:
    """Every source number cited, in order of first appearance, deduplicated.

    Handles [1], [2][3] and [2, 3]. Returns the raw numbers as written — including
    out-of-range ones, which `resolve_citations` reports rather than hides.
    """
    seen: list[int] = []
    for group in _CITATION.findall(answer):
        for part in group.split(","):
            number = int(part.strip())
            if number not in seen:
                seen.append(number)
    return seen


def resolve_citations(answer: str, chunks: list[dict]) -> dict:
    """Map cited numbers back to (source, page). Reports invalid markers.

    Returns {"citations": [...], "invalid": [...], "unused": [...]}, where each
    citation carries the marker, source, page_number, chunk_id and rowid.
    """
    cited = extract_citations(answer)
    citations, invalid = [], []

    for number in cited:
        if 1 <= number <= len(chunks):
            chunk = chunks[number - 1]
            citations.append({
                "marker": number,
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk.get("chunk_id", chunk.get("id")),
                "rowid": chunk.get("rowid"),
            })
        else:
            invalid.append(number)

    used = {c["marker"] for c in citations}
    unused = [i for i in range(1, len(chunks) + 1) if i not in used]
    return {"citations": citations, "invalid": invalid, "unused": unused}


def pages_cited(resolved: dict) -> list[int]:
    """Just the distinct page numbers, sorted — what a user actually reads."""
    return sorted({c["page_number"] for c in resolved["citations"]})


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #

def answer_question(conn, question: str, top_k: int = TOP_K,
                    candidates: int = search.CANDIDATES,
                    model: str = CHAT_MODEL) -> dict:
    """Retrieve, rerank, generate, and resolve citations to pages."""
    chunks = rerank.search_and_rerank(conn, question, top_k=top_k,
                                      candidates=candidates)
    if not chunks:
        return {"question": question, "answer": NO_ANSWER, "refused": True,
                "chunks": [], "citations": [], "invalid": [], "unused": [],
                "pages": [], "top_score": None, "elapsed": 0.0}

    result = chat(build_messages(question, chunks), model=model)
    answer = result["text"]
    resolved = resolve_citations(answer, chunks)

    return {
        "question": question,
        "answer": answer,
        "refused": NO_ANSWER in answer,
        "chunks": chunks,
        "citations": resolved["citations"],
        "invalid": resolved["invalid"],
        "unused": resolved["unused"],
        "pages": pages_cited(resolved),
        # Negative reranker logits mean retrieval found nothing convincing. Useful
        # for telling "sources did not cover it" from "model declined anyway".
        "top_score": chunks[0].get("rerank_score"),
        "elapsed": result["elapsed"],
        "eval_count": result["eval_count"],
        "prompt_eval_count": result["prompt_eval_count"],
    }


def report(result: dict) -> None:
    """Print one answer with its citations resolved to pages."""
    print(f"\n{'=' * 78}\nQ: {result['question']}")
    score = result.get("top_score")
    print(f"  retrieval top score: {score:+.2f}" if score is not None
          else "  retrieval: no candidates")

    print(f"\n{result['answer']}\n")

    if result["refused"]:
        print("  -> declined (sources judged insufficient)")
    for citation in result["citations"]:
        print(f"  [{citation['marker']}] -> {os.path.basename(citation['source'])} "
              f"page {citation['page_number']}")
    if result["invalid"]:
        print(f"  !! INVALID markers (out of range 1-{len(result['chunks'])}): "
              f"{result['invalid']}")
    if result["pages"]:
        print(f"  pages cited: {result['pages']}")
    if result.get("eval_count"):
        tps = result["eval_count"] / result["elapsed"] if result["elapsed"] else 0
        print(f"  {result['elapsed']:.1f}s | {result['eval_count']} tokens "
              f"({tps:.1f} tok/s) | prompt {result['prompt_eval_count']} tokens")


def selftest() -> bool:
    """Prompt building and citation parsing. No model call."""
    results = []

    chunks = [
        {"id": "a_p3_c0", "source": "sample-pdfs/a.pdf", "page_number": 3,
         "text": "Multi-head attention runs h attention functions in parallel.",
         "rowid": 11},
        {"id": "a_p7_c1", "source": "sample-pdfs/a.pdf", "page_number": 7,
         "text": "We used the Adam optimiser with beta1 = 0.9.", "rowid": 12},
    ]

    # --- prompt shape ---
    formatted = format_sources(chunks)
    results.append(("sources numbered from 1", formatted.startswith("[1] ")))
    results.append(("second source numbered 2", "\n\n[2] " in formatted))
    results.append(("whitespace collapsed", "\n" not in formatted.split("\n\n")[0]))
    results.append(("no page numbers leak into prompt",
                    "page 3" not in formatted and "page_number" not in formatted))

    messages = build_messages("What is multi-head attention?", chunks)
    results.append(("system then user", [m["role"] for m in messages] == ["system", "user"]))
    results.append(("question in user message",
                    "What is multi-head attention?" in messages[1]["content"]))
    results.append(("refusal token in system prompt", NO_ANSWER in messages[0]["content"]))

    # --- citation extraction ---
    results.append(("single marker", extract_citations("Yes [1].") == [1]))
    results.append(("adjacent markers", extract_citations("Both [1][2].") == [1, 2]))
    results.append(("comma group", extract_citations("Both [1, 2].") == [1, 2]))
    results.append(("deduplicated, order kept",
                    extract_citations("A [2]. B [1]. C [2].") == [2, 1]))
    results.append(("no citations", extract_citations("I do not know.") == []))
    results.append(("ignores bracketed non-numbers",
                    extract_citations("See [note] and [1].") == [1]))

    # --- resolution ---
    resolved = resolve_citations("Parallel heads [1]. Adam [2].", chunks)
    results.append(("resolves to pages",
                    [c["page_number"] for c in resolved["citations"]] == [3, 7]))
    results.append(("carries chunk_id", resolved["citations"][0]["chunk_id"] == "a_p3_c0"))
    results.append(("no invalid", resolved["invalid"] == []))
    results.append(("no unused", resolved["unused"] == []))

    # out-of-range marker must be reported, not silently dropped or crash
    bad = resolve_citations("Claimed [7].", chunks)
    results.append(("flags out-of-range", bad["invalid"] == [7]))
    results.append(("no bogus citation for it", bad["citations"] == []))
    results.append(("zero is invalid", resolve_citations("X [0].", chunks)["invalid"] == [0]))

    partial = resolve_citations("Only the first [1].", chunks)
    results.append(("reports unused sources", partial["unused"] == [2]))
    results.append(("pages_cited sorted distinct",
                    pages_cited(resolve_citations("A [2]. B [1]. C [2].", chunks)) == [3, 7]))

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


# Questions the PDF answers, plus two it does not — the hallucination canary.
ANSWERABLE = [
    "What is multi-head attention?",
    "What optimiser was used for training, and with what learning rate schedule?",
    "How many GPUs were used to train the base model, and for how long?",
    "What BLEU score did the big model reach on WMT 2014 English-to-German?",
    "Why is the dot product scaled by the square root of the key dimension?",
]
UNANSWERABLE = [
    "What is the capital of Australia?",
    "What were the authors' salaries while writing this paper?",
]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if "--selftest" in sys.argv:
        print("Phase 7 self-test (no model call)\n")
        sys.exit(0 if selftest() else 1)

    conn = store.connect()
    store.init_schema(conn)
    if store.stats(conn)["chunks"] == 0:
        print("index is empty — run `python src/ingest.py` first")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    try:
        if args:
            report(answer_question(conn, " ".join(args)))
        elif "--canary" in sys.argv:
            print(f"Hallucination canary — the PDF does NOT answer these.\n"
                  f"Expect {NO_ANSWER}, not a plausible guess.")
            for question in UNANSWERABLE:
                report(answer_question(conn, question))
        else:
            for question in ANSWERABLE:
                report(answer_question(conn, question))
            print(f"\n{'#' * 78}\n# Canary: questions the PDF does not answer\n{'#' * 78}")
            for question in UNANSWERABLE:
                report(answer_question(conn, question))
    except OllamaError as exc:
        print(f"\nOllamaError: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        conn.close()
