"""Phase 3: Embeddings — turn chunk text into vectors via a local Ollama model.

Talks to Ollama's raw REST API rather than going through a client library, so the
request/response shape is visible. Two endpoints exist and they are not the same:

  POST /api/embed       newer, batched:   {"input": ["a", "b"]} -> {"embeddings": [[...], [...]]}
  POST /api/embeddings  legacy, single:   {"prompt": "a"}       -> {"embedding": [...]}

We use the batched endpoint and fall back to the legacy one (parallelised across a
thread pool) if the installed Ollama is too old to have it.

nomic-embed-text is asymmetric: indexed text must be prefixed "search_document: "
and queries "search_query: ". Skipping the prefixes silently degrades retrieval
quality rather than erroring, so they are applied here, in one place, and never
left to callers.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

EMBED_DIM = 768          # nomic-embed-text output width; asserted on every response
BATCH_SIZE = 32          # texts per /api/embed request
MAX_PARALLEL = 8         # threads for the legacy single-text fallback
TIMEOUT = 120            # seconds; a cold model load can take a while on first call

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class OllamaError(RuntimeError):
    """Ollama is unreachable, the model is missing, or a response was malformed."""


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #

def _post(path: str, payload: dict) -> tuple[int, dict]:
    """POST JSON to Ollama. Returns (status_code, parsed_body)."""
    try:
        resp = requests.post(f"{OLLAMA_URL}{path}", json=payload, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_URL}.\n"
            f"  Install:  winget install Ollama.Ollama\n"
            f"  Serve:    ollama serve\n"
            f"  Model:    ollama pull {MODEL}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(f"Ollama timed out after {TIMEOUT}s on {path}.") from exc

    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text[:500]}

    if resp.status_code == 404 and "model" in str(body.get("error", "")).lower():
        raise OllamaError(
            f"Model '{MODEL}' is not pulled. Run:  ollama pull {MODEL}"
        )
    return resp.status_code, body


def _validate(vectors: list, expected: int) -> list[list[float]]:
    """Check shape and width before anything downstream trusts these numbers."""
    if len(vectors) != expected:
        raise OllamaError(f"Expected {expected} embeddings, got {len(vectors)}.")
    for vec in vectors:
        if not isinstance(vec, list) or len(vec) != EMBED_DIM:
            got = len(vec) if isinstance(vec, list) else type(vec).__name__
            raise OllamaError(
                f"Expected {EMBED_DIM}-dim vectors from '{MODEL}', got {got}. "
                f"If you changed EMBED_MODEL, update EMBED_DIM to match — the "
                f"Phase 4 vec0 table is declared with a fixed width."
            )
    return vectors


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """One /api/embed call for a list of texts, with legacy fallback."""
    status, body = _post("/api/embed", {"model": MODEL, "input": texts})

    if status == 404:
        # Old Ollama: no batch endpoint. One request per text, in parallel.
        return _embed_legacy(texts)
    if status != 200:
        raise OllamaError(f"/api/embed returned {status}: {body.get('error', body)}")

    return _validate(body.get("embeddings", []), len(texts))


def _embed_legacy(texts: list[str]) -> list[list[float]]:
    """Fallback for Ollama versions predating /api/embed."""
    def one(text: str) -> list[float]:
        status, body = _post("/api/embeddings", {"model": MODEL, "prompt": text})
        if status != 200:
            raise OllamaError(f"/api/embeddings returned {status}: {body.get('error', body)}")
        return body.get("embedding", [])

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        return _validate(list(pool.map(one, texts)), len(texts))


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def embed_documents(texts: list[str], progress: bool = False) -> list[list[float]]:
    """Embed text destined for the index. Applies the search_document prefix."""
    if not texts:
        return []

    vectors: list[list[float]] = []
    batches = list(_batched([DOC_PREFIX + t for t in texts], BATCH_SIZE))
    for i, batch in enumerate(batches, 1):
        vectors.extend(_embed_batch(batch))
        if progress:
            print(f"  embedded {len(vectors)}/{len(texts)} (batch {i}/{len(batches)})")
    return vectors


def embed_query(query: str) -> list[float]:
    """Embed a search query. Applies the search_query prefix."""
    return _embed_batch([QUERY_PREFIX + query])[0]


def embed_chunks(chunks: list[dict], progress: bool = False) -> list[dict]:
    """Attach an 'embedding' key to each Phase 2 chunk, preserving order."""
    vectors = embed_documents([c["text"] for c in chunks], progress=progress)
    return [{**chunk, "embedding": vec} for chunk, vec in zip(chunks, vectors)]


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

SIMILAR = (
    "The cat sat on the warm windowsill.",
    "A feline rested on the sunny ledge.",
)
UNRELATED = (
    "The cat sat on the warm windowsill.",
    "Quarterly amortisation schedules were revised by the audit committee.",
)


def _check_similarity() -> bool:
    """Roadmap check: paraphrases must score higher than unrelated sentences."""
    vectors = embed_documents([SIMILAR[0], SIMILAR[1], UNRELATED[1]])
    sim = cosine_similarity(vectors[0], vectors[1])
    dis = cosine_similarity(vectors[0], vectors[2])

    print("  paraphrase pair ..... %.4f" % sim)
    print("    %r\n    %r" % SIMILAR)
    print("  unrelated pair ...... %.4f" % dis)
    print("    %r\n    %r" % UNRELATED)

    ok = sim > dis
    print(f"\n  {'PASS' if ok else 'FAIL'}: paraphrase {'>' if ok else '<='} unrelated")
    return ok


def _check_asymmetry() -> None:
    """Show why the prefixes matter.

    Absolute cosine scores are NOT comparable between prefix pairings — changing a
    prefix shifts the whole score distribution. What matters for retrieval is the
    *margin* between a relevant passage and a distractor. This measures that margin
    with the question prefixed correctly (search_query) versus incorrectly
    (search_document), which is the mistake the model card warns about.
    """
    question = "What mechanism replaces recurrence in the Transformer?"
    relevant = "The Transformer uses multi-head self-attention instead of recurrence."
    distractor = "Training used the Adam optimiser with a warmup schedule of 4000 steps."

    docs = embed_documents([relevant, distractor])
    right = embed_query(question)                            # search_query:  (correct)
    wrong = _embed_batch([DOC_PREFIX + question])[0]         # search_document: (wrong)

    for label, qvec in (("search_query:    (correct)", right),
                        ("search_document: (wrong)  ", wrong)):
        rel = cosine_similarity(qvec, docs[0])
        dis = cosine_similarity(qvec, docs[1])
        print(f"  {label}  relevant={rel:.4f}  distractor={dis:.4f}  margin={rel - dis:+.4f}")

    print("  (compare margins, not absolute scores — prefixes shift the whole scale)")


def _selftest() -> bool:
    """Exercise the pure logic with a stub transport — no Ollama needed."""
    # Patch this module's own globals: running as __main__ means `import embed`
    # would bind a *second* module object and the stub would never be seen.
    g = globals()

    results = []

    # cosine_similarity against known values
    results.append(("cosine identical", abs(cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-6))
    results.append(("cosine orthogonal", abs(cosine_similarity([1, 0], [0, 1])) < 1e-6))
    results.append(("cosine opposite", abs(cosine_similarity([1, 0], [-1, 0]) + 1.0) < 1e-6))
    results.append(("cosine zero vector", cosine_similarity([0, 0], [1, 1]) == 0.0))

    # batching splits into the right group sizes
    groups = list(_batched(list(range(70)), 32))
    results.append(("batching", [len(g) for g in groups] == [32, 32, 6]))

    # prefixes applied, order preserved, shape validated — via a stubbed endpoint
    seen: list[str] = []

    def fake_batch(texts: list[str]) -> list[list[float]]:
        seen.extend(texts)
        return [[float(i)] * EMBED_DIM for i in range(len(texts))]

    real = g["_embed_batch"]
    g["_embed_batch"] = fake_batch
    try:
        chunks = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]
        out = embed_chunks(chunks)
        results.append(("doc prefix", seen == [DOC_PREFIX + "alpha", DOC_PREFIX + "beta"]))
        results.append(("chunk metadata kept", out[0]["id"] == "a" and out[1]["id"] == "b"))
        results.append(("vector width", all(len(c["embedding"]) == EMBED_DIM for c in out)))

        seen.clear()
        embed_query("why")
        results.append(("query prefix", seen == [QUERY_PREFIX + "why"]))

        results.append(("empty input", embed_documents([]) == []))
    finally:
        g["_embed_batch"] = real

    # a wrong-width response must be rejected, not passed downstream
    try:
        _validate([[0.0] * 42], 1)
        results.append(("rejects bad dim", False))
    except OllamaError:
        results.append(("rejects bad dim", True))

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


if __name__ == "__main__":
    import sys
    import time

    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if "--selftest" in sys.argv:
        print("Phase 3 self-test (no Ollama required)\n")
        sys.exit(0 if _selftest() else 1)

    print(f"Ollama: {OLLAMA_URL}   model: {MODEL}\n")

    try:
        print("[1] Similarity sanity check")
        passed = _check_similarity()

        print("\n[2] Prefix asymmetry")
        _check_asymmetry()

        print("\n[3] End-to-end: embed every chunk of the sample PDF")
        from chunk_pdf import chunk_pages
        from parse_pdf import parse_pdf

        pdf = sys.argv[1] if len(sys.argv) > 1 else "sample-pdfs/attention-is-all-you-need.pdf"
        chunks = chunk_pages(parse_pdf(pdf))

        start = time.perf_counter()
        embedded = embed_chunks(chunks, progress=True)
        elapsed = time.perf_counter() - start

        print(f"\n  {len(embedded)} chunks -> {EMBED_DIM}-dim vectors in {elapsed:.1f}s "
              f"({elapsed / max(len(embedded), 1) * 1000:.0f} ms/chunk)")

        # Retrieval smoke test: nearest chunk to a question answered by the PDF
        question = "What is multi-head attention?"
        qvec = embed_query(question)
        ranked = sorted(
            ((cosine_similarity(qvec, c["embedding"]), c) for c in embedded),
            key=lambda pair: pair[0],
            reverse=True,
        )
        print(f"\n  Top 3 for {question!r} (brute force — Phase 4 replaces this with SQL):")
        for score, chunk in ranked[:3]:
            preview = " ".join(chunk["text"].split())[:90]
            print(f"    {score:.4f}  p{chunk['page_number']:<3} {preview}...")

        sys.exit(0 if passed else 1)

    except OllamaError as exc:
        print(f"\nOllamaError: {exc}", file=sys.stderr)
        print("\nRun `python src/embed.py --selftest` to check the logic without Ollama.",
              file=sys.stderr)
        sys.exit(2)
