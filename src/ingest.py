"""Phase 4 entry point: parse -> chunk -> embed -> store, for one or more PDFs.

    python src/ingest.py                         # index the sample PDF
    python src/ingest.py path/to/a.pdf b.pdf     # index specific PDFs
    python src/ingest.py --reset                 # wipe the index first
    python src/ingest.py --verify                # integrity + raw SQL checks only
    python src/ingest.py --selftest              # schema checks with fake vectors,
                                                 # no Ollama and no PDF required
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import store
from chunk_pdf import chunk_pages
from embed import EMBED_DIM, OllamaError, embed_chunks, embed_query
from parse_pdf import parse_pdf

SAMPLE_PDF = "sample-pdfs/attention-is-all-you-need.pdf"


def ingest(pdf_paths: list[str], db_path: str = store.DB_PATH,
           wipe: bool = False) -> dict:
    """Index PDFs end to end. Returns insert totals."""
    conn = store.connect(db_path)
    store.init_schema(conn)
    if wipe:
        store.reset(conn)
        print("index reset\n")

    totals = {"inserted": 0, "skipped": 0}
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"SKIP (not found): {path}")
            continue

        start = time.perf_counter()
        pages = parse_pdf(path)
        chunks = chunk_pages(pages)
        print(f"{path}\n  {len(pages)} pages -> {len(chunks)} chunks -> embedding...")

        embedded = embed_chunks(chunks)
        result = store.insert_chunks(conn, embedded)
        elapsed = time.perf_counter() - start

        print(f"  inserted {result['inserted']}, skipped {result['skipped']} "
              f"(already indexed)  [{elapsed:.1f}s]\n")
        totals["inserted"] += result["inserted"]
        totals["skipped"] += result["skipped"]

    conn.close()
    return totals


def verify(db_path: str = store.DB_PATH, use_ollama: bool = True) -> bool:
    """Roadmap verification: integrity, then raw SQL against each index."""
    conn = store.connect(db_path)
    store.init_schema(conn)

    print("[1] Integrity — three tables aligned by rowid")
    try:
        report = store.sanity_check(conn)
    except store.StoreError as exc:
        print(f"  FAIL: {exc}")
        return False
    for key, value in report.items():
        print(f"  {key:<16} {value}")
    print(f"  PASS: row counts match, no orphans, no text drift")

    print(f"\n  stats: {store.stats(conn)}")

    if report["chunks"] == 0:
        print("\n  index is empty — run `python src/ingest.py` first")
        return False

    print("\n[2] Raw SQL keyword search (BM25 over chunks_fts)")
    for term in ("attention", "encoder", "positional encoding"):
        hits = store.keyword_search(conn, term, k=3)
        print(f"  MATCH {term!r} -> {len(hits)} hits")
        for hit in hits:
            preview = " ".join(hit["text"].split())[:70]
            print(f"    bm25={hit['score']:7.3f}  p{hit['page_number']:<3} {preview}...")

    print("\n[3] Porter stemming (the FTS5 tokenizer pitfall)")
    for term in ("encode", "encoded", "encoding", "encodings"):
        hits = store.keyword_search(conn, term, k=50)
        print(f"  MATCH {term!r:<12} -> {len(hits):>2} hits")
    print("  all four should return hits — that is the stemmer working")

    ok = True
    if use_ollama:
        print("\n[4] Raw SQL vector search (KNN over chunks_vec)")
        try:
            question = "What is multi-head attention?"
            hits = store.vector_search(conn, embed_query(question), k=3)
            print(f"  {question!r}")
            for hit in hits:
                preview = " ".join(hit["text"].split())[:70]
                print(f"    dist={hit['distance']:.4f}  p{hit['page_number']:<3} {preview}...")

            print("\n[5] Both indexes agree on rowid identity")
            vec_top = hits[0]
            roundtrip = store.get_chunk(conn, vec_top["rowid"])
            same = roundtrip["text"] == vec_top["text"]
            print(f"  rowid {vec_top['rowid']} -> chunks.text matches vector hit: {same}")
            ok = same
        except OllamaError as exc:
            print(f"  SKIP: {exc.args[0].splitlines()[0]}")

    conn.close()
    return ok


def selftest() -> bool:
    """Schema-level checks with deterministic fake vectors. No Ollama, no PDF."""
    import tempfile

    results = []
    tmpdir = tempfile.mkdtemp()
    db = os.path.join(tmpdir, "selftest.db")

    conn = store.connect(db)
    store.init_schema(conn)

    def fake(i: int, text: str, page: int) -> dict:
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0           # one-hot: nearest neighbour is predictable
        return {"id": f"doc_p{page}_c{i}", "source": "doc.pdf", "page_number": page,
                "text": text, "embedding": vec}

    corpus = [
        fake(0, "The quick brown fox jumps over the lazy dog.", 1),
        fake(1, "Positional encodings are added to the input embeddings.", 2),
        fake(2, "Multi-head attention projects queries, keys and values.", 3),
        fake(3, "We trained on the WMT 2014 English-German dataset.", 4),
        fake(4, "Layer normalisation is applied after each sub-layer.", 5),
    ]

    result = store.insert_chunks(conn, corpus)
    results.append(("inserts 5 chunks", result == {"inserted": 5, "skipped": 0}))

    report = store.sanity_check(conn)
    results.append(("three tables aligned", report["chunks"] == report["chunks_vec"] == report["chunks_fts"] == 5))
    results.append(("no orphans", report["orphan_chunks"] == 0))
    results.append(("no text drift", report["text_mismatches"] == 0))

    # roadmap verify step: FTS query for a keyword known to be in exactly one chunk
    hits = store.keyword_search(conn, "WMT", k=5)
    results.append(("fts finds known keyword", len(hits) == 1 and hits[0]["page_number"] == 4))

    # porter stemming: query a different inflection than the indexed text
    results.append(("porter stems", len(store.keyword_search(conn, "encoding", k=5)) >= 1))

    # vector search returns the one-hot chunk it should
    probe = [0.0] * EMBED_DIM
    probe[2] = 1.0
    vhits = store.vector_search(conn, probe, k=1)
    results.append(("knn finds nearest", vhits and vhits[0]["page_number"] == 3))

    # rowid identity holds across all three tables
    results.append(("rowid roundtrip",
                    store.get_chunk(conn, vhits[0]["rowid"])["text"] == vhits[0]["text"]))

    # Both retrievers must return every field downstream phases read. Phase 7
    # needs `source` to resolve a citation to a file; it was missing from these
    # SELECTs and hand-made test fixtures hid it, so assert the real shape here.
    required = {"rowid", "chunk_id", "source", "page_number", "text"}
    results.append(("vector_search returns full chunk shape",
                    required <= set(vhits[0].keys())))
    results.append(("keyword_search returns full chunk shape",
                    required <= set(hits[0].keys())))

    # re-inserting the same chunks must skip, not duplicate
    again = store.insert_chunks(conn, corpus)
    results.append(("idempotent re-ingest", again == {"inserted": 0, "skipped": 5}))
    results.append(("still 5 rows", store.sanity_check(conn)["chunks"] == 5))

    # a bad vector width must abort the whole transaction, leaving no partial row
    bad = fake(9, "This should never be stored.", 9)
    bad["embedding"] = [0.1] * 10
    try:
        store.insert_chunks(conn, [bad])
        results.append(("rejects wrong dim", False))
    except store.StoreError:
        results.append(("rejects wrong dim", True))
    results.append(("rollback left no partial row", store.sanity_check(conn)["chunks"] == 5))

    # persistence: close, reopen, data still queryable without re-embedding
    conn.close()
    conn2 = store.connect(db)
    results.append(("survives reopen", store.sanity_check(conn2)["chunks"] == 5))
    results.append(("queryable after reopen",
                    len(store.keyword_search(conn2, "WMT", k=5)) == 1))

    # reset wipes all three
    store.reset(conn2)
    results.append(("reset empties all three", store.sanity_check(conn2)["chunks"] == 0))
    conn2.close()

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if "--selftest" in flags:
        print("Phase 4 self-test (no Ollama, no PDF)\n")
        sys.exit(0 if selftest() else 1)

    if "--verify" in flags:
        sys.exit(0 if verify() else 1)

    try:
        pdfs = args or [SAMPLE_PDF]
        print(f"db: {store.DB_PATH}\n")
        totals = ingest(pdfs, wipe="--reset" in flags)
        print(f"total: {totals}\n")
        print("=" * 70)
        sys.exit(0 if verify() else 1)
    except OllamaError as exc:
        print(f"\nOllamaError: {exc}", file=sys.stderr)
        print("\nRun `python src/ingest.py --selftest` to check the schema without Ollama.",
              file=sys.stderr)
        sys.exit(2)
