# Production RAG with Citations — Implementation Roadmap

Stack: LangGraph + SQLite-vec + cross-encoder reranker + local Ollama models.

Goal: PDF Q&A bot that cites page numbers, using hybrid search (vector + BM25),
cross-encoder reranking, and grounding verification.

---

## Architecture Overview

```
PDF -> Parse (page-aware) -> Chunk -> Embed -> Store (SQLite-vec + FTS5)
                                                    |
Query -> [Vector Search + BM25 Search] -> RRF Fusion -> Cross-Encoder Rerank
                                                    |
                              Top-K chunks -> LLM (grounded prompt) -> Answer + Citations
                                                    |
                                    Grounding Verifier (optional but recommended)
```

---

## Phase 1: PDF Ingestion

**Goal:** Extract text from a PDF while keeping page numbers attached to every chunk of text.

**What to build:** A function that takes a PDF path and returns a list of
`{page_number, text}` per page.

**How to approach it:**
- Install `PyMuPDF` and read its docs for `page.get_text()`.
- Loop through pages, extract text, store page number (1-indexed, matches what a human sees).
- Test on 2-3 different PDFs: a clean text-based one, one with tables, and if you have it,
  a scanned one — see what breaks.

**Verify it works:** Print `pages[10]["text"]` and manually open the PDF to page 11
(0-indexed) — confirm the text matches.

**Pitfall to watch for:** Scanned PDFs return empty strings. Decide now whether you're
handling OCR or just excluding scanned docs from v1 — don't get stuck on this early.

---

## Phase 2: Chunking

**Goal:** Split page text into smaller retrievable pieces without losing the page number.

**What to build:** A function that takes your page list and returns chunks, each carrying
`text`, `page_number`, `source`, and a unique `id`.

**How to approach it:**
- Start naive: fixed character-length chunks with ~15% overlap. Don't overthink chunk
  strategy yet.
- Look at `RecursiveCharacterTextSplitter` from `langchain_text_splitters` for a ready-made
  splitter, or write your own splitter on `\n\n` / `\n` / sentence boundaries — writing it
  yourself will teach you more about the tradeoffs.
- Decide your chunk_size experimentally: print a few chunks and read them — do they read
  as coherent thoughts, or is text getting butchered mid-sentence?

**Verify it works:** Pick 5 random chunks, print them with their page number, and confirm
they're readable standalone.

**Pitfall to watch for:** If a logical section spans two pages, decide up front how you'll
cite it (e.g., cite the page where the chunk *starts*). Keep this rule consistent.

---

## Phase 3: Embeddings

**Goal:** Turn each chunk's text into a vector using your local Ollama embedding model.

**What to build:** A function that batches your chunk texts through Ollama and returns
vectors.

**How to approach it:**
- `ollama pull nomic-embed-text`, then hit it via `langchain_ollama.OllamaEmbeddings` or
  Ollama's raw REST API (`/api/embeddings`) — try the raw API once so you understand what's
  happening under the hood before hiding it behind a library.
- Check the model card for `nomic-embed-text` — it expects a `"search_document: "` prefix
  for indexed text and `"search_query: "` for queries. This is a common gotcha, don't skip it.

**Verify it works:** Embed two clearly similar sentences and two clearly unrelated ones,
compute cosine similarity by hand (numpy), confirm similar ones score higher.

**Pitfall to watch for:** Embedding one-at-a-time over hundreds of chunks is slow — batch
requests if the API supports it, or at least parallelize with a thread pool.

---

## Phase 4: Storage (SQLite-vec + FTS5)

**Goal:** Persist chunks, their vectors, and a keyword index, so you can query without
re-embedding everything each run.

**What to build:** A SQLite schema with three pieces: a metadata table (`chunks`), a
`vec0` virtual table for embeddings, and an `fts5` virtual table for keyword search.

**How to approach it:**
- Read the `sqlite-vec` README directly — it's short and the API (`vec0`, `MATCH`,
  `serialize_float32`) is unusual if you've only used regular SQL before.
- Write the schema yourself rather than copy-pasting — you'll hit the "how do I keep three
  tables in sync by the same id" problem, which is the actual design decision here.
- Write an `insert_chunk()` function that writes to all three tables atomically (same
  transaction).

**Verify it works:** Insert 5 known chunks, then run a raw SQL query against `chunks_fts`
for a keyword you know is in one of them — confirm it comes back.

**Pitfall to watch for:** `rowid` alignment between the three tables is the most common
source of subtle bugs here — if they drift, your search results will point to the wrong
text. Write a sanity-check function that joins all three by id and confirms row counts
match.

---

## Phase 5: Hybrid Search

**Goal:** Given a query, retrieve candidates using both vector similarity and keyword
(BM25) search, then merge the two ranked lists into one.

**What to build:** Two separate search functions (vector, FTS5), plus a fusion function
combining their rankings.

**How to approach it:**
- Get vector-only search working and tested first. Then get FTS5-only search working and
  tested. Only then combine — debugging both at once is harder.
- For fusion, implement **Reciprocal Rank Fusion** yourself (it's ~5 lines:
  `score += 1 / (k + rank)` for each list, k≈60) rather than trying to normalize raw
  distance/BM25 scores, which live on incompatible scales.
- Test with a query where you know the answer is a rare technical term (BM25 should win)
  and one that's paraphrased (vector should win) — confirm hybrid beats either alone.

**Verify it works:** Run the same query through vector-only, FTS-only, and hybrid — print
top 5 of each side by side and eyeball the difference.

**Pitfall to watch for:** SQLite's default FTS5 tokenizer is basic (no stemming). If your
keyword recall feels weak, look into the `porter` tokenizer option before assuming your
fusion logic is broken.

---

## Phase 6: Cross-Encoder Reranking

**Goal:** Re-score your ~20 hybrid candidates with a model that looks at query+chunk
together, and keep only the true top 5.

**What to build:** A function that takes the query and candidate list, scores each pair,
and returns them sorted.

**How to approach it:**
- Install `sentence-transformers`, load `cross-encoder/ms-marco-MiniLM-L-6-v2` (small,
  fast, CPU-friendly).
- Read the model card to understand it takes `[query, passage]` pairs and outputs a
  relevance score — no prefix tricks needed here, that's an embedding-model-specific quirk
  from Phase 3.
- Time it — this is your slowest step per-query, so know your latency budget (should be
  well under a second for ~20 candidates).

**Verify it works:** Take a query where the hybrid top-1 result is actually *not* the best
answer (you'll need to hunt for one), and confirm reranking fixes the ordering.

**Pitfall to watch for:** Reranking more than ~30-50 candidates gets slow with no accuracy
benefit — tune your hybrid search's `k` down if reranking becomes your bottleneck.

---

## Phase 7: Grounded Generation

**Goal:** Feed the top reranked chunks to your local LLM with a prompt that forces it to
cite sources by number, then map those numbers back to real page citations.

**What to build:** A prompt template, an Ollama chat call, and a post-processing function
that extracts `[n]` markers and resolves them to `(source, page)`.

**How to approach it:**
- `ollama pull llama3.1:8b` (or similar instruct model).
- Write the prompt yourself — the key design choice is numbering each source chunk
  explicitly and instructing "cite only using these numbers, say you don't know if the
  sources don't answer it." Iterate on wording by testing against 3-4 questions where you
  know the right answer.
- Use `temperature=0` — for a citation task you want determinism, not creativity.
- Write a regex to pull `\[(\d+)\]` out of the model's answer and map back to your chunk
  list.

**Verify it works:** Ask a question you know the PDF answers, confirm the citation numbers
point at the *correct* page, not just *a* page. Then ask a question the PDF does NOT
answer, and confirm the model says so instead of guessing.

**Pitfall to watch for:** Small local models are more prone to citing plausibly rather
than correctly. Don't skip the "ask something unanswerable" test — that's your
hallucination canary.

---

## Phase 8: Grounding Verification

**Goal:** Catch cases where the LLM cited a source that doesn't actually support its claim.

**What to build:** A second, smaller LLM call (or the same model) that checks each cited
sentence against its cited chunk and returns yes/no.

**How to approach it:**
- Split the answer into sentences, find which have a `[n]` citation, and for each, ask the
  model a strict yes/no "does this source support this claim" question.
- This is intentionally a separate, narrower prompt from generation — narrow prompts are
  more reliable than asking one model to both answer and self-critique in one shot.

**Verify it works:** Manually construct a fake case where a claim doesn't match its cited
chunk, run it through your verifier, confirm it catches it.

**Pitfall to watch for:** This roughly doubles your latency (or worse if done
per-sentence). Decide if you're running it always, or only as a background quality
check/logging feature for v1.

---

## Phase 9: LangGraph Orchestration

**Goal:** Replace your linear `retrieve -> rerank -> generate -> verify` function calls
with an explicit graph that gives you conditional branches and clean state.

**What to build:** A `TypedDict` state, one node function per phase (you already wrote the
logic in Phases 5-8 — nodes just wrap them), and edges connecting them.

**How to approach it:**
- Do this refactor *last*, once each phase works standalone and you trust it. LangGraph is
  orchestration, not logic — don't debug your retrieval quality inside the graph.
- Read LangGraph's quickstart for `StateGraph`, `add_node`, `add_edge`,
  `add_conditional_edges`.
- Once linear works, add ONE conditional edge as practice: route to a "no answer found"
  node if your top fusion score is below a threshold, instead of always calling the LLM.

**Verify it works:** Run `app.invoke({"query": ...})` end to end, and separately trace
through LangSmith or simple print-logging at each node to confirm state flows correctly.

**Pitfall to watch for:** Don't over-engineer the graph with retries/loops before the
linear path is solid — get a straight line working first.

---

## Phase 10: Evaluation

**Goal:** Know objectively whether your system is any good, not just vibes-based testing.

**What to build:** A test set of ~20 questions with known correct page numbers, and a
script that runs each through your pipeline and checks: was the correct page in the top-k
retrieved? Was it in the final citations?

**How to approach it:** Build this by hand from your own test PDF — write questions you
know the answer to and note the page. Don't reach for `ragas` or automated eval frameworks
until you've done this manually once; it'll teach you what "good" looks like for your
specific documents.

**Verify it works:** You'll get a recall number (e.g., 17/20 correct page retrieved).
Track it as you tune chunk size, k, and reranker choice — this is what tells you if a
change actually helped.

---

## Suggested Pacing

Do Phases 1-4 first and stop — get a PDF fully indexed and query-able with raw SQL before
writing any retrieval logic. That's your foundation; everything else is comparatively
quick once it's solid.

1. Steps 2-3 (parse + chunk) — verify you can open a PDF and get clean page-tagged chunks
2. Steps 4-5 (embed + store) — get one PDF fully indexed, sanity-check with raw SQL queries
3. Step 6 (hybrid search) — test retrieval quality manually before adding LLM anything
4. Step 7 (rerank) — compare top-5 before/after reranking on a few queries
5. Step 8 (generate) — get citations working end-to-end
6. Step 9 (grounding check) — add once the happy path works
7. Step 10 (LangGraph) — refactor into the graph once each piece is proven standalone
8. Steps 11-12 (eval + serve)

Start with a single PDF and a handful of test questions before scaling to a corpus — most
of the hard bugs (page misalignment, chunk boundary issues, citation index drift) show up
immediately at small scale and are much harder to spot once you have hundreds of chunks.
