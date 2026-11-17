"""Phase 4: Storage — persist chunks, vectors and a keyword index in one SQLite file.

Three tables, and the whole design rests on keeping them addressable by the same
integer key:

  chunks       ordinary table. Its implicit `rowid` is the canonical chunk key.
  chunks_vec   vec0 virtual table. rowid == chunks.rowid, holds the 768-dim vector.
  chunks_fts   fts5 virtual table. rowid == chunks.rowid, indexes the text.

`chunks_fts` is declared as an *external content* table (content='chunks'), so FTS5
indexes the text but does not store a second copy of it — the text it reports is read
straight out of `chunks`. That removes one of the two ways these tables can drift.
The other (a vector written under the wrong rowid) is prevented by inserting all
three rows inside a single transaction, and is detectable after the fact by
`sanity_check()`.

Chunk ids from Phase 2 are strings ("<source>_p3_c1"). They live in a UNIQUE column
so re-ingestion is idempotent, but they are NOT the join key — vec0 indexes by
integer rowid only.
"""

from __future__ import annotations

import os
import sqlite3

import sqlite_vec

from embed import EMBED_DIM

DB_PATH = os.environ.get("RAG_DB", "rag.db")

# FTS5 tokenizer: 'porter' stems, so a query for "encoding" also matches "encoded".
# unicode61 handles case folding and accents.
TOKENIZER = "porter unicode61"


class StoreError(RuntimeError):
    """Schema is missing, corrupt, or the three tables have drifted apart."""


# --------------------------------------------------------------------------- #
# connection + schema
# --------------------------------------------------------------------------- #

def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open the database with the sqlite-vec extension loaded."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the three tables if absent. Safe to call repeatedly."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT    NOT NULL UNIQUE,
            source      TEXT    NOT NULL,
            page_number INTEGER NOT NULL,
            text        TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_source_page
            ON chunks(source, page_number);
    """)

    existing = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    if "chunks_vec" not in existing:
        conn.execute(
            f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{EMBED_DIM}])"
        )

    if "chunks_fts" not in existing:
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "text, content='chunks', content_rowid='rowid', "
            f"tokenize='{TOKENIZER}')"
        )

    conn.commit()


def reset(conn: sqlite3.Connection) -> None:
    """Drop all three tables and recreate them. Destroys the index."""
    conn.executescript("""
        DROP TABLE IF EXISTS chunks_fts;
        DROP TABLE IF EXISTS chunks_vec;
        DROP TABLE IF EXISTS chunks;
    """)
    conn.commit()
    init_schema(conn)


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #

def insert_chunks(conn: sqlite3.Connection, chunks: list[dict]) -> dict:
    """Write chunks to all three tables atomically.

    Each chunk needs: id (or chunk_id), source, page_number, text, embedding.
    Chunks whose id is already stored are skipped, so re-running an ingest is a
    no-op rather than a source of duplicates.

    Returns {"inserted": n, "skipped": n}.
    """
    if not chunks:
        return {"inserted": 0, "skipped": 0}

    stored = {row["chunk_id"] for row in conn.execute("SELECT chunk_id FROM chunks")}

    inserted = skipped = 0
    try:
        with conn:  # one transaction: all three tables commit together, or none do
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id", chunk.get("id"))
                if chunk_id is None:
                    raise StoreError(f"chunk has no id: {sorted(chunk)}")
                if chunk_id in stored:
                    skipped += 1
                    continue

                vector = chunk["embedding"]
                if len(vector) != EMBED_DIM:
                    raise StoreError(
                        f"{chunk_id}: expected {EMBED_DIM}-dim vector, got {len(vector)}"
                    )

                cur = conn.execute(
                    "INSERT INTO chunks(chunk_id, source, page_number, text) "
                    "VALUES (?, ?, ?, ?)",
                    (chunk_id, chunk["source"], chunk["page_number"], chunk["text"]),
                )
                rowid = cur.lastrowid  # the key the other two tables must reuse

                conn.execute(
                    "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(vector)),
                )
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                    (rowid, chunk["text"]),
                )

                stored.add(chunk_id)
                inserted += 1
    except sqlite3.Error as exc:
        raise StoreError(f"insert failed, transaction rolled back: {exc}") from exc

    return {"inserted": inserted, "skipped": skipped}


# --------------------------------------------------------------------------- #
# reads (raw, so Phase 4 is verifiable before Phase 5 adds retrieval logic)
# --------------------------------------------------------------------------- #

def get_chunk(conn: sqlite3.Connection, rowid: int) -> dict | None:
    """Fetch one chunk's metadata by rowid."""
    row = conn.execute(
        "SELECT rowid, chunk_id, source, page_number, text FROM chunks WHERE rowid = ?",
        (rowid,),
    ).fetchone()
    return dict(row) if row else None


def keyword_search(conn: sqlite3.Connection, query: str, k: int = 5) -> list[dict]:
    """BM25 keyword search over chunks_fts, joined back to chunk metadata."""
    rows = conn.execute("""
        SELECT c.rowid, c.chunk_id, c.page_number, c.text, bm25(chunks_fts) AS score
        FROM chunks_fts f
        JOIN chunks c ON c.rowid = f.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """, (query, k)).fetchall()
    return [dict(row) for row in rows]


def vector_search(conn: sqlite3.Connection, vector: list[float], k: int = 5) -> list[dict]:
    """K-nearest-neighbour search over chunks_vec, joined back to chunk metadata."""
    rows = conn.execute("""
        SELECT c.rowid, c.chunk_id, c.page_number, c.text, v.distance
        FROM chunks_vec v
        JOIN chunks c ON c.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
    """, (sqlite_vec.serialize_float32(vector), k)).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# integrity
# --------------------------------------------------------------------------- #

def sanity_check(conn: sqlite3.Connection) -> dict:
    """Confirm the three tables agree. Raises StoreError if they have drifted."""
    counts = {
        "chunks": conn.execute("SELECT count(*) FROM chunks").fetchone()[0],
        "chunks_vec": conn.execute("SELECT count(*) FROM chunks_vec").fetchone()[0],
        "chunks_fts": conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0],
    }

    # Every chunk must have exactly one vector under the same rowid.
    orphans = conn.execute("""
        SELECT count(*) FROM chunks c
        LEFT JOIN chunks_vec v ON v.rowid = c.rowid
        WHERE v.rowid IS NULL
    """).fetchone()[0]

    # The text FTS5 reports for a rowid must be the text in `chunks`. External
    # content makes this structural, but verify rather than assume.
    mismatched = conn.execute("""
        SELECT count(*) FROM chunks c
        JOIN chunks_fts f ON f.rowid = c.rowid
        WHERE f.text IS NOT c.text
    """).fetchone()[0]

    report = {**counts, "orphan_chunks": orphans, "text_mismatches": mismatched}

    if len(set(counts.values())) != 1:
        raise StoreError(f"row counts disagree: {counts}")
    if orphans:
        raise StoreError(f"{orphans} chunks have no vector at their rowid")
    if mismatched:
        raise StoreError(f"{mismatched} rows disagree between chunks and chunks_fts")

    return report


def stats(conn: sqlite3.Connection) -> dict:
    """Summary of what is currently indexed."""
    row = conn.execute("""
        SELECT count(*) AS chunks,
               count(DISTINCT source) AS sources,
               min(page_number) AS first_page,
               max(page_number) AS last_page
        FROM chunks
    """).fetchone()
    return dict(row)
