"""Phase 2: Chunking — split page text into retrievable pieces, keeping page numbers."""

import re

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split text on the first separator that keeps pieces under chunk_size.

    Falls through to finer-grained separators (paragraph -> line -> sentence ->
    word) only for pieces that are still too big, and hard-splits by character
    count as a last resort so nothing ever exceeds chunk_size.
    """
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0]
    remaining_seps = separators[1:]
    parts = text.split(sep)

    chunks = []
    current = ""
    for idx, part in enumerate(parts):
        piece = part + (sep if idx < len(parts) - 1 else "")
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(piece) > chunk_size:
            if remaining_seps:
                chunks.extend(_split_recursive(piece, chunk_size, remaining_seps))
            else:
                for i in range(0, len(piece), chunk_size):
                    chunks.append(piece[i:i + chunk_size])
        else:
            current = piece

    if current:
        chunks.append(current)
    return chunks


def _overlap_tail(chunk: str, overlap: int) -> str:
    """The last `overlap` characters of `chunk`, snapped to a word boundary.

    A raw `chunk[-overlap:]` slice lands wherever the character count happens to
    fall, which is usually inside a word: a chunk then begins "o make use of the
    order" instead of "To make use of the order". Retrievers mostly shrug that
    off — BM25 loses one token, and a bi-encoder pools it away — but a
    cross-encoder reads the passage as language, so a severed opening word costs
    real relevance score.

    So if the slice point is not already preceded by whitespace, advance to the
    next whitespace and start there. That drops a partial word rather than
    inventing one. If the tail has no whitespace at all (a single long token),
    return "" — better no overlap than a fragment.
    """
    if overlap <= 0 or not chunk:
        return ""

    start = max(0, len(chunk) - overlap)
    if start > 0 and not chunk[start - 1].isspace():
        boundary = re.search(r"\s", chunk[start:])
        if boundary is None:
            return ""
        start += boundary.end()
    return chunk[start:]


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prefix each chunk (after the first) with the tail of the previous chunk."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        result.append(_overlap_tail(chunks[i - 1], overlap) + chunks[i])
    return result


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 120) -> list[dict]:
    """Split page-tagged text into chunks, each carrying id, text, page_number, source."""
    chunks = []
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        raw_chunks = _split_recursive(text, chunk_size, SEPARATORS)
        raw_chunks = _add_overlap(raw_chunks, overlap)

        for j, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "id": f"{page['source']}_p{page['page_number']}_c{j}",
                "text": chunk_text,
                "page_number": page["page_number"],
                "source": page["source"],
            })
    return chunks


def selftest() -> bool:
    """Overlap and splitting behaviour. No PDF needed."""
    results = []

    # --- _overlap_tail: the mid-word bug this guards against ---
    # "To make use of the order" sliced at 20 chars lands inside "To".
    sentence = "We must inject information. To make use of the order"
    # raw slice would be 'ake use of the order' — severing "make"
    results.append(("drops severed leading word",
                    _overlap_tail(sentence, 20) == "use of the order"))
    # a slice landing on whitespace keeps the whole following word
    results.append(("keeps word when slice hits space",
                    _overlap_tail(sentence, 25) == "To make use of the order"))
    results.append(("advances past partial word",
                    _overlap_tail(sentence, 30) == "To make use of the order"))
    results.append(("slice already on boundary is kept",
                    _overlap_tail("alpha beta", 4) == "beta"))
    results.append(("whole chunk when overlap exceeds length",
                    _overlap_tail("abc", 99) == "abc"))
    results.append(("single long token yields no overlap",
                    _overlap_tail("x" + "y" * 200, 20) == ""))
    results.append(("zero overlap", _overlap_tail("alpha beta", 0) == ""))
    results.append(("empty chunk", _overlap_tail("", 10) == ""))

    # --- _add_overlap over a realistic split ---
    pieces = ["We must inject information. ", "To make use of the order of tokens."]
    joined = _add_overlap(pieces, 20)
    results.append(("first chunk untouched", joined[0] == pieces[0]))
    results.append(("second chunk carries context", "information" in joined[1]))
    results.append(("second chunk keeps its own text",
                    joined[1].endswith("To make use of the order of tokens.")))

    # --- end to end: no chunk may begin with a severed word ---
    prose = ("Positional encodings are added to the input embeddings. " * 40)
    pages = [{"page_number": 1, "source": "t.pdf", "text": prose}]
    chunks = chunk_pages(pages, chunk_size=200, overlap=40)
    words = set(prose.split())
    severed = [c for c in chunks if c["text"].split()[0].strip(".,()") not in
               {w.strip(".,()") for w in words}]
    results.append((f"no severed opening word ({len(chunks)} chunks)", not severed))
    results.append(("page number preserved", all(c["page_number"] == 1 for c in chunks)))
    results.append(("ids unique", len({c["id"] for c in chunks}) == len(chunks)))

    # Known behaviour, not a bug: overlap is prepended after splitting, so a
    # chunk may exceed chunk_size by up to `overlap`.
    longest = max(len(c["text"]) for c in chunks)
    results.append((f"length <= chunk_size + overlap (max {longest})", longest <= 200 + 40))

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


if __name__ == "__main__":
    import os
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if "--selftest" in sys.argv:
        print("Phase 2 self-test (no PDF)\n")
        sys.exit(0 if selftest() else 1)

    from parse_pdf import parse_pdf

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample-pdfs/attention-is-all-you-need.pdf"
    pages = parse_pdf(pdf_path)
    chunks = chunk_pages(pages)

    lengths = [len(c["text"]) for c in chunks]
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(f"chunk length: min={min(lengths)} max={max(lengths)} avg={sum(lengths)//len(lengths)}\n")

    print("--- Sample chunk (index 5) ---")
    sample = chunks[5]
    print(f"id: {sample['id']}")
    print(f"page_number: {sample['page_number']}")
    print(sample["text"])
