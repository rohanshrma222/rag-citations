"""Phase 2: Chunking — split page text into retrievable pieces, keeping page numbers."""

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


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prefix each chunk (after the first) with the tail of the previous chunk."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        result.append(tail + chunks[i])
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


if __name__ == "__main__":
    import os
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
