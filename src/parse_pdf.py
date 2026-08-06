"""Phase 1: PDF ingestion — extract page-tagged text from a PDF."""

import fitz  # PyMuPDF


def parse_pdf(path: str) -> list[dict]:
    """Extract text from a PDF, one entry per page, keeping page numbers.

    Returns a list of dicts: {page_number, text, source}, where page_number
    is 1-indexed to match what a human sees when reading the PDF.
    """
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_number": i + 1,
            "text": text,
            "source": path,
        })
    doc.close()
    return pages


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample-pdfs/attention-is-all-you-need.pdf"
    pages = parse_pdf(pdf_path)

    print(f"Parsed {len(pages)} pages from {pdf_path}\n")

    empty_pages = [p["page_number"] for p in pages if not p["text"].strip()]
    if empty_pages:
        print(f"WARNING: empty text on pages {empty_pages} (possibly scanned/image content)\n")

    # Print first ~300 chars of page 1 so you can eyeball correctness
    first = pages[0]
    print(f"--- Page {first['page_number']} preview ---")
    print(first["text"][:300])
