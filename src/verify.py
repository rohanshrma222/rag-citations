"""Phase 8: Grounding verification — does each cited source actually support its claim?

Phase 7 guarantees that a citation marker points at a real chunk. It does not
guarantee that the chunk says what the sentence claims. A small model will happily
attach [2] to a sentence that [2] does not support, because the citation is a
formatting habit rather than a retrieval of evidence.

So this is a second pass, and deliberately a *narrow* one: one claim, one source,
one yes/no question, nothing else in the prompt. Asking a model to answer and
critique itself in a single shot gives it room to rationalise; asking "does this
text support this sentence" is a judgement it can make much more reliably.

Verdicts per sentence:
  supported     at least one cited source was judged to support it
  unsupported   it cited sources, and every one was judged NOT to support it
  uncited       it makes a claim with no citation marker at all

`uncited` matters as much as `unsupported`. Phase 7's prompt demands a citation on
every factual sentence, so a bare factual sentence is an instruction failure and an
ungrounded claim at the same time.

Cost: one model call per (sentence, citation) pair. On CPU that roughly doubles to
triples end-to-end latency, so verification is opt-in rather than automatic — see
the roadmap's own note about running it always versus as a background check.
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate
import store
from embed import OllamaError, _post

VERIFY_MODEL = os.environ.get("VERIFY_MODEL", generate.CHAT_MODEL)
VERIFY_TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "300"))

VERIFY_SYSTEM = """You are a strict fact-checker. You are given one SOURCE text and
one CLAIM. Decide whether the SOURCE contains enough information to support the
CLAIM.

Answer with exactly one word: YES or NO.

Say YES only if the SOURCE itself states or directly implies the CLAIM. Say NO if
the CLAIM adds facts the SOURCE does not contain, contradicts the SOURCE, or is
merely on a similar topic. Do not use outside knowledge. Do not explain."""

VERIFY_USER = """SOURCE:
{source}

CLAIM: {claim}

Does the SOURCE support the CLAIM? Answer YES or NO."""

# Split on sentence-ending punctuation followed by whitespace and either a capital
# letter or an opening citation bracket. Requiring whitespace after the period keeps
# decimals intact ("28.4 BLEU", "step_num-0.5"), and requiring a capital or bracket
# next avoids splitting at "Fig. 2" or "et al. 2017".
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

# A sentence with no letters (a bare formula or stray punctuation) is not a claim.
_HAS_WORDS = re.compile(r"[A-Za-z]{2,}")


# --------------------------------------------------------------------------- #
# sentence handling
# --------------------------------------------------------------------------- #

def split_sentences(answer: str) -> list[str]:
    """Split an answer into sentences, dropping fragments that assert nothing."""
    parts = _SENTENCE_SPLIT.split(answer.strip())
    return [p.strip() for p in parts if p.strip() and _HAS_WORDS.search(p)]


def claim_text(sentence: str) -> str:
    """The sentence with its citation markers removed.

    The markers are our scaffolding, not part of the claim, and leaving them in
    invites the verifier to reason about bracket numbers instead of the facts.
    """
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", sentence).strip()


# --------------------------------------------------------------------------- #
# the verifier call
# --------------------------------------------------------------------------- #

def _ask(source_text: str, claim: str, model: str = VERIFY_MODEL) -> dict:
    """One yes/no grounding judgement."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": VERIFY_USER.format(
                source=" ".join(source_text.split()), claim=claim)},
        ],
        "stream": False,
        # num_predict is tiny on purpose: we want one word, and capping it stops a
        # chatty model from writing an essay we would then have to parse.
        "options": {"temperature": 0.0, "num_predict": 5},
    }
    status, body = _post("/api/chat", payload, timeout=VERIFY_TIMEOUT)
    if status != 200:
        raise OllamaError(f"/api/chat returned {status}: {body.get('error', body)}")

    raw = body.get("message", {}).get("content", "").strip()
    return {"supported": parse_verdict(raw), "raw": raw}


def parse_verdict(raw: str) -> bool | None:
    """Read YES/NO out of the model's reply. None if it answered neither.

    Checks for a leading yes/no first, since "NO, the source only mentions..."
    contains both words and position is what disambiguates it.
    """
    text = raw.strip().upper()
    if not text:
        return None

    # Match whole words only. A substring search reads "NOT", "NONE" and
    # "NOTHING" as "NO", so "I am not sure" would come back as a confident NO
    # instead of the unparseable reply it is.
    words = re.findall(r"[A-Z]+", text)
    if words:
        if words[0] == "YES":
            return True
        if words[0] == "NO":
            return False

    # Fall back to whichever whole word appears first anywhere in the reply.
    yes = re.search(r"\bYES\b", text)
    no = re.search(r"\bNO\b", text)
    if not yes and not no:
        return None
    if not no:
        return True
    if not yes:
        return False
    return yes.start() < no.start()


# --------------------------------------------------------------------------- #
# verification over a Phase 7 result
# --------------------------------------------------------------------------- #

def verify_answer(result: dict, model: str = VERIFY_MODEL) -> dict:
    """Check every cited sentence of a Phase 7 answer against its sources.

    Takes the dict `generate.answer_question` returns. Adds no network calls for
    a refused answer — there is nothing to ground.
    """
    started = time.perf_counter()
    answer = result.get("answer", "")
    chunks = result.get("chunks", [])

    if result.get("refused") or not answer:
        return {"sentences": [], "supported": 0, "unsupported": 0, "uncited": 0,
                "spurious": [], "calls": 0, "elapsed": 0.0,
                "grounded_ratio": None, "skipped": "answer was a refusal"}

    cache: dict[tuple[str, int], dict] = {}
    sentences, calls = [], 0

    for sentence in split_sentences(answer):
        markers = generate.extract_citations(sentence)
        claim = claim_text(sentence)
        checks = []

        for marker in markers:
            if not 1 <= marker <= len(chunks):
                checks.append({"marker": marker, "supported": None,
                               "raw": "out of range", "page_number": None})
                continue

            chunk = chunks[marker - 1]
            key = (claim, marker)
            if key not in cache:
                cache[key] = _ask(chunk["text"], claim, model=model)
                calls += 1
            checks.append({**cache[key], "marker": marker,
                           "page_number": chunk["page_number"]})

        if not markers:
            verdict = "uncited"
        elif any(check["supported"] for check in checks):
            verdict = "supported"
        else:
            verdict = "unsupported"

        sentences.append({"sentence": sentence, "claim": claim,
                          "markers": markers, "checks": checks,
                          "verdict": verdict})

    counts = {name: sum(1 for s in sentences if s["verdict"] == name)
              for name in ("supported", "unsupported", "uncited")}
    checked = len(sentences)

    # A sentence citing [1][2] is marked supported when either source backs it,
    # which is the right verdict for the claim but hides the other citation. Count
    # those separately: a marker whose source does not support the sentence is a
    # spurious citation, and it points a reader at the wrong page even when the
    # claim itself is sound.
    spurious = [
        {"marker": check["marker"], "page_number": check["page_number"],
         "claim": item["claim"]}
        for item in sentences
        for check in item["checks"]
        if check["supported"] is False and item["verdict"] == "supported"
    ]

    return {
        "sentences": sentences,
        **counts,
        "spurious": spurious,
        "calls": calls,
        "elapsed": time.perf_counter() - started,
        "grounded_ratio": counts["supported"] / checked if checked else None,
    }


def answer_and_verify(conn, question: str, **kwargs) -> dict:
    """Phase 7 answer plus its Phase 8 grounding report."""
    result = generate.answer_question(conn, question, **kwargs)
    result["verification"] = verify_answer(result)
    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

MARK = {"supported": "OK  ", "unsupported": "FAIL", "uncited": "BARE"}


def report(result: dict) -> None:
    """Print a Phase 7 answer with its grounding verdicts."""
    generate.report(result)
    verification = result.get("verification")
    if not verification:
        return

    if verification.get("skipped"):
        print(f"\n  verification skipped: {verification['skipped']}")
        return

    print(f"\n  --- grounding ---")
    for item in verification["sentences"]:
        print(f"  [{MARK[item['verdict']]}] {item['claim'][:66]}")
        for check in item["checks"]:
            state = {True: "supports", False: "DOES NOT support",
                     None: "unparseable"}[check["supported"]]
            page = f"p{check['page_number']}" if check["page_number"] else "?"
            print(f"         [{check['marker']}] {page} {state}"
                  f"  (said {check['raw'][:18]!r})")

    ratio = verification["grounded_ratio"]
    print(f"\n  grounded {verification['supported']}/{len(verification['sentences'])}"
          f"{f' ({ratio:.0%})' if ratio is not None else ''}"
          f" | unsupported {verification['unsupported']}"
          f" | uncited {verification['uncited']}")
    for item in verification.get("spurious", []):
        print(f"  ~~ spurious citation [{item['marker']}] -> p{item['page_number']}: "
              f"claim is grounded elsewhere, but this source does not support it")
    print(f"  {verification['calls']} verifier calls, {verification['elapsed']:.1f}s")


def selftest() -> bool:
    """Sentence splitting, verdict parsing and aggregation. No model calls."""
    results = []
    g = globals()

    # --- sentence splitting ---
    answer = "Adam was used [1]. The warmup was 4000 steps [2]."
    results.append(("splits two sentences", len(split_sentences(answer)) == 2))
    results.append(("keeps decimals intact",
                    len(split_sentences("The score was 28.4 BLEU [1].")) == 1))
    results.append(("does not split at Fig. 2",
                    len(split_sentences("See Fig. 2 for details [1].")) == 1))
    results.append(("splits before a citation-led sentence",
                    len(split_sentences("First claim [1]. [2] shows more.")) == 2))
    # A bare marker left on its own by a split asserts nothing and must be dropped.
    results.append(("drops wordless fragments",
                    split_sentences("Real claim [1]. [2]") == ["Real claim [1]."]))
    results.append(("empty answer", split_sentences("") == []))

    # --- marker stripping ---
    results.append(("strips single marker", claim_text("Adam was used [1].") == "Adam was used."))
    results.append(("strips adjacent markers",
                    claim_text("Both hold [1][2].") == "Both hold."))
    results.append(("strips comma group", claim_text("Both [1, 2].") == "Both."))
    results.append(("leaves text without markers",
                    claim_text("No citation here.") == "No citation here."))

    # --- verdict parsing ---
    results.append(("plain yes", parse_verdict("YES") is True))
    results.append(("plain no", parse_verdict("NO") is False))
    results.append(("lowercase", parse_verdict("yes") is True))
    results.append(("trailing punctuation", parse_verdict("NO.") is False))
    results.append(("leading no wins over later yes",
                    parse_verdict("NO, the source does not say yes to this") is False))
    results.append(("leading yes", parse_verdict("Yes, it does") is True))
    results.append(("unparseable", parse_verdict("I am not sure") is None))
    results.append(("NOT is not NO", parse_verdict("It is not clear") is None))
    results.append(("NOTHING is not NO", parse_verdict("Nothing to report") is None))
    results.append(("NONE is not NO", parse_verdict("None of the above") is None))
    results.append(("empty reply", parse_verdict("") is None))

    # --- aggregation, with the model call stubbed ---
    chunks = [
        {"id": "a", "source": "a.pdf", "page_number": 3, "text": "Adam was used.",
         "rowid": 1},
        {"id": "b", "source": "a.pdf", "page_number": 7, "text": "Dropout was 0.1.",
         "rowid": 2},
    ]
    asked = []

    def fake_ask(source_text, claim, model=None):
        asked.append((source_text, claim))
        # Source 1 supports anything about Adam; source 2 supports nothing here.
        return {"supported": "Adam" in source_text and "Adam" in claim,
                "raw": "YES" if "Adam" in source_text and "Adam" in claim else "NO"}

    real = g["_ask"]
    g["_ask"] = fake_ask
    try:
        verified = verify_answer({
            "answer": "Adam was used [1]. Dropout was 0.5 [2]. Unsourced claim.",
            "chunks": chunks, "refused": False,
        })
        verdicts = [s["verdict"] for s in verified["sentences"]]
        results.append(("classifies supported", verdicts[0] == "supported"))
        results.append(("classifies unsupported", verdicts[1] == "unsupported"))
        results.append(("classifies uncited", verdicts[2] == "uncited"))
        results.append(("counts", (verified["supported"], verified["unsupported"],
                                   verified["uncited"]) == (1, 1, 1)))
        results.append(("grounded ratio", abs(verified["grounded_ratio"] - 1 / 3) < 1e-9))
        results.append(("one call per cited pair", verified["calls"] == 2))
        results.append(("markers stripped before asking",
                        all("[" not in claim for _, claim in asked)))
        results.append(("no spurious when all cited sources agree",
                        verified["spurious"] == []))

        # Over-citation: [1] supports the claim, [2] does not. The sentence is
        # still grounded, but [2] misdirects a reader to the wrong page.
        over = verify_answer({"answer": "Adam was used [1][2].", "chunks": chunks,
                              "refused": False})
        results.append(("over-citation still supported",
                        over["sentences"][0]["verdict"] == "supported"))
        results.append(("over-citation reported as spurious",
                        [s["marker"] for s in over["spurious"]] == [2]))

        # a refusal must not be verified at all
        asked.clear()
        skipped = verify_answer({"answer": generate.NO_ANSWER, "chunks": chunks,
                                 "refused": True})
        results.append(("skips refusals", skipped["calls"] == 0 and asked == []))

        # identical (claim, marker) pairs are cached, not re-asked
        asked.clear()
        dup = verify_answer({"answer": "Adam was used [1]. Adam was used [1].",
                             "chunks": chunks, "refused": False})
        results.append(("caches duplicate pairs", dup["calls"] == 1))

        # an out-of-range marker is recorded, never sent to the model
        asked.clear()
        oor = verify_answer({"answer": "Claimed [9].", "chunks": chunks,
                             "refused": False})
        results.append(("out-of-range not asked", oor["calls"] == 0))
        results.append(("out-of-range is unsupported",
                        oor["sentences"][0]["verdict"] == "unsupported"))
    finally:
        g["_ask"] = real

    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return all(ok for _, ok in results)


def fake_citation_test(conn, model: str = VERIFY_MODEL) -> bool:
    """The roadmap's check: a deliberately mis-cited claim must be caught.

    Builds a result by hand — a real chunk from the index, paired with a claim that
    chunk does not support — and confirms the verifier returns NO.
    """
    row = conn.execute(
        "SELECT rowid, chunk_id, source, page_number, text FROM chunks "
        "WHERE text LIKE '%Adam%' LIMIT 1"
    ).fetchone()
    if row is None:
        print("  no suitable chunk found — run `python src/ingest.py` first")
        return False

    chunk = dict(row)
    page = chunk["page_number"]
    print(f"  source chunk: p{page} {' '.join(chunk['text'].split())[:62]}...")

    cases = [
        # (claim, should the verifier say supported?)
        ("The Adam optimizer was used for training [1].", True),
        ("The model was trained on 4096 TPUv4 chips for three weeks [1].", False),
        ("The authors used the SGD optimizer with momentum 0.99 [1].", False),
        ("The capital of Australia is Canberra [1].", False),
    ]

    ok = True
    for claim, expected in cases:
        verified = verify_answer({"answer": claim, "chunks": [chunk],
                                  "refused": False}, model=model)
        item = verified["sentences"][0]
        got = item["verdict"] == "supported"
        verdict = "PASS" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  {verdict}  expected {'supported' if expected else 'unsupported'
                                       }, got {item['verdict']:<12} "
              f"({item['checks'][0]['raw'][:14]!r})  {claim[:44]!r}")
    return ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if "--selftest" in sys.argv:
        print("Phase 8 self-test (no model calls)\n")
        sys.exit(0 if selftest() else 1)

    conn = store.connect()
    store.init_schema(conn)
    if store.stats(conn)["chunks"] == 0:
        print("index is empty — run `python src/ingest.py` first")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    try:
        if "--fake" in sys.argv:
            print("Mis-citation test: claims deliberately not supported by their source\n")
            sys.exit(0 if fake_citation_test(conn) else 1)
        elif args:
            report(answer_and_verify(conn, " ".join(args)))
        else:
            for question in generate.ANSWERABLE[:3]:
                report(answer_and_verify(conn, question))
            for question in generate.UNANSWERABLE[:1]:
                report(answer_and_verify(conn, question))
    except OllamaError as exc:
        print(f"\nOllamaError: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        conn.close()
