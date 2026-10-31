"""Question-and-answer pairs for fine-tuning the open arm.

`C2-oss-tuned` is Qwen2.5-7B fine-tuned on the EEE4114F corpus. The corpus is
prose, not examples, so it has to become examples first: a model reads it one
passage at a time and writes the questions a student revising the course might
ask, answered from that passage.

The default writer is the open model itself. H1 sets the tuned 7B against
C1-prompted, which is Claude, and a 7B trained on answers Claude wrote would
partly be Claude competing with itself. But two samples of 20 passages showed
where a 7B writer fails. About one pair in six was wrong, almost all of them in
DSP mathematics and worked solutions: algebra redone and slipped on, lost
diagrams replaced with invented values, and gaps filled from memory with
something false. Tightening the prompt moved individual errors around without
lowering the rate. Course prose - the ML notes - came out clean in both. So a
frontier model can be named as the writer instead, and the manifest and every
pair record which model wrote it, because the report has to say so.

`--like` reruns exactly the same passages, so a change of prompt or writer is
compared with the last one fairly.

The contamination rule in protocols/evaluation.md is enforced here twice over.
No benchmark question or reference answer is ever shown to the writer. And every
pair it writes is checked against the benchmark afterwards: it is dropped if its
question closely matches a benchmark question, or if it shares an eight-word
phrase with a benchmark question or reference answer. Dropped pairs are written
out with the reason, so the filter can be audited rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .retrieval import Chunk, dedupe_chunks, tokenize
from .runner import looks_degenerate

DEFAULT_PAIRS = Path("training/pairs.jsonl")
SAMPLE_PAIRS = Path("training/sample/pairs.jsonl")
DRY_RUN_PAIRS = Path("training/dry-run/pairs.jsonl")

PAIRS_PER_CHUNK = 3
MIN_WORDS = 80

# Provisional until checked against a real sample. Too strict, and ordinary
# revision questions are thrown away for sharing a benchmark item's vocabulary;
# too loose, and a benchmark item gets through reworded. The dropped file is
# where that is judged.
QUESTION_SIMILARITY = 0.6
SHARED_PHRASE_WORDS = 8

_ASSESSMENT = re.compile(r"exam|class.?test|prob\d|tutorial", re.IGNORECASE)
_SOLUTION = re.compile(r"sol", re.IGNORECASE)
# The course handout is administration, and its topic list - FIR and IIR design,
# wavelets - is wider than what the convener says is taught. A tutor trained on
# it would claim that scope.
_ADMINISTRATION = re.compile(r"handout", re.IGNORECASE)
_QUESTION = re.compile(r"^[ \t]*#{1,6}[ \t]*QUESTION\b[^\n]*(?:\n|$)", re.IGNORECASE | re.MULTILINE)
_ANSWER = re.compile(r"^[ \t]*#{1,6}[ \t]*ANSWER\b[^\n]*(?:\n|$)", re.IGNORECASE | re.MULTILINE)
# A malformed heading left inside the question, such as "### ANSWIS" on the line
# before the real "### ANSWER".
_STRAY_HEADING = re.compile(r"\n[ \t]*#")

# The course formula sheet, recognised by its table header or its opening line.
# Not by the phrase "information sheet": that also appears on some twenty cover
# pages that only say one is attached, above the real questions.
_SHEET = re.compile(r"transform\s+properties\s+sequences|t\s?ransforms\s+x\s?\(\s?j", re.IGNORECASE)

_FIGURE = re.compile(
    r"\b(?:from|in|given|shown in|according to|based on) the (?:sketch|figure|diagram|plot|graph)\b"
    r"|\b(?:sketch|figure|diagram|plot|graph)\s+(?:provided|given|shown|below|above)\b"
    r"|\bshown (?:below|above)\b"
    r"|\bfigure\s+\d+",
    re.IGNORECASE,
)
_MENTIONS_SOURCE = re.compile(
    r"\b(?:the|this|given) (?:passage|excerpt)\b|\bthe text\b|\bthe notes\b", re.IGNORECASE
)


def teaching_text(chunk: Chunk) -> str:
    """The part of a chunk worth writing pairs from.

    Test and exam papers end with the course formula sheet: tables of transform
    properties and pairs whose rows PDF extraction runs together into one line
    of symbols. The model misreads them - in the first sample it dropped the
    conjugate from the time-reversal row - and a pair about a table row teaches
    lookup rather than understanding. Everything from the start of the sheet
    onward is cut, which keeps a worked solution printed above an attached
    sheet.
    """
    match = _SHEET.search(chunk.text)
    return chunk.text[: match.start()] if match else chunk.text


def skip_reason(chunk: Chunk) -> str | None:
    """Why a chunk is not written from, or None if it is.

    Question papers without worked solutions are left out. A pair written from
    one would carry the writer's own solution to an exam question with nothing
    on the page to check it against, which is exactly the unverified material
    this step must not train on. Solution papers are used, and so is a question
    page that is word for word the same as a page of its solutions.
    """
    if _ADMINISTRATION.search(chunk.source):
        return "course administration"
    names = (chunk.source, *chunk.also_in)
    if all(_ASSESSMENT.search(n) and not _SOLUTION.search(n) for n in names):
        return "question paper without worked solutions"
    if len(teaching_text(chunk).split()) < MIN_WORDS:
        if _SHEET.search(chunk.text) and len(chunk.text.split()) >= MIN_WORDS:
            return "formula sheet"
        return f"shorter than {MIN_WORDS} words"
    return None


def generator_system_prompt(config) -> str:
    # A plain marked-up reply rather than JSON. JSON escapes backslashes, and
    # this material is dense with LaTeX: "\frac" parses as a form feed followed
    # by "rac", "\theta" as a tab and "heta", "\nu" as a newline. The parse
    # succeeds and the mathematics is silently corrupted.
    #
    # The rules about copying formulas exactly and leaving garbled parts out
    # come from the first sample, where most algebra errors were the writer
    # "simplifying" or continuing a derivation the page had already finished.
    #
    # The rule about restoring broken layout comes from the DeepSeek sample.
    # Told to copy formulas exactly, it copied the PDF extraction damage too -
    # subscripts on lines of their own, a fraction whose bar was lost reading
    # as a product - so exactness now applies to what a formula says, not to
    # how the extracted text happens to lay it out.
    return "\n".join([
        f"You write revision material for students of {config.code} ({config.name}).",
        "",
        "You will be given one passage of course material. Write up to "
        f"{PAIRS_PER_CHUNK} question-and-answer pairs that a student revising the "
        "course might ask and that this passage answers.",
        "",
        "Rules:",
        "- Every fact, formula and number in an answer must come from the passage. "
        "Add nothing the passage does not state, even something you believe is true.",
        "- Keep every formula and result exactly as the passage gives it. Do not "
        "simplify, rearrange or re-derive them, and do not carry a calculation past "
        "the point where the passage stops.",
        "- Write all mathematics in LaTeX. The passage was extracted from PDFs, so "
        "subscripts, superscripts and fractions are often broken across lines or run "
        "together; restore the formula the page meant rather than copying the broken "
        "layout, and if you cannot tell what a formula was, leave it out.",
        "- Where the passage works through a calculation or derivation, give its "
        "steps in the same order.",
        "- If part of the passage is garbled or ambiguous, leave that part out rather "
        "than guess what it said.",
        "- Figures, sketches and plots are not included in the passage. Do not ask "
        "about anything whose answer depends on one.",
        '- Write each answer as a tutor explaining to a student. Never mention '
        '"the passage", "the text" or "the notes".',
        "- Vary the questions: what a concept means, why a result holds, how to "
        "apply a method, a worked calculation.",
        "- A cover page, a contents page, administrative detail, or a fragment too "
        "thin to teach from gives no pairs.",
        "",
        "Reply in exactly this format, once for each pair, and with nothing else:",
        "",
        "### QUESTION",
        "<the question>",
        "### ANSWER",
        "<the answer>",
        "",
        "If the passage gives nothing to write pairs from, reply with the single "
        "word NONE.",
    ])


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for block in _QUESTION.split(text)[1:]:
        parts = _ANSWER.split(block, maxsplit=1)
        if len(parts) != 2:
            continue
        question = _STRAY_HEADING.split(parts[0])[0].strip()
        answer = parts[1].strip()
        if question and len(answer) >= 20:
            pairs.append((question, answer))
    return pairs


def _repeats_itself(text: str, width: int = 40, step: int = 5, times: int = 4) -> bool:
    """True if a stretch of text recurs many times inside one answer.

    looks_degenerate catches a line repeated down the page. This catches the
    other loop: in the first sample one answer rewrote the same fraction five
    times in a row on a single line.
    """
    windows = Counter(text[i:i + width] for i in range(0, max(0, len(text) - width), step))
    return bool(windows) and windows.most_common(1)[0][1] >= times


def pair_problem(question: str, answer: str, number: int) -> str | None:
    """A reason to drop a pair that needs no benchmark to see, or None."""
    if number > PAIRS_PER_CHUNK:
        return f"beyond the {PAIRS_PER_CHUNK}-pair limit per passage"
    if _FIGURE.search(question + " " + answer):
        return "depends on a figure the text does not contain"
    if _MENTIONS_SOURCE.search(answer):
        return "refers to its source instead of teaching"
    if _repeats_itself(answer):
        return "repeats itself"
    return None


def _vector(text: str, idf: dict[str, float], rare: float) -> dict[str, float]:
    counts = Counter(tokenize(text))
    return {term: count * idf.get(term, rare) for term, count in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    dot = sum(weight * b.get(term, 0.0) for term, weight in a.items())
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _phrases(text: str, length: int = SHARED_PHRASE_WORDS) -> set[str]:
    words = tokenize(text)
    return {" ".join(words[i:i + length]) for i in range(len(words) - length + 1)}


class BenchmarkGuard:
    """Decides whether a generated pair is too close to a benchmark item."""

    def __init__(self, items, corpus_texts: list[str]):
        documents = list(corpus_texts) + [item.question for item in items]
        df: Counter = Counter()
        for doc in documents:
            df.update(set(tokenize(doc)))
        n = len(documents)
        self._idf = {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}
        # A word found nowhere in the corpus or the benchmark is as rare as a word
        # gets, so it carries the largest weight. Weighting it at zero instead
        # would let a question full of new vocabulary look closer to a benchmark
        # item than it is.
        self._rare = math.log(n + 1) + 1.0
        self._questions = [
            (item.id, _vector(item.question, self._idf, self._rare)) for item in items
        ]
        self._phrases: dict[str, str] = {}
        for item in items:
            for text in (item.question, item.reference_answer or ""):
                for phrase in _phrases(text):
                    self._phrases.setdefault(phrase, item.id)

    def check(self, question: str, answer: str) -> str | None:
        vec = _vector(question, self._idf, self._rare)
        item_id, score = max(
            ((iid, _cosine(vec, qvec)) for iid, qvec in self._questions),
            key=lambda pair: pair[1],
        )
        if score >= QUESTION_SIMILARITY:
            return f"question similar to {item_id} ({score:.2f})"
        for text in (question, answer):
            for phrase in _phrases(text):
                if phrase in self._phrases:
                    return f"shares '{phrase}' with {self._phrases[phrase]}"
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _spread(chunks: list[Chunk], k: int) -> list[Chunk]:
    """k chunks evenly spaced through the list, so a sample crosses many files."""
    if k >= len(chunks):
        return list(chunks)
    if k <= 1:
        return chunks[:1]
    step = (len(chunks) - 1) / (k - 1)
    return [chunks[round(i * step)] for i in range(k)]


def _bucket(reason: str) -> str:
    for prefix in (
        "question similar to", "shares", "cut off", "repeats an earlier",
        "beyond the", "depends on a figure", "refers to its source", "repeats itself",
    ):
        if reason.startswith(prefix):
            return prefix
    return reason


def generate_pairs(
    *,
    config,
    chunks: list[Chunk],
    item_set,
    client,
    model_id: str,
    out_path: Path,
    max_tokens: int | None = None,
    effort: str = "n/a",
    workers: int = 1,
    sample: int | None = None,
    only_chunk_ids: list[str] | None = None,
    corpus_digest: str | None = None,
    dry_run: bool = False,
    progress=print,
) -> dict:
    """Write pairs from the corpus to `out_path`, resuming where an earlier call stopped.

    Beside `out_path` go `.dropped.jsonl` (pairs removed, with the reason),
    `.chunks.jsonl` (one line per chunk attempted, which is what makes a run
    resumable and what `only_chunk_ids` is read back from) and `.manifest.json`
    (what produced the file).

    `only_chunk_ids` targets exactly those chunks, in that order, whatever the
    current skip rules say about them. A chunk the rules now skip is recorded as
    skipped rather than silently dropped, so a comparison with an earlier run
    still accounts for every passage.

    With `workers` above one, requests overlap - worth it for a hosted writer,
    whose time is network latency - but results are still checked and written
    one chunk at a time in order, so the files come out the same either way.
    """
    out_path = Path(out_path)
    stem = out_path.with_suffix("")
    dropped_path = Path(f"{stem}.dropped.jsonl")
    ledger_path = Path(f"{stem}.chunks.jsonl")
    manifest_path = Path(f"{stem}.manifest.json")
    if max_tokens is None:
        max_tokens = config.model.local_max_new_tokens

    system = generator_system_prompt(config)
    prompt_digest = hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]

    earlier = _read_jsonl(ledger_path)
    if any(r.get("dry_run") != dry_run for r in earlier):
        raise ValueError(
            f"{ledger_path} would mix dry-run and real output. Placeholder text must "
            "never sit beside real pairs: use a different --out, or delete it."
        )
    if any(r.get("prompt_digest") != prompt_digest for r in earlier):
        raise ValueError(
            f"{ledger_path} was written under a different generator prompt, so "
            "resuming would mix pairs written to two sets of instructions. Use a "
            "different --out, or delete it."
        )
    if any(r.get("model") not in (None, model_id) for r in earlier):
        raise ValueError(
            f"{ledger_path} was written by a different model, so resuming would mix "
            "two writers in one file. Use a different --out, or delete it."
        )

    corpus = dedupe_chunks(chunks)
    guard = BenchmarkGuard(list(item_set), [c.text for c in corpus])

    skipped: Counter = Counter()
    eligible: list[Chunk] = []
    for chunk in corpus:
        reason = skip_reason(chunk)
        if reason:
            skipped[reason] += 1
        else:
            eligible.append(chunk)

    if only_chunk_ids is not None:
        by_id = {c.chunk_id: c for c in corpus}
        missing = [cid for cid in only_chunk_ids if cid not in by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} requested chunk id(s) are not in the corpus, "
                f"for example {missing[0]}."
            )
        targets = [by_id[cid] for cid in only_chunk_ids]
    elif sample:
        targets = _spread(eligible, sample)
    else:
        targets = eligible
    target_ids = {c.chunk_id for c in targets}

    # A chunk whose request failed is not done: an error is not an answer, and a
    # resumed run asks again. Skipped chunks and empty replies are outcomes, and
    # stay done.
    done = {r["chunk_id"] for r in earlier if r.get("status") != "error"}
    seen = {" ".join(r["question"].lower().split()) for r in _read_jsonl(out_path)}
    pending = [c for c in targets if c.chunk_id not in done]

    progress(
        f"{len(chunks)} chunks, {len(corpus)} after removing exact copies: "
        f"{len(eligible)} eligible; "
        + "; ".join(f"{n} skipped, {r}" for r, n in skipped.items())
        + "."
    )
    progress(
        f"Writer {model_id}. Writing from {len(targets)} chunk(s); "
        f"{len(targets) - len(pending)} already done.\n"
    )

    def ask(chunk: Chunk):
        return client.answer(
            system=system,
            user=f"PASSAGE\n\n{teaching_text(chunk).strip()}",
            model=model_id,
            max_tokens=max_tokens,
            effort=effort,
        )

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            chunk.chunk_id: pool.submit(ask, chunk)
            for chunk in pending
            if skip_reason(chunk) is None
        }
        for position, chunk in enumerate(pending, start=1):
            ledger_row = {
                "chunk_id": chunk.chunk_id,
                "model": model_id,
                "dry_run": dry_run,
                "prompt_digest": prompt_digest,
            }
            reason = skip_reason(chunk)
            if reason:
                _append(ledger_path, {
                    **ledger_row, "status": "skipped", "skip_reason": reason,
                    "pairs_kept": 0, "pairs_dropped": 0, "stop_reason": None,
                    "output_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0, "error": None,
                })
                progress(f"  [{position}/{len(pending)}] {chunk.citation()[:44]:<44} skipped: {reason}")
                continue

            answer = futures[chunk.chunk_id].result()
            text = answer.text or ""
            base = {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "also_in": list(chunk.also_in),
                "page": chunk.page,
                "topic": chunk.topic,
                "model": model_id,
            }
            kept = dropped = 0

            if answer.error:
                status = "error"
            elif looks_degenerate(text):
                status = "degenerate"
            else:
                pairs = parse_pairs(text)
                if pairs:
                    status = "pairs"
                elif text.strip().upper().startswith("NONE"):
                    status = "none"
                else:
                    status = "unparseable"
                for number, (question, reply) in enumerate(pairs, start=1):
                    record = {
                        **base,
                        "pair_id": f"{chunk.chunk_id}#{number}",
                        "question": question,
                        "answer": reply,
                    }
                    key = " ".join(question.lower().split())
                    if answer.stop_reason == "max_tokens" and number == len(pairs):
                        reason = "cut off at the generation cap"
                    else:
                        reason = pair_problem(question, reply, number)
                    if not reason and key in seen:
                        reason = "repeats an earlier question"
                    if not reason:
                        reason = guard.check(question, reply)
                    if reason:
                        _append(dropped_path, {**record, "reason": reason})
                        dropped += 1
                    else:
                        _append(out_path, record)
                        seen.add(key)
                        kept += 1

            _append(ledger_path, {
                **ledger_row,
                "status": status,
                "pairs_kept": kept,
                "pairs_dropped": dropped,
                "stop_reason": answer.stop_reason,
                "output_tokens": answer.output_tokens,
                "latency_s": answer.latency_s,
                "cost_usd": round(answer.cost_usd, 6),
                "error": answer.error,
            })
            progress(
                f"  [{position}/{len(pending)}] {chunk.citation()[:44]:<44} "
                f"{status:<11} kept {kept}  dropped {dropped}"
            )

    attempts = [r for r in _read_jsonl(ledger_path) if r["chunk_id"] in target_ids]
    latest: dict[str, dict] = {}
    for row in attempts:
        latest[row["chunk_id"]] = row
    ledger = list(latest.values())
    kept_records = [r for r in _read_jsonl(out_path) if r["chunk_id"] in target_ids]
    dropped_records = [r for r in _read_jsonl(dropped_path) if r["chunk_id"] in target_ids]
    summary = {
        "generator_model": model_id,
        "dry_run": dry_run,
        "sample": sample,
        "only_chunk_ids": len(only_chunk_ids) if only_chunk_ids is not None else None,
        "workers": workers,
        "last_started_utc": started,
        "items_digest": item_set.digest,
        "config_digest": config.digest,
        "corpus_digest": corpus_digest,
        "prompt_digest": prompt_digest,
        "system_prompt": system,
        "max_tokens": max_tokens,
        "pairs_per_chunk": PAIRS_PER_CHUNK,
        "min_words": MIN_WORDS,
        "question_similarity_threshold": QUESTION_SIMILARITY,
        "shared_phrase_words": SHARED_PHRASE_WORDS,
        "corpus_chunks": len(chunks),
        "unique_chunks": len(corpus),
        "eligible_chunks": len(eligible),
        "skipped_chunks": dict(skipped),
        "chunks_targeted": len(targets),
        "chunks_by_status": dict(Counter(r["status"] for r in ledger)),
        "pairs_kept": len(kept_records),
        "pairs_dropped": len(dropped_records),
        "dropped_by_reason": dict(Counter(_bucket(r["reason"]) for r in dropped_records)),
        "failed_attempts": sum(1 for r in attempts if r.get("status") == "error"),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in attempts),
        "cost_usd": round(sum(r.get("cost_usd") or 0.0 for r in attempts), 4),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return summary
