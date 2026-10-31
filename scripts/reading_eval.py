"""How well each engine reads a question off the screen.

The tutor can only answer the question it was given, so when the question arrives
as a picture the reading is upstream of everything else the system does: a lost
subscript is not a transcription blemish, it is a different problem, answered
confidently. This measures that step on its own, against the course's own past
papers, before any of it reaches the tutor.

Two numbers per engine.

**Character error rate** is the edit distance between the reading and the truth,
over the length of the truth. It says how much of the text survived.

**Expression accuracy** is the share of the mathematical expressions in the truth
that appear intact in the reading. It is the number that matters, and it is not
implied by the first: a page can come back with a character error rate of 5% and
still have turned every delta into a 4, because the mathematics is a small
fraction of the characters and all of the meaning.

Comparing a vision model that writes LaTeX against an engine that writes plain
characters needs care, or the metric measures notation rather than reading. Both
readings and the truth are put through the same canonical form first: LaTeX
commands become the symbols they denote, fractions become a slash, markup that
carries no meaning is dropped, and - deliberately generous to the plain-text
engine - the subscript and superscript markers themselves are removed, so that
`h_1[n]`, `h_{1}[n]` and `h1[n]` all count as the same reading and only a wrong
character counts against an engine.

    python scripts/reading_eval.py run --engine vision
    python scripts/reading_eval.py run --engine tesseract
    python scripts/reading_eval.py score
    python scripts/reading_eval.py query

In that order: `score` writes the report from scratch and `query` appends to it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The transcriptions are full of Greek and the Windows console is not.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

BANK = ROOT / "stress" / "bank" / "reading"
# Which cut of the same crops. `images` is 140 dpi, `images-screen` is 90 dpi and
# about the size a student actually screenshots; see scripts/build_reading_set.py.
IMAGES = BANK / "images"
TRUTH = BANK / "truth.jsonl"
RESULTS = ROOT / "stress" / "results" / "reading"


# --------------------------------------------------------------------------
# Canonical form
# --------------------------------------------------------------------------

# LaTeX commands become the symbol they stand for, so that a reading which writes
# the symbol directly and one which writes the command agree. Longest first, so
# that \omega_0 is not eaten by a shorter key.
_COMMANDS = {
    r"\delta": "δ", r"\Delta": "Δ", r"\omega": "ω", r"\Omega": "Ω",
    r"\theta": "θ", r"\Theta": "Θ", r"\sigma": "σ", r"\Sigma": "Σ",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\lambda": "λ",
    r"\mu": "μ", r"\phi": "φ", r"\tau": "τ", r"\rho": "ρ", r"\epsilon": "ε",
    r"\pi": "π", r"\infty": "∞",
    r"\sum": "Σ", r"\prod": "Π", r"\oint": "∮", r"\int": "∫",
    r"\partial": "∂", r"\nabla": "∇",
    r"\Longleftrightarrow": "⟺", r"\longleftrightarrow": "⟺",
    r"\Leftrightarrow": "⟺", r"\leftrightarrow": "⟺", r"\iff": "⟺",
    r"\rightarrow": "→", r"\to": "→",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠",
    r"\approx": "≈", r"\equiv": "≡", r"\pm": "±",
    r"\times": "x", r"\cdot": "*", r"\ast": "*",
    r"\cos": "cos", r"\sin": "sin", r"\tan": "tan", r"\log": "log",
    r"\ln": "ln", r"\exp": "exp", r"\sqrt": "sqrt", r"\max": "max",
    r"\min": "min", r"\arg": "arg",
}

# Markup that carries no meaning for a reader.
_NOISE = [
    r"\begin{cases}", r"\end{cases}", r"\begin{array}", r"\end{array}",
    r"\begin{matrix}", r"\end{matrix}", r"\begin{bmatrix}", r"\end{bmatrix}",
    r"\begin{aligned}", r"\end{aligned}", r"\begin{equation}", r"\end{equation}",
    r"\displaystyle", r"\textstyle", r"\limits", r"\nolimits",
    r"\left", r"\right", r"\quad", r"\qquad", r"\thinspace", r"\!",
    r"\,", r"\;", r"\:", r"\ ", r"\\", r"$$", "$",
    r"\(", r"\)", r"\[", r"\]",
]

# Characters a renderer or an engine varies without changing the mathematics.
_CHARS = {
    "\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-",
    "\u00d7": "x", "\u00b7": "*", "\u2219": "*",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2061": "", "\u200b": "",
    "\u2264": "≤", "\u2265": "≥",
}

_TEXTISH = re.compile(r"\\(?:text|mathrm|mathbf|mathit|operatorname|mbox)\s*\{")
# A description the reader was asked to write where a figure is. Scored separately
# from the words, because there is no single right wording for it.
_FIGURE_LINE = re.compile(
    r"^\s*\[(?:figure|plot|diagram|image|graph|illegible|unreadable)\b[^\]]*\]\s*$",
    re.IGNORECASE,
)


def _match_brace(text: str, start: int) -> int:
    """Index just past the `}` closing the `{` at `start`, or -1."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _expand_frac(text: str) -> str:
    """`\\frac{A}{B}` becomes `(A)/(B)`, innermost last, so nesting survives."""
    for _ in range(12):
        at = text.find("\\frac")
        if at < 0:
            break
        first = text.find("{", at)
        if first < 0:
            text = text.replace("\\frac", "/", 1)
            continue
        end_first = _match_brace(text, first)
        if end_first < 0:
            text = text.replace("\\frac", "/", 1)
            continue
        second = text.find("{", end_first)
        if second < 0 or text[end_first:second].strip():
            text = text[:at] + "(" + text[first + 1:end_first - 1] + ")/" + text[end_first:]
            continue
        end_second = _match_brace(text, second)
        if end_second < 0:
            text = text.replace("\\frac", "/", 1)
            continue
        numer = text[first + 1:end_first - 1]
        denom = text[second + 1:end_second - 1]
        text = text[:at] + "(" + numer + ")/(" + denom + ")" + text[end_second:]
    return text


def _strip_textish(text: str) -> str:
    """`\\text{otherwise}` becomes `otherwise`."""
    for _ in range(20):
        found = _TEXTISH.search(text)
        if not found:
            break
        brace = found.end() - 1
        close = _match_brace(text, brace)
        if close < 0:
            text = text[:found.start()] + text[found.end():]
            continue
        text = text[:found.start()] + text[brace + 1:close - 1] + text[close:]
    return text


def canon(text: str) -> str:
    """One form both a LaTeX reading and a plain-character reading map into."""
    text = text or ""
    for wrong, right in _CHARS.items():
        text = text.replace(wrong, right)
    text = _strip_textish(text)
    text = _expand_frac(text)
    for command in sorted(_COMMANDS, key=len, reverse=True):
        text = text.replace(command, _COMMANDS[command])
    for noise in _NOISE:
        text = text.replace(noise, "")
    # Markdown emphasis and table rules an engine adds of its own accord.
    text = re.sub(r"^[\s|:\-=+]*$", "", text, flags=re.MULTILINE)
    # The markers themselves, not the characters they lift: this is where the
    # comparison is made generous to an engine that cannot write a subscript.
    # Markdown emphasis goes; a lone asterisk stays. Stripping it too made the
    # convolution `x[n] * y[n]` identical to the product `x[n]y[n]`, which is the
    # one distinction a transform table exists to draw.
    text = text.replace("**", "")
    for drop in ("^", "_", "{", "}", "|", "&", "`", "~"):
        text = text.replace(drop, "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def strip_figures(text: str) -> str:
    """Drop the bracketed figure descriptions before comparing words."""
    return "\n".join(line for line in text.splitlines() if not _FIGURE_LINE.match(line))


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------
# Running and scoring
# --------------------------------------------------------------------------

# Flagged matches checked by hand against the reading, 2 October 2026. Each entry is
# a credit the substring test gave that the reading did not earn, and it is subtracted
# from the headline figure. The audit can only ever lower a score, never raise one.
#
# The other four flags were checked and stand: `H(z)` and `	heta_0` each appear on
# their own in the sentence around the equation, `a^n u[n]` is its own row of the
# table, and `\pi\delta(\omega + 2\pi k)` its own term in the u[n] transform.
SPURIOUS: dict[tuple[str, str], tuple[str, ...]] = {
    # The row reads `x[n]`; the `n` multiplier that makes it the frequency
    # differentiation property is gone. It scored as found because the row above
    # ends `e^{j\omega_0 n}x[n]`.
    ("vision", "info-props"): ("nx[n]",),
}


def load_truth() -> list[dict]:
    return [json.loads(line) for line in TRUTH.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_run(args) -> int:
    global IMAGES
    from ym07 import ocr

    IMAGES = BANK / args.images
    if not IMAGES.is_dir():
        print(f"no image set at {IMAGES}")
        return 1
    items = load_truth()
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "" if args.images == "images" else "-" + args.images.replace("images-", "")
    out = RESULTS / f"raw-{args.engine}{tag}.jsonl"

    # Load the weights before the clock starts. A reader that takes half a minute to
    # come up pays that once, when the server starts, not on a student's question -
    # so charging it to the first image would misreport what every later one costs.
    if args.engine == "vision":
        started = time.perf_counter()
        ocr._load_vlm()
        print(f"reader up in {time.perf_counter() - started:.1f}s")

    rows = []
    for n, item in enumerate(items, start=1):
        path = IMAGES / f"{item['id']}.png"
        print(f"[{n}/{len(items)}] {item['id']:14}", end=" ", flush=True)
        started = time.perf_counter()
        try:
            result = ocr.extract(path.read_bytes(), engine=args.engine)
            row = {"id": item["id"], "engine": result.engine, "text": result.text,
                   "latency_s": result.latency_s, "notes": result.notes}
            print(f"{result.latency_s:6.2f}s  {len(result.text):5} chars")
        except (ocr.OcrUnavailable, ocr.OcrFailed) as exc:
            row = {"id": item["id"], "engine": args.engine, "text": "",
                   "latency_s": round(time.perf_counter() - started, 2),
                   "notes": [], "error": str(exc)}
            print(f"FAILED: {exc}")
        rows.append(row)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


def _contains(haystack: str, needle: str) -> bool:
    """Whether the reading contains the expression.

    A plain substring test, because the canonical form has no whitespace left in it
    and so has no word boundaries to anchor against: requiring a non-alphanumeric
    character on each side rejects almost every true match, since an expression
    canonicalises hard up against the words around it.

    The cost is that a lost expression can score as found when it happens to be a
    fragment of a longer correct one. That is not hypothetical - the properties
    table lists `nx[n]`, the reading dropped the `n`, and the row above ends
    `e^{j\\omega_0 n}x[n]`, which contains `nx[n]`. `suspect_matches` finds every
    such case so it can be checked by hand rather than believed.
    """
    return needle in haystack


def shadowed_by(expression: str, others: list[str]) -> list[str]:
    """Other expressions on the same image that contain this one.

    The case that inflates the count is narrow: a short expression that is a proper
    substring of a longer one on the same image. If the longer one was read and the
    shorter one was not, the shorter one still scores as found. Flagging exactly
    that keeps the audit short enough to check by hand, where a general
    word-boundary rule flags nearly every true match and is worth nothing.
    """
    wanted = canon(expression)
    return [other for other in others
            if other != expression and wanted and wanted in canon(other)]


def score_one(item: dict, text: str) -> dict:
    truth_text = item["truth"]
    hypothesis = strip_figures(text) if item.get("figure") else text
    t, h = canon(truth_text), canon(hypothesis)
    distance = edit_distance(t, h)
    cer = distance / max(len(t), 1)

    every = item.get("expressions", [])
    found, missing, suspect = [], [], []
    for expression in every:
        wanted = canon(expression)
        if wanted and _contains(h, wanted):
            found.append(expression)
        else:
            missing.append(expression)
    # A short expression can score as found on the strength of a longer one that
    # contains it. Flag it only when that longer one was itself read, which is the
    # only way the credit can be spurious.
    for expression in found:
        covering = [o for o in shadowed_by(expression, every) if o in found]
        if covering:
            suspect.append((expression, covering[0]))
    total = len(found) + len(missing)
    return {"id": item["id"], "cer": cer, "chars": len(t),
            "expr_ok": len(found), "expr_total": total,
            "expr_acc": len(found) / total if total else 0.0,
            "missing": missing, "suspect": suspect}


def cmd_score(args) -> int:
    items = {item["id"]: item for item in load_truth()}
    engines, scored = [], {}
    for path in sorted(RESULTS.glob("raw-*.jsonl")):
        engine = path.stem.replace("raw-", "")
        engines.append(engine)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        scored[engine] = {}
        for row in rows:
            item = items.get(row["id"])
            if item is None:
                continue
            marks = score_one(item, row.get("text", ""))
            marks["latency_s"] = row.get("latency_s", 0.0)
            marks["error"] = row.get("error", "")
            scored[engine][row["id"]] = marks
    if not engines:
        print("no results yet - run first")
        return 1

    lines = ["# Reading the question off the screen", ""]
    lines.append(f"{len(items)} images cropped from EEE4114F past papers, class tests and the "
                 "course information sheet, transcribed by hand to give the truth.")
    lines.append("")
    lines.append("| engine | character error rate | expressions read intact | median latency |")
    lines.append("| --- | --- | --- | --- |")
    summary = {}
    for engine in engines:
        marks = list(scored[engine].values())
        cer = sum(m["cer"] for m in marks) / len(marks)
        raw = sum(m["expr_ok"] for m in marks)
        docked = sum(len(SPURIOUS.get((engine, key), ())) for key in scored[engine])
        ok = raw - docked
        total = sum(m["expr_total"] for m in marks)
        latencies = sorted(m["latency_s"] for m in marks)
        median = latencies[len(latencies) // 2]
        summary[engine] = (cer, ok, total, median, docked)
        note = f" ({raw} before the audit)" if docked else ""
        lines.append(f"| {engine} | {cer:.1%} | {ok}/{total} ({ok / max(total, 1):.1%}){note} "
                     f"| {median:.1f} s |")
    lines.append("")

    lines.append("## Per image")
    lines.append("")
    head = "| image | topic | " + " | ".join(
        f"{e} CER | {e} expressions" for e in engines) + " |"
    lines.append(head)
    lines.append("| --- | --- | " + " | ".join("--- | ---" for _ in engines) + " |")
    for key, item in items.items():
        cells = []
        for engine in engines:
            marks = scored[engine].get(key)
            if marks is None:
                cells += ["-", "-"]
            else:
                cells += [f"{marks['cer']:.1%}",
                          f"{marks['expr_ok']}/{marks['expr_total']}"]
        lines.append(f"| {key} | {item['topic']} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Matches that may be fragments")
    lines.append("")
    lines.append("Every occurrence of these sits inside a longer string, so the substring "
                 "test may be crediting an expression the reading does not actually "
                 "contain. Checked by hand; listed so the count can be checked too.")
    lines.append("")
    any_suspect = False
    for engine in engines:
        for key in items:
            marks = scored[engine].get(key)
            for expression, covering in (marks or {}).get("suspect", []):
                any_suspect = True
                lines.append(f"- {engine} / **{key}**: `{expression}` "
                             f"could be coming from `{covering}`")
    if not any_suspect:
        lines.append("None.")
    lines.append("")

    lines.append("## Expressions each engine lost")
    lines.append("")
    for engine in engines:
        lines.append(f"### {engine}")
        lines.append("")
        any_missing = False
        for key in items:
            marks = scored[engine].get(key)
            if marks and marks["missing"]:
                any_missing = True
                lines.append(f"- **{key}**")
                for expression in marks["missing"]:
                    lines.append(f"  - `{expression}`")
        if not any_missing:
            lines.append("Nothing lost.")
        lines.append("")

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = RESULTS / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"{'engine':12} {'CER':>8} {'expressions':>16} {'median':>9}")
    for engine in engines:
        cer, ok, total, median, docked = summary[engine]
        tail = f"  (-{docked} audited)" if docked else ""
        print(f"{engine:12} {cer:7.1%} {ok:7}/{total:<8} {median:8.1f}s{tail}")
    print(f"\nwrote {report.relative_to(ROOT)}")
    return 0


def cmd_query(args) -> int:
    """What the transcription costs retrieval if it is searched as LaTeX.

    The reader writes mathematics as LaTeX because that is right for the student and
    right for the model. The corpus is text pulled out of PDFs and spells
    mathematics in Unicode. Searching one against the other matches on nothing, so
    the web tutor converts the query first; this is the measurement that says
    whether that conversion is worth making.
    """
    from ym07.ocr import plain
    from ym07.retrieval import build_retriever, course_query

    rows = [json.loads(line) for line in
            (RESULTS / "raw-vision.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever = build_retriever(args.chunks)

    lines = ["## Searching with the transcription", "",
             "Total retrieval score over the top 4 passages, with the reading sent as "
             "written and after conversion to the spelling the corpus uses.", "",
             "| image | as LaTeX | converted | |", "| --- | --- | --- | --- |"]
    better = worse = level = 0
    for row in rows:
        text = row.get("text") or ""
        if not text:
            continue
        raw = sum(h.score for h in retriever.search(course_query(text, args.code), 4))
        conv = sum(h.score for h in
                   retriever.search(course_query(plain(text), args.code), 4))
        if conv > raw * 1.02:
            verdict, better = "converted", better + 1
        elif raw > conv * 1.02:
            verdict, worse = "as LaTeX", worse + 1
        else:
            verdict, level = "level", level + 1
        lines.append(f"| {row['id']} | {raw:.3f} | {conv:.3f} | {verdict} |")
        print(f"{row['id']:14} {raw:8.3f} {conv:8.3f}  {verdict}")
    lines += ["", f"Converted better on {better}, as LaTeX better on {worse}, "
                  f"level on {level}.", ""]

    report = RESULTS / "report.md"
    body = report.read_text(encoding="utf-8") if report.exists() else ""
    marker = "## Searching with the transcription"
    body = body.split(marker)[0].rstrip() + "\n\n" + "\n".join(lines)
    report.write_text(body, encoding="utf-8")
    print(f"\nconverted better on {better}, as LaTeX better on {worse}, level on {level}")
    print(f"wrote {report.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="read every image with one engine")
    p.add_argument("--engine", required=True, choices=["vision", "tesseract"])
    p.add_argument("--images", default="images",
                   help="which cut of the crops: images (140 dpi) or images-screen (90 dpi)")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("score", help="score whatever has been run")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("query", help="what the transcription costs retrieval")
    p.add_argument("--chunks", default="corpus/processed/chunks.jsonl")
    p.add_argument("--code", default="EEE4114F")
    p.set_defaults(func=cmd_query)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
