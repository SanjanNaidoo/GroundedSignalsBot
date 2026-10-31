"""Every model on the same 99 questions, in one table.

The comparison this project needs is not "how well does the delivered system do" but
"what is the binding constraint" - the hardware, the weights, the course adaptation, or
the architecture. That only reads if every column answers identical questions under
identical marking, which is what `stress/bank/arena/calc-bank.jsonl` is for: 9 items
from each of the 11 families, 33 of each phrasing, drawn from the held-back split.

Local models are run by `ym07 stress run --bank stress/bank/arena/calc-bank.jsonl`.
Frontier models, which there is no budget to call, are pasted in by hand and scored by
`scripts/frontier_paste.py` against the same rules. Both land here.

    python scripts/arena_report.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

BANK = ROOT / "stress" / "bank" / "arena" / "calc-bank.jsonl"
LOCAL = ROOT / "stress" / "results" / "arena" / "calc"
PASTED = ROOT / "stress" / "frontier" / "results"
OUT = ROOT / "stress" / "results" / "arena" / "report.md"

# How each column should be read. Anything not listed is reported without a note.
ABOUT = {
    "C2-oss-1.5b": "Qwen2.5-1.5B, size ladder",
    "C2-oss-3b": "Qwen2.5-3B, size ladder",
    "C2-oss-7b": "Qwen2.5-7B, the delivered base",
    "C2-oss-14b": "Qwen2.5-14B - does not fit the card",
    "A-qwen3-8b": "Qwen3-8B, a generation newer",
    "A-r1-distill-7b": "DeepSeek-R1-Distill-7B, reasoning-distilled",
    "A-qwen3-4b-thinking": "Qwen3-4B-Thinking, reasoning at half the size",
    "A-r1-distill-7b-k2": "the winner on two passages, not four",
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def declined(text: str) -> bool:
    """Whether a reply withheld the answer rather than getting it wrong.

    A model that says "this is assessed work, show me what you have tried" scores zero
    on every rule, exactly as a wrong answer does, and the two mean opposite things
    about capability. The first candidate run refused an aliasing calculation outright,
    so the rate is counted rather than assumed away: a column with a high refusal rate
    is not comparable with one without, whatever its score says.
    """
    from ym07.hybrid import SCAFFOLD
    from ym07.stress import DECLINE

    body = text or ""
    return bool(SCAFFOLD.search(body) or DECLINE.search(body.lower()))


def summarise(rows: list[dict], wanted: set[str]) -> dict | None:
    rows = [r for r in rows if r["id"] in wanted]
    if not rows:
        return None
    answered = [r for r in rows if not r.get("error")]
    if not answered:
        return {"n": 0, "errors": len(rows), "full": 0, "mean": 0.0, "median_s": 0.0,
                "refused": 0, "coverage": len(rows)}
    full = sum(1 for r in answered if all((r.get("per_check") or {}).values()))
    refused = sum(1 for r in answered if declined(r.get("text", "")))
    # Answers that only arrived once part of the prompt was given up. A row carries
    # `served` only when the first attempt failed, so counting them measures how often
    # this card forced the system to answer with less than it was built to use - which
    # is the difference between what the model can do and what this machine can run.
    rungs = Counter(r["served"] for r in answered if r.get("served"))
    return {
        "n": len(answered),
        "errors": len(rows) - len(answered),
        "full": full,
        "refused": refused,
        "degraded": sum(rungs.values()),
        "rungs": rungs,
        "mean": sum(r.get("score", 0.0) for r in answered) / len(answered),
        "median_s": statistics.median([r.get("latency_s", 0.0) for r in answered]),
        "coverage": len(rows),
    }


def main() -> int:
    bank = load(BANK)
    if not bank:
        print(f"no arena set at {BANK}; build it with scripts/build_arena.py")
        return 1
    wanted = {i["id"] for i in bank}
    families = sorted({i["family"] for i in bank})
    by_id = {i["id"]: i for i in bank}

    columns: list[tuple[str, list[dict]]] = []
    for path in sorted(LOCAL.glob("*.jsonl")):
        columns.append((path.stem, load(path)))
    for path in sorted(PASTED.glob("*.jsonl")):
        columns.append((path.stem + " (pasted)", load(path)))
    if not columns:
        print("no results yet")
        return 1

    lines = ["# Every model on the same questions", "",
             f"{len(bank)} calculation items - 9 from each of {len(families)} families, "
             "33 of each phrasing - drawn from the held-back split. Identical questions "
             "and identical marking in every column.", "",
             "| model | | answered | every rule satisfied | withheld | needed less context "
             "| mean score | median |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    marks: dict[str, dict] = {}
    for name, rows in columns:
        got = summarise(rows, wanted)
        if got is None:
            continue
        marks[name] = got
        note = ABOUT.get(name.replace(" (pasted)", ""), "")
        share = f"{got['full']}/{got['n']} ({got['full'] / max(got['n'], 1):.0%})"
        missing = f" · {got['errors']} failed" if got["errors"] else ""
        partial = (f" · {got['coverage']}/{len(bank)} attempted"
                   if got["coverage"] < len(bank) else "")
        held = (f"{got['refused']} ({got['refused'] / max(got['n'], 1):.0%})"
                if got["refused"] else "-")
        cut = (f"{got['degraded']} ({got['degraded'] / max(got['n'], 1):.0%})"
               if got["degraded"] else "-")
        lines.append(f"| **{name}** | {note} | {got['n']}{missing}{partial} | {share} "
                     f"| {held} | {cut} | {got['mean']:.3f} | {got['median_s']:.1f} s |")
    lines.append("")

    # Which rung answered, where any did. An answer served without its passages is not
    # the same answer as one served with them, and the count belongs beside the score
    # rather than inside it: a column with a high rate is describing the card it ran on
    # as much as the model it ran.
    if any(got["degraded"] for got in marks.values()):
        lines += ["## Where the card ran short", "",
                  "Answers that arrived only after part of the prompt was given up. The "
                  "first attempt gets four passages and the whole conversation; each "
                  "rung below that gives up more. A model absent from this table "
                  "answered everything at full context.", "",
                  "| model | rung | answers |", "| --- | --- | --- |"]
        for name, got in marks.items():
            for rung, count in got["rungs"].most_common():
                lines.append(f"| {name} | {rung} | {count} |")
        lines.append("")

    lines += ["## By family", "",
              "| family | " + " | ".join(n for n in marks) + " |",
              "| --- | " + " | ".join("---" for _ in marks) + " |"]
    for family in families:
        ids = {i for i in wanted if by_id[i]["family"] == family}
        cells = []
        for name, rows in columns:
            if name not in marks:
                continue
            got = summarise(rows, ids)
            cells.append("-" if not got or not got["n"]
                         else f"{got['full'] / got['n']:.0%}")
        lines.append(f"| {family} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## By phrasing", "",
              "| phrasing | " + " | ".join(n for n in marks) + " |",
              "| --- | " + " | ".join("---" for _ in marks) + " |"]
    for style in sorted({i["style"] for i in bank}):
        ids = {i for i in wanted if by_id[i]["style"] == style}
        cells = []
        for name, rows in columns:
            if name not in marks:
                continue
            got = summarise(rows, ids)
            cells.append("-" if not got or not got["n"]
                         else f"{got['full'] / got['n']:.0%}")
        lines.append(f"| {style} | " + " | ".join(cells) + " |")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

    width = max(len(n) for n in marks)
    print(f"{'model':{width}}  {'answered':>8}  {'all rules':>11}  {'mean':>6}  {'median':>7}")
    for name, got in marks.items():
        print(f"{name:{width}}  {got['n']:8}  "
              f"{got['full']}/{got['n']} ({got['full'] / max(got['n'], 1):.0%})".ljust(width + 26)
              + f"  {got['mean']:.3f}  {got['median_s']:6.1f}s")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
