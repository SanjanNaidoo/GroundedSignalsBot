"""A frontier reference line on the suite this project scores itself with.

The cross-model comparison in this report rests on 47 hand-marked answers. The
calculation suite - 264 generated items with programmatic answers and rule-based
marking - has only ever been run by the open arm, so there is no frontier number on
the benchmark the system is actually tuned against. Without one, "approaches a
frontier model" has nothing to approach.

There is no budget for API calls, so the frontier answers are fetched by hand: this
writes the questions out in batches to paste into a chat window, and reads the
replies back. The marking is the same `score_answer` the local runs use, on the same
checks, so the number that comes out sits beside them without an asterisk - the
answers arrived differently, the marking did not.

    python scripts/frontier_paste.py build --per-family 8
    # paste each batch into the model, save each reply under replies/
    python scripts/frontier_paste.py score --model deepseek-v4 --replies <dir>

What is deliberately *not* done here: nothing is re-asked, re-worded or retried. A
reply that does not parse is recorded as unparsed rather than quietly dropped, for
the same reason the batch runner records a refusal as a refusal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# The shared comparison set, built by scripts/build_arena.py. The frontier models
# answer exactly what the local ones answer; an earlier version sampled separately
# and the two overlapped on 22 items, which would have made the columns unreadable
# against each other.
BANK = ROOT / "stress" / "bank" / "arena" / "calc-bank.jsonl"
OUT = ROOT / "stress" / "frontier"
BATCH = 20

# The instruction that rides on every batch. It asks for the same thing the course
# asks a student for - the method, the arithmetic, and a final answer that can be
# read off - and it fixes the shape of the final line so the reply can be marked by
# the same rules as every local run, rather than by eye.
PREAMBLE = """You are answering questions from a 4th-year university course in
digital signal processing and machine learning (EEE4114F).

Answer every numbered question below. For each one:

- show the method and the arithmetic, briefly - a few lines, not an essay
- give every quantity the question asks for
- write sequences in the form {a, b, c, d}
- write a named quantity with an equals sign, like A = 1.5, not "A is 1.5"
- where the question asks what something is - stable or unstable, aliased or not,
  which class - say so in a full sentence, like "The system is stable because the
  pole lies inside the unit circle", not as a bare word
- end each answer with a line beginning exactly "Final answer:"

Format your reply as, for each question:

### 1
<your working>
Final answer: <the answer>

### 2
...

Do not skip any question. Do not add commentary before or after.

QUESTIONS

"""


def load_bank() -> list[dict]:
    return [json.loads(line) for line in BANK.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def chosen(items: list[dict], per_family: int) -> list[dict]:
    """A stratified sample: the same number from each family, in bank order.

    Stratified rather than random because the families differ in difficulty by a
    factor of three, and an unbalanced draw would move the headline number without
    anything about the model changing.
    """
    picked: list[dict] = []
    seen: dict[str, int] = defaultdict(int)
    for item in items:
        if seen[item["family"]] < per_family:
            picked.append(item)
            seen[item["family"]] += 1
    return picked


def cmd_build(args) -> int:
    items = load_bank() if args.per_family <= 0 else chosen(load_bank(), args.per_family)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")

    batches = [items[i:i + args.batch] for i in range(0, len(items), args.batch)]
    paste_dir = OUT / "to-paste"
    paste_dir.mkdir(exist_ok=True)
    for old in paste_dir.glob("batch-*.txt"):
        old.unlink()
    for n, batch in enumerate(batches, start=1):
        lines = [PREAMBLE]
        for index, item in enumerate(batch, start=1):
            lines.append(f"### {index}")
            lines.append(item["question"].strip())
            lines.append("")
        (paste_dir / f"batch-{n:02d}.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"{len(items)} questions, {len(batches)} batches of up to {args.batch}")
    print(f"  questions to paste : {paste_dir.relative_to(ROOT)}\\batch-01.txt ...")
    print(f"  the selection       : {(OUT / 'selection.jsonl').relative_to(ROOT)}")
    print()
    print("Paste each batch into the model, then save its whole reply as")
    print(f"  {(OUT / 'replies').relative_to(ROOT)}\\batch-01.txt  (same number)")
    return 0


_HEAD = re.compile(r"^#{1,6}\s*(\d{1,3})\s*$|^\*{0,2}(?:Question|Q)\s*(\d{1,3})\*{0,2}\s*[:.]?\s*$",
                   re.MULTILINE | re.IGNORECASE)


def split_reply(text: str) -> dict[int, str]:
    """Pull `### n` sections out of one reply. Tolerant of the model's own headings."""
    marks = [(m.start(), int(m.group(1) or m.group(2))) for m in _HEAD.finditer(text or "")]
    out: dict[int, str] = {}
    for i, (start, number) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[start:end]
        body = body.split("\n", 1)[1] if "\n" in body else ""
        out[number] = body.strip()
    return out


def cmd_score(args) -> int:
    from ym07.checks import Check, score_answer

    items = [json.loads(line) for line in
             (OUT / "selection.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    batches = [items[i:i + args.batch] for i in range(0, len(items), args.batch)]

    replies = Path(args.replies)
    rows, missing = [], []
    for n, batch in enumerate(batches, start=1):
        path = replies / f"batch-{n:02d}.txt"
        if not path.exists():
            missing += [i["id"] for i in batch]
            continue
        answers = split_reply(path.read_text(encoding="utf-8"))
        for index, item in enumerate(batch, start=1):
            text = answers.get(index, "")
            # The bank stores the rule without the item id the dataclass carries;
            # options come back from JSON as a list and the dataclass wants a tuple.
            checks = []
            for raw in item["checks"]:
                spec = dict(raw)
                spec.setdefault("item_id", item["id"])
                if isinstance(spec.get("options"), list):
                    spec["options"] = tuple(spec["options"])
                checks.append(Check(**spec))
            scored = score_answer(checks, text) if text.strip() else {
                "correctness": 0.0, "per_check": {c.label: False for c in checks}}
            rows.append({"id": item["id"], "family": item["family"], "style": item["style"],
                         "condition": args.model, "route": "frontier (pasted)",
                         "text": text, "score": scored["correctness"],
                         "per_check": scored["per_check"],
                         "error": "" if text.strip() else "no answer parsed",
                         "latency_s": 0.0, "tokens": 0, "retrieved": 0,
                         "stop_reason": "pasted"})

    out_dir = OUT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")

    answered = [r for r in rows if not r["error"]]
    full = sum(1 for r in answered if all(r["per_check"].values()))
    mean = sum(r["score"] for r in answered) / max(len(answered), 1)
    print(f"{args.model}: {len(answered)} answered of {len(rows)} selected"
          + (f", {len(missing)} batches not supplied" if missing else ""))
    print(f"  every rule satisfied : {full}/{len(answered)} = {full / max(len(answered), 1):.0%}")
    print(f"  mean score           : {mean:.3f}")
    unparsed = [r["id"] for r in rows if r["error"]]
    if unparsed:
        print(f"  unparsed             : {len(unparsed)} ({', '.join(unparsed[:6])}...)")
    print(f"\nwrote {out_path.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="write the question batches to paste")
    p.add_argument("--per-family", type=int, default=0,
                   help="re-sample N per family; 0 (the default) uses the whole "
                        "arena set, which is already balanced")
    p.add_argument("--batch", type=int, default=BATCH)
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("score", help="mark the replies with the suite's own rules")
    p.add_argument("--model", required=True, help="a name for the column, e.g. deepseek-v4")
    p.add_argument("--replies", required=True, help="directory of batch-NN.txt replies")
    p.add_argument("--batch", type=int, default=BATCH)
    p.set_defaults(func=cmd_score)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
