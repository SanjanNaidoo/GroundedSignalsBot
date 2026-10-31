"""What the marks rule does on its own, no model asked (E29).

    python scripts/e29_marks_check.py dev       # examples written for building, not the set
    python scripts/e29_marks_check.py heldout   # pass mark M3, and the set without a model

dev
    A handful of questions written while building, none of them in the registered set:
    the rule's reply to each, or that it leaves the question alone.

heldout
    M3: the lookup with and without the marks rule, on the 220 measured course questions
    (dev, final and check wordings, both fresh sets) and the 1,629 held-out questions and
    conversation turns; every question on which the two decide differently is listed, and
    there should be none. Then the registered marks set: which items the rule answers, with
    no model, and how its replies score.

Writes stress/results/course-facts/marks-<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07 import coursefacts, coursequeries  # noqa: E402
from ym07.config import load_config  # noqa: E402
from ym07.retrieval import build_retriever  # noqa: E402

import e27_lookup_check as check  # noqa: E402

DEV = [
    "If I get 50% for the tests and 50% for the project, what do I need in the exam to pass?",
    "test 1 70%, test 2 50%, project 60%, exam 45%: what is my final mark?",
    "I have 40% for test one and 60% for test two and 70% for the project, what do I need on the exam to pass?",
    "I got 40% on the first test, what do I need on the exam to pass?",
    "What is 20% of 80?",
    "How much is the exam worth?",
    "If my test accuracy is 90%, do I need more data?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev", "heldout"))
    args = parser.parse_args()

    config = load_config(ROOT / "configs/eee4114f.yaml")
    retriever = build_retriever(ROOT / "corpus/processed/chunks.jsonl", config.retrieval.method)
    plain = coursefacts.build(retriever, config.handout, version=1, marks=False)
    marks = coursefacts.build(retriever, config.handout, version=1, marks=True)
    record: dict = {"split": args.split}

    if args.split == "dev":
        rows = []
        for q in DEV:
            got = marks.reply(q)
            rows.append({"question": q, "kind": got.kind if got else None,
                         "text": got.text if got else ""})
            print(f"- {q}\n  -> {got.kind if got else 'left alone'}: {got.text if got else ''}\n")
        record["dev"] = rows
    else:
        measured = {
            "course questions, dev, final and check": [q[k] for q in coursequeries.COURSE_QUERIES
                                                       for k in ("dev", "final", "check")],
            "first fresh set": [i["question"] for i in coursequeries.fresh_items()],
            "second fresh set": [i["question"] for i in coursequeries.fresh2_items()],
        }
        heldout = check.leave_alone_sets("heldout")
        heldout.pop("F3 fresh content", None)
        heldout.update(check.conversation_turns())
        differ: dict = {}
        total = 0
        for name, questions in {**measured, **heldout}.items():
            changed = []
            for q in questions:
                a, b = plain.reply(q), marks.reply(q)
                if (a.text if a else None) != (b.text if b else None):
                    changed.append({"question": q, "without": a.text if a else None,
                                    "with": b.text if b else None})
            total += len(questions)
            differ[name] = {"questions": len(questions), "decided_differently": len(changed),
                            "rows": changed}
            print(f"{name:<45} {len(questions):>5} questions, the marks rule changes {len(changed)}")
        m3 = sum(v["decided_differently"] for v in differ.values())
        print(f"\nM3: questions on which the marks rule decides anything: {m3} of {total}\n")
        record["m3"] = {"questions": total, "decided_differently": m3, "sets": differ}

        rows = []
        for item in coursequeries.marks_items():
            got = marks.reply(item["question"])
            handled = got is not None
            right = coursequeries.score_marks(item, got.text if got else "", handled) if (
                handled or item["kind"] == "content") else None
            rows.append({"id": item["id"], "kind": item["kind"], "handled": handled,
                         "right_without_model": right, "text": got.text if got else ""})
            mark = "left to the model" if right is None else ("RIGHT" if right else "WRONG")
            print(f"{mark:<18} {item['id']:<13} {item['question']}")
            if got is not None:
                print(f"                   {got.text}")
        record["marks_set_without_model"] = rows

    out = ROOT / "stress/results/course-facts" / f"marks-{args.split}.json"
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
