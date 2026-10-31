"""Where version 2's similarity threshold comes from (E28), on development material only.

    python scripts/e28_threshold.py

Measures, against version 2's 133 references (`coursefacts.seen_wordings`):
  - each reference course wording against the other 132, the lowest of which is how far
    below a known course question the threshold must stay;
  - every development content question, every turn of the development follow-up and
    pushback conversations included, the highest of which the threshold must clear.
`coursefacts.THRESHOLD_V2` is set just above that highest. Writes
stress/results/course-facts/threshold-v2.json; the second fresh set is never read here.
"""

from __future__ import annotations

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
from ym07.calcbank import load_bank  # noqa: E402
from ym07.config import load_config  # noqa: E402
from ym07.retrieval import build_retriever  # noqa: E402
from ym07.stress import pushback_items  # noqa: E402

import e27_lookup_check as check  # noqa: E402


def development_content() -> dict[str, list[str]]:
    lines = check.lines
    follow = lines("stress/bank/followup-dev.jsonl")
    push = pushback_items(load_bank(ROOT / "stress/bank/calc-bank.jsonl"))
    return {
        "concept-dev": [r["question"] for r in lines("stress/bank/concept-dev.jsonl")],
        "calc-dev": [r["question"] for r in lines("stress/bank/calc-bank.jsonl")],
        "errors-dev": [r["message"] for r in lines("stress/bank/errors-dev.jsonl")],
        "followup-dev, every turn": [t for r in follow for t in r["turns"]],
        "pushback-dev, every turn": [t for p in push for t in p["turns"]],
        "past-question queries (E25)": [r["question"] for r in
                                        lines("stress/bank/problems/problems.jsonl")],
        "other courses": check.OUT_OF_COURSE,
        "first fresh set, content": [i["question"] for i in coursequeries.fresh_items()
                                     if i["kind"] == "content"],
    }


def main() -> int:
    config = load_config(ROOT / "configs/eee4114f.yaml")
    retriever = build_retriever(ROOT / "corpus/processed/chunks.jsonl", config.retrieval.method)
    lookup = coursefacts.build(retriever, config.handout, version=2)
    refs = coursefacts.seen_wordings()

    vectors = lookup._encoder.embed(refs)
    pairwise = vectors @ vectors.T
    pairwise.fill_diagonal_(-1.0)
    leave_one_out = pairwise.max(dim=1).values.tolist()
    low = min(range(len(refs)), key=lambda i: leave_one_out[i])
    print(f"references: {len(refs)}; lowest against the others {leave_one_out[low]:.3f} "
          f"({refs[low]!r})")

    record = {"references": len(refs), "reference_lowest": round(leave_one_out[low], 4),
              "reference_lowest_wording": refs[low], "sets": {}}
    highest, which = -1.0, ""
    for name, questions in development_content().items():
        scores = [lookup.similarity(q) for q in questions]
        top = max(range(len(scores)), key=lambda i: scores[i])
        record["sets"][name] = {"questions": len(questions), "highest": round(scores[top], 4),
                                "highest_question": questions[top]}
        print(f"{name:<30} {len(questions):>4}  highest {scores[top]:.3f}  {questions[top][:70]!r}")
        if scores[top] > highest:
            highest, which = scores[top], questions[top]
    record["content_highest"] = round(highest, 4)
    record["content_highest_question"] = which
    record["threshold"] = coursefacts.THRESHOLD_V2
    threshold = coursefacts.THRESHOLD_V2
    ok = highest < threshold
    record["clears_every_development_content_question"] = ok
    # Which references would themselves fall below it, by kind: a fact wording below it
    # would be left to the model, an absent wording below it would not be declined.
    facts = {q[k] for q in coursequeries.COURSE_QUERIES for k in ("dev", "final", "check")}
    facts |= {i["question"] for i in coursequeries.fresh_items() if i["kind"] == "fact"}
    below = [(refs[i], round(leave_one_out[i], 4), "fact" if refs[i] in facts else "absent")
             for i in range(len(refs)) if leave_one_out[i] < threshold]
    record["references_below"] = below
    lowest_fact = min(leave_one_out[i] for i in range(len(refs)) if refs[i] in facts)
    record["lowest_fact_reference"] = round(lowest_fact, 4)
    print(f"\ncontent highest {highest:.3f}; threshold {threshold}; lowest fact reference "
          f"{lowest_fact:.3f}")
    print("threshold clears every development content question:", ok)
    for wording, score, kind in below:
        print(f"   below it: {kind:<6} {score:.3f}  {wording}")

    out = ROOT / "stress/results/course-facts/threshold-v2.json"
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written to {out.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
