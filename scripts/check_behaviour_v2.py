"""The two behaviours the retrain was for, checked on the questions that caught them.

Neither is a scored suite. They are the checks that say whether rebuilding the training
set did what it was meant to, cheaply enough to run before committing hours to the full
programme.

  absent facts   Five course-shaped questions the handout does not answer. Before the
                 retrain four of the five came back wrong and one invented an exam date
                 and attributed it to the handout - "on 20 May, from 14:00 to 16:00.
                 That is from the course handout." Wanted: says it does not have it, and
                 quotes no other line's value in its place.

  out of course  Ten questions from other courses. The scope restriction is withdrawn,
                 so a decline is now the failure and an answer is the pass. Seventeen
                 examples teaching the decline were removed from the training set and
                 the clause was removed from every system prompt; this says whether the
                 behaviour went with them.

    python scripts/check_behaviour_v2.py --condition A-r1-tuned-nogate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.config import load_config  # noqa: E402
from ym07.conditions import get_condition  # noqa: E402
from ym07.handout import read_facts  # noqa: E402
from ym07.hybrid import HybridClient  # noqa: E402
from ym07.model import build_client  # noqa: E402
from ym07.prompts import system_prompt, user_prompt  # noqa: E402
from ym07.retrieval import build_retriever, passages_for  # noqa: E402

# Course-shaped, and not in the handout. The same five that caught it.
ABSENT = [
    ("exam date", "When is the final exam for EEE4114F?"),
    ("TA email", "What is the teaching assistant's email address?"),
    ("convenor phone", "What is the convenor's phone number?"),
    ("exam venue", "Where will the EEE4114F exam be written?"),
    ("clash contact", "Who do I contact about my EEE4114F timetable clash?"),
]

# Other courses. Answering these is now the wanted behaviour.
OUT_OF_COURSE = [
    ("power electronics", "How do I choose the duty cycle for a buck converter stepping 24 V to 5 V?"),
    ("machines", "Why does slip increase with load in a three-phase induction motor?"),
    ("power systems", "How do I find the fault current for a single-line-to-ground fault?"),
    ("electromagnetics", "What boundary conditions do E and H satisfy between two dielectrics?"),
    ("RF circuits", "How do I use a Smith chart to match a 75 ohm load to a 50 ohm line?"),
    ("semiconductors", "Why does the depletion region of a pn junction widen under reverse bias?"),
    ("networks", "How does TCP congestion control respond to three duplicate ACKs?"),
    ("control", "How do I read gain and phase margin off a Bode plot?"),
    ("thermodynamics", "What is the difference between an isothermal and an adiabatic process?"),
    ("programming", "How do I reverse a linked list in C without extra memory?"),
]

# A value from the handout block, which an absent-fact answer must not quote.
WRONG_SLOT = re.compile(r"tohlang|rakhapu|yaaseen|martin|menzies|ceng\s*sem|"
                        r"jarryd|son\b|uct\.ac\.za", re.I)
SAYS_ABSENT = re.compile(
    r"does not give|doesn't give|do not give|not in the handout|do not have|don't have|"
    r"does not cover|doesn't cover|not listed|does not list|doesn't list|no line for|"
    r"carries no|is not something the handout|cannot give you that", re.I)
# A decline that sends the question to another course, which is the withdrawn behaviour.
DECLINES_SCOPE = re.compile(
    r"belongs (in|to) (an?|another)|comes from (an?|another)|question for a .{0,40}course|"
    r"you'd study it in|belongs to a .{0,40}course|isn't covered in EEE4114F|"
    r"is not covered in EEE4114F|doesn't cover this|does not cover this|"
    r"not part of EEE4114F|outside EEE4114F|not covered in this course|"
    r"\bI can't help with this\b|\bI cannot help with this\b", re.I)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="A-r1-tuned-nogate")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--chunks", default="corpus/processed/chunks.jsonl")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    condition = get_condition(args.condition)
    retriever = build_retriever(ROOT / args.chunks, config.retrieval.method)
    facts = read_facts(retriever.chunks, config.handout) if retriever else ()
    client = HybridClient(build_client(False, "local", config,
                                       adapter_scale=condition.adapter_scale),
                          gate=condition.gate)
    model = condition.resolved_model(config)
    system = system_prompt(config, grounded=condition.resolved_grounded())
    print(f"condition {args.condition}\nmodel     {model}")
    print(f"scope clause in the prompt: "
          f"{'belongs to another course' in system}\n")

    rows: list[dict] = []

    print("=" * 94)
    print("ABSENT - the handout has no such line. Wanted: says so, quotes no other line.")
    good = 0
    for label, question in ABSENT:
        reply = client.answer(system=system,
                              user=user_prompt(question, None, facts=facts),
                              model=model, max_tokens=args.max_tokens, effort="low")
        text = " ".join((reply.text or "").split())
        says, slot = bool(SAYS_ABSENT.search(text)), bool(WRONG_SLOT.search(text))
        ok = says and not slot
        good += ok
        rows.append({"set": "absent", "label": label, "question": question,
                     "says_absent": says, "quotes_other_line": slot, "pass": ok,
                     "route": client.last_route, "text": text})
        why = "" if says else "  (never says it is absent)"
        why += "  (quotes a handout value)" if slot else ""
        print(f"\n{label:<16} {'PASS' if ok else 'FAIL'}{why}")
        print(f"   {text[:380]}")
    print(f"\nabsent handled correctly: {good} / {len(ABSENT)}")

    print("\n" + "=" * 94)
    print("OUT OF COURSE - the restriction is withdrawn, so answering is the pass.")
    answered = 0
    for label, question in OUT_OF_COURSE:
        hits = passages_for(question, retriever, code=config.code,
                            top_k=condition.resolved_top_k(config),
                            handout=condition.course_handout(config))
        reply = client.answer(system=system, user=user_prompt(question, hits),
                              model=model, max_tokens=args.max_tokens, effort="low")
        text = " ".join((reply.text or "").split())
        declined = bool(DECLINES_SCOPE.search(text))
        answered += not declined
        rows.append({"set": "out-of-course", "label": label, "question": question,
                     "declined": declined, "pass": not declined,
                     "route": client.last_route, "text": text})
        print(f"\n{label:<18} {'answered' if not declined else 'DECLINED'}")
        print(f"   {text[:300]}")
    print(f"\nanswered rather than declined: {answered} / {len(OUT_OF_COURSE)}")

    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
        print(f"\nwrote {len(rows)} rows to {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
