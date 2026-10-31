"""What the handout lookup does on its own, no model asked (E27).

    python scripts/e27_lookup_check.py dev       # development material: build against this
    python scripts/e27_lookup_check.py heldout   # pass mark F4, and F3 on the fresh set
    python scripts/e27_lookup_check.py turns     # every held-out conversation turn
    python scripts/e27_lookup_check.py dev --version 2       # E28: the first fresh set too
    python scripts/e27_lookup_check.py heldout --version 2   # E28: G4, and G3 without a model

dev
    The 23 development wordings, which the lookup should answer, each by its fact's
    registered rule; the five absent-fact questions of scripts/check_behaviour_v2.py,
    which it should leave alone; and every development question of the other suites,
    which it should leave alone.

heldout
    F4: every held-out question of the other suites (concept, calculation, errors,
    follow-up first turns, scope, benchmark), which it must leave alone. F3: the ten
    fresh content items. Run once, after the build is finished; see
    protocols/course-facts.md.

Writes stress/results/course-facts/lookup-<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07 import coursefacts, coursequeries  # noqa: E402
from ym07.config import load_config  # noqa: E402
from ym07.retrieval import about_the_course, build_retriever  # noqa: E402


def lines(path: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def first_turn(item: dict) -> str:
    turn = item["turns"][0]
    return turn if isinstance(turn, str) else turn.get("content", "")


OUT_OF_COURSE = [
    "How do I choose the duty cycle for a buck converter stepping 24 V to 5 V?",
    "Why does slip increase with load in a three-phase induction motor?",
    "How do I find the fault current for a single-line-to-ground fault?",
    "What boundary conditions do E and H satisfy between two dielectrics?",
    "How do I use a Smith chart to match a 75 ohm load to a 50 ohm line?",
    "Why does the depletion region of a pn junction widen under reverse bias?",
    "How does TCP congestion control respond to three duplicate ACKs?",
    "How do I read gain and phase margin off a Bode plot?",
    "What is the difference between an isothermal and an adiabatic process?",
    "How do I reverse a linked list in C without extra memory?",
]


def leave_alone_sets(split: str) -> dict[str, list[str]]:
    if split == "dev":
        return {
            "concept-dev": [r["question"] for r in lines("stress/bank/concept-dev.jsonl")],
            "calc-dev": [r["question"] for r in lines("stress/bank/calc-bank.jsonl")],
            "errors-dev": [r["message"] for r in lines("stress/bank/errors-dev.jsonl")],
            "followup-dev": [first_turn(r) for r in lines("stress/bank/followup-dev.jsonl")],
            "other-course": OUT_OF_COURSE,
            "absent (old five)": [r["question"] for r in lines("stress/bank/course-absent.jsonl")],
        }
    from ym07.stress import scope_items

    sets = {
        "concept-final": [r["question"] for r in lines("stress/bank/concept-final.jsonl")],
        "calc-final": [r["question"] for r in lines("stress/bank/final/calc-bank.jsonl")],
        "errors-final": [r["message"] for r in lines("stress/bank/final/errors.jsonl")],
        "followup-final": [first_turn(r) for r in lines("stress/bank/final/followup.jsonl")],
        "calc-fresh": [r["question"] for r in lines("stress/bank/fresh/calc-bank.jsonl")],
        "errors-fresh": [r["message"] for r in lines("stress/bank/fresh/errors.jsonl")],
        "followup-fresh": [first_turn(r) for r in lines("stress/bank/fresh/followup.jsonl")],
        "scope-final": [r["question"] for r in scope_items("final")],
        "F3 fresh content": [i["question"] for i in coursequeries.fresh_items()
                             if i["kind"] == "content"],
    }
    # The 47-item benchmark. Its item file only: bench/checks.jsonl carries comment lines
    # and holds rules, not questions.
    sets["benchmark"] = [r["question"] for r in lines("bench/eee4114f.jsonl")]
    return sets


def conversation_turns() -> dict[str, list[str]]:
    """Every turn of the held-out follow-up and pushback conversations.

    Added on 14 October, after the E27 run, while designing E28: the registered F4 asked
    only the first turns of the follow-up conversations and none of the pushback ones,
    and a pushback opener ("This is for my EEE4114F assignment, due tomorrow ...") reads
    as a course question by similarity. Reported beside F4, not as part of it.
    """
    from ym07.calcbank import load_bank
    from ym07.stress import pushback_items

    sets = {}
    for split in ("final", "fresh"):
        follow = lines(f"stress/bank/{split}/followup.jsonl")
        sets[f"followup-{split} all turns"] = [t for r in follow for t in r["turns"]]
        push = pushback_items(load_bank(ROOT / f"stress/bank/{split}/calc-bank.jsonl"))
        sets[f"pushback-{split} all turns"] = [t for p in push for t in p["turns"]]
    return sets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("dev", "heldout", "turns"))
    parser.add_argument("--version", type=int, default=1,
                        help="1 for E27's lookup (A-r1-facts), 2 for E28's (A-r1-facts2)")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--chunks", default="corpus/processed/chunks.jsonl")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    retriever = build_retriever(ROOT / args.chunks, config.retrieval.method)
    lookup = coursefacts.build(retriever, config.handout, version=args.version)
    print(f"handout facts read: {len(lookup.values)}; pages: "
          f"{ {k: v + 1 for k, v in lookup.page_of.items()} }\n")

    record: dict = {"split": args.split, "version": args.version, "threshold": lookup.threshold}

    if args.split == "dev":
        rows = []
        for q in coursequeries.COURSE_QUERIES:
            got = lookup.reply(q["dev"])
            ok = got is not None and coursequeries.holds(q["pattern"], got.text)
            rows.append({"id": q["id"], "question": q["dev"], "answered": got is not None,
                         "right": ok, "fields": list(got.fields) if got else [],
                         "similarity": round(lookup.similarity(q["dev"]), 3),
                         "text": got.text if got else ""})
            print(f"{'RIGHT' if ok else 'MISS '} {q['id']}  {q['dev']}")
            print(f"        {got.text if got else '(left to the model)'}")
        right = sum(r["right"] for r in rows)
        print(f"\ndevelopment wordings answered right: {right} / {len(rows)}\n")
        record["dev_wordings"] = {"right": right, "of": len(rows), "rows": rows}

    if args.split == "dev" and args.version >= 2:
        # E28 is built after reading E27's fresh set, which is development material now.
        by_kind: dict = {}
        for item in coursequeries.fresh_items():
            got = lookup.reply(item["question"])
            text = got.text if got else ""
            if item["kind"] == "content":
                ok = got is None
            elif got is None:
                ok = None  # left to the model; its answer is not known here
            else:
                ok = coursequeries.score_fresh(item, text, True)
            tally = by_kind.setdefault(item["kind"], {"items": 0, "handled": 0, "right": 0,
                                                      "left_to_model": 0, "rows": []})
            tally["items"] += 1
            tally["handled"] += got is not None
            tally["right"] += bool(ok)
            tally["left_to_model"] += got is None and item["kind"] != "content"
            tally["rows"].append({"id": item["id"], "question": item["question"],
                                  "handled": got is not None, "right": ok,
                                  "kind": got.kind if got else None, "text": text})
            mark = ("left to the model" if ok is None else "RIGHT" if ok else "WRONG")
            if item["kind"] != "content" or got is not None:
                print(f"{mark:<17} {item['id']:<11} {item['question']}")
        for kind, t in by_kind.items():
            print(f"first fresh set, {kind}: {t['items']} items, handled {t['handled']}, "
                  f"right {t['right']}, left to the model {t['left_to_model']}")
        print()
        record["first_fresh_set"] = by_kind

    fired: dict = {}
    sets = conversation_turns() if args.split == "turns" else leave_alone_sets(args.split)
    if args.version >= 2 and args.split == "dev":
        # Every turn of the development conversations, as G4 asks of the held-out ones.
        from ym07.calcbank import load_bank
        from ym07.stress import pushback_items

        follow = lines("stress/bank/followup-dev.jsonl")
        sets["followup-dev all turns"] = [t for r in follow for t in r["turns"]]
        push = pushback_items(load_bank(ROOT / "stress/bank/calc-bank.jsonl"))
        sets["pushback-dev all turns"] = [t for p in push for t in p["turns"]]
    if args.version >= 2 and args.split == "heldout":
        # G4 asks every held-out conversation turn as well; G3's content items without a
        # model. E27's fresh content items are development material for version 2.
        sets.pop("F3 fresh content", None)
        sets.update(conversation_turns())
        sets["G3 second fresh set, content"] = [i["question"] for i in coursequeries.fresh2_items()
                                                if i["kind"] == "content"]
    for name, questions in sets.items():
        hits = []
        widened = 0
        for question in questions:
            got = lookup.reply(question)
            if got is not None:
                hits.append({"question": question, "fields": list(got.fields),
                             "text": got.text})
            if not about_the_course(question) and lookup.about_the_course(question):
                widened += 1
        # How many the word list already routes to the handout, in every configuration:
        # the delivered system's own routing, beside which "newly" is counted.
        by_word_list = sum(bool(about_the_course(q)) for q in questions)
        fired[name] = {"questions": len(questions), "answered_by_lookup": len(hits),
                       "newly_called_course_questions": widened,
                       "word_list_routes_to_handout": by_word_list, "answered": hits}
        print(f"{name:<22} {len(questions):>4} questions, lookup answered {len(hits)}, "
              f"newly routed as course questions {widened}")
        for h in hits:
            print(f"      {h['fields']}  {h['question'][:100]}")
    record["leave_alone"] = fired

    name = f"lookup-{args.split}.json" if args.version == 1 else f"lookup-v{args.version}-{args.split}.json"
    out = ROOT / "stress/results/course-facts" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
