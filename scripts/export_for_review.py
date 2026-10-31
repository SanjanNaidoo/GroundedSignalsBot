"""The marking set as text files, for a second opinion while marking.

Writes one file per benchmark item, containing the question, the reference answer, the
numbered points the marking guide asks for, and the three systems' answers under the
same A/B/C labels the browser grader shows. Each file opens with instructions asking for
a CSV back, in the columns `ym07 grade --from-csv` reads.

The labels only match the grader if both use the same shuffle, so this fixes the seed and
prints the grader command to match. Which system is A, B or C is written to a key file
that the per-item files do not contain, so pasting an item into another model does not
tell it whose answer it is looking at.

    python scripts/export_for_review.py --conditions C2-oss-hybrid C2-oss C1-prompted

Marks made with a model's help are model-assisted marks, whatever produced them, and the
report has to say so. Marking first and comparing afterwards keeps that clean, and turns
the comparison into a measurement of its own.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ym07.config import load_config          # noqa: E402
from ym07.items import load_items            # noqa: E402

SEED = 4114
HEADER = """You are helping me check my own marking of a tutor's answers. Do not tell me
which answer you think came from which system, and do not guess.

For each answer below, decide for every numbered point whether the answer makes that
point, however briefly. Elaboration beyond the point neither adds nor subtracts. Then
decide whether the answer states anything the course material does not support.

Reply with CSV only, no commentary, with this header and one row per point per answer:

item_id,label,point,made,faithfulness,note

- item_id: exactly as given below
- label: A, B or C
- point: the number of the point, 1-based
- made: yes or no
- faithfulness: grounded, partly or unfaithful - the same value repeated on each of that
  answer's rows
- note: optional, no commas

"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        default=["C2-oss-hybrid", "C2-oss", "C1-prompted"])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--items", default="bench/eee4114f.jsonl")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--out", default="grading/review")
    args = parser.parse_args()

    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)

    wanted = set(args.conditions)
    responses = defaultdict(list)
    for manifest_path in sorted(Path(args.runs).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["condition"] not in wanted:
            continue
        for line in (manifest_path.parent / "responses.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                responses[record["item_id"]].append(record)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    key, written, combined = {}, 0, []

    for item in item_set:
        pooled = responses.get(item.id, [])
        if not pooled:
            continue
        rng.shuffle(pooled)
        lines = [HEADER, f"## Item {item.id} ({item.type}, {item.topic})", "",
                 "QUESTION", item.question, "", "REFERENCE ANSWER",
                 item.reference_answer or "(none recorded)", "", "POINTS THE ANSWER SHOULD MAKE"]
        lines += [f"{n}. {point}" for n, point in enumerate(item.must_include, start=1)]
        if item.must_not_include:
            lines += ["", "THINGS THAT WOULD BE WRONG TO SAY"]
            lines += [f"- {bad}" for bad in item.must_not_include]
        for index, record in enumerate(pooled):
            label = chr(ord("A") + index)
            key[f"{item.id}-{label}"] = record["condition"]
            answer = record["answer"]
            body = f"[the call failed: {answer['error']}]" if answer.get("error") else (
                answer.get("text") or "(empty answer)")
            lines += ["", "-" * 70, f"ANSWER {label}", "-" * 70, body]
        (out / f"{item.id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        combined.append("\n".join(lines[1:]))  # the instructions are written once, at the top
        written += 1

    (out / "_key.json").write_text(json.dumps({"seed": SEED, "labels": key}, indent=2) + "\n",
                                   encoding="utf-8")

    # Everything in one file as well, for working through in one sitting. The per-item
    # files stay, because one item is the right size to read and paste at a time.
    whole = [HEADER.rstrip(), "",
             f"There are {written} items below, each with up to three answers to mark. "
             "Every item starts with a line beginning '## Item'.", "", "CONTENTS", ""]
    whole += [f"  {n}. {item_id}" for n, item_id in
              enumerate(sorted({labelled.rsplit("-", 1)[0] for labelled in key}), start=1)]
    whole += ["", "=" * 78, ""]
    whole.append(("\n\n" + "=" * 78 + "\n\n").join(combined))
    everything = out / "ALL-ITEMS.txt"
    everything.write_text("\n".join(whole) + "\n", encoding="utf-8")

    print(f"Wrote {written} item files to {out}, all of them together in {everything} "
          f"({everything.stat().st_size // 1024} KB), and the key to {out / '_key.json'}.")
    print(f"\nMark in the browser with the same shuffle:\n"
          f"  .\\.venv\\Scripts\\ym07.exe grade --web --seed {SEED} "
          f"--conditions {' '.join(args.conditions)}")
    print(f"\nRead a returned CSV back in with:\n"
          f"  .\\.venv\\Scripts\\ym07.exe grade --from-csv <file.csv> --seed {SEED} "
          f"--conditions {' '.join(args.conditions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
