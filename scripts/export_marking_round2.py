"""Two marking sets: the TC2 consistency re-mark, and the delivered system's benchmark.

TC2 re-mark. Twenty of the 141 answers already marked on 22 September, drawn at random
and given fresh identifiers in a fresh order, so that a re-mark cannot be matched against
the first mark from the sheet. The answers come from the runs the first marking used,
which the export of 21 September fixes: the hybrid run of 21 September, and with it the
untuned and frontier runs. Scored by comparing the label each answer receives with the
label it received first; the pass mark is 17 of 20.

Delivered benchmark. The 47 answers the delivered system gave on 24 September, after the
calculator, the course facts and the two pattern repairs. These decide the two
hand-marked acceptance tests for the system as delivered, which until now have been
measured on the frozen run and its routing fault.

    python scripts/export_marking_round2.py

The key mapping the identifiers back is written beside the sets. Do not read it before
marking.
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

POOL_SEED = 4114          # the shuffle the first marking used; do not change
DRAW_SEED = 20260925      # the draw for the re-mark, recorded so it can be repeated
MARKED_HYBRID_RUN = "20260921T064410Z-C2-oss-hybrid"
DELIVERED_RUN = "20260924T144633Z-C2-oss-hybrid"
CONDITIONS = ["C2-oss-hybrid", "C2-oss", "C1-prompted"]

INSTRUCTIONS = """HOW TO MARK

For each answer below, go through the numbered points and decide whether the answer
makes that point, however briefly it puts it. Extra material around a point neither
adds to it nor takes away from it. Then decide how faithful the answer is to the course
material as a whole:

  grounded    nothing in it goes beyond what the material supports
  partly      mostly supported, with something stated more strongly than the material does
  unfaithful  it states something the material does not support, or contradicts it

Answers are in a random order and carry no indication of where they came from. Some
questions may appear more than once with different answers. Mark each answer on its own,
without looking back at what you gave another one.

HOW TO REPLY

One line per answer, in a text file, in this form:

  {first_id}: y n y n n | grounded

That is: the identifier, a colon, one mark per point in order (y or n, separated by
spaces or commas), then a vertical bar and the faithfulness word. Anything after a
second bar is treated as a note and ignored by the scoring.

Lines may be in any order, and any line that does not start with an identifier is
ignored, so notes to yourself are fine. Every answer states how many points it has;
a line with the wrong number of marks will be reported back rather than guessed at.
"""


def load_pool(runs_dir: Path, hybrid_run: str) -> dict[str, list[dict]]:
    """Every condition's answers per item, taking one named run for the hybrid arm."""
    responses: dict[str, list[dict]] = defaultdict(list)
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        condition = manifest["condition"]
        if condition not in CONDITIONS:
            continue
        if condition == "C2-oss-hybrid" and manifest_path.parent.name != hybrid_run:
            continue
        for line in (manifest_path.parent / "responses.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                responses[record["item_id"]].append(record)
    return responses


def answer_text(record: dict) -> str:
    answer = record.get("answer") or {}
    if answer.get("error"):
        return "[the call failed: %s]" % answer["error"]
    return (answer.get("text") or "").strip() or "(empty answer)"


def block(marker_id: str, item, text: str) -> list[str]:
    lines = ["=" * 78, "", "%s   (%d points to mark)" % (marker_id, len(item.must_include)), "",
             "QUESTION", item.question, "",
             "REFERENCE ANSWER", item.reference_answer or "(none recorded)", "",
             "POINTS THE ANSWER SHOULD MAKE"]
    lines += ["%d. %s" % (n, point) for n, point in enumerate(item.must_include, start=1)]
    if item.must_not_include:
        lines += ["", "THINGS THAT WOULD BE WRONG TO SAY"]
        lines += ["- %s" % bad for bad in item.must_not_include]
    lines += ["", "-" * 78, "ANSWER", "-" * 78, text, ""]
    return lines


def write_set(path: Path, title: str, preamble: str, blocks: list[tuple[str, object, str]]) -> None:
    first_id = blocks[0][0]
    lines = [title, "=" * len(title), "", preamble.rstrip(), "",
             INSTRUCTIONS.format(first_id=first_id).rstrip(), "",
             "There are %d answers below." % len(blocks), ""]
    for marker_id, item, text in blocks:
        lines += block(marker_id, item, text)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--items", default="bench/eee4114f.jsonl")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--out", default="grading/round2")
    args = parser.parse_args()

    config = load_config(args.config)
    items = load_items(args.items, known_topics=config.topic_ids)
    by_id = {item.id: item for item in items}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- TC2: reproduce the pool the first marking saw, then draw twenty answers -------
    pooled = load_pool(Path(args.runs), MARKED_HYBRID_RUN)
    rng = random.Random(POOL_SEED)
    labelled: list[tuple[str, str, dict]] = []      # (item_id, label, record)
    for item in items:
        answers = pooled.get(item.id, [])
        if not answers:
            continue
        rng.shuffle(answers)
        for index, record in enumerate(answers):
            labelled.append((item.id, chr(ord("A") + index), record))

    original_key = json.loads(Path("grading/review/_key.json").read_text(encoding="utf-8"))["labels"]
    mismatch = [f"{i}-{l}" for i, l, r in labelled
                if original_key.get(f"{i}-{l}") != r["condition"]]
    if mismatch:
        print("The reconstructed pool does not match the first marking's key:", mismatch[:5])
        return 1

    draw = random.Random(DRAW_SEED)
    sample = draw.sample(labelled, 20)
    draw.shuffle(sample)

    remark_blocks, remark_key = [], {}
    for n, (item_id, label, record) in enumerate(sample, start=1):
        marker_id = "R%02d" % n
        remark_key[marker_id] = {"item_id": item_id, "first_label": label,
                                 "condition": record["condition"]}
        remark_blocks.append((marker_id, by_id[item_id], answer_text(record)))

    write_set(out / "TC2-remark-20.txt",
              "Consistency re-mark: twenty answers",
              "These twenty answers were marked once before. Marking them again, without "
              "reference to the first marks, measures how repeatable the marking is. The "
              "identifiers and the order are new, so nothing here lines up with the first "
              "sheet.",
              remark_blocks)

    # --- The delivered system's 47 benchmark answers ---------------------------------
    delivered_path = Path(args.runs) / DELIVERED_RUN / "responses.jsonl"
    delivered = {}
    for line in delivered_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            delivered[record["item_id"]] = record

    order = [item for item in items if item.id in delivered]
    random.Random(DRAW_SEED + 1).shuffle(order)
    bench_blocks, bench_key = [], {}
    for n, item in enumerate(order, start=1):
        marker_id = "B%02d" % n
        bench_key[marker_id] = {"item_id": item.id, "condition": delivered[item.id]["condition"]}
        bench_blocks.append((marker_id, item, answer_text(delivered[item.id])))

    write_set(out / "benchmark-47-delivered.txt",
              "Benchmark marking: forty-seven answers",
              "These are one system's answers to the whole benchmark. They decide the two "
              "hand-marked acceptance tests. Unlike the first marking, there is only one "
              "system here, so there is nothing to tell apart - mark each answer against "
              "its points as it stands.",
              bench_blocks)

    (out / "_key.json").write_text(json.dumps(
        {"pool_seed": POOL_SEED, "draw_seed": DRAW_SEED,
         "marked_hybrid_run": MARKED_HYBRID_RUN, "delivered_run": DELIVERED_RUN,
         "remark": remark_key, "benchmark": bench_key}, indent=2) + "\n", encoding="utf-8")

    counts = {"R": sum(len(b[1].must_include) for b in remark_blocks),
              "B": sum(len(b[1].must_include) for b in bench_blocks)}
    print("Wrote %s (20 answers, %d points)" % (out / "TC2-remark-20.txt", counts["R"]))
    print("Wrote %s (47 answers, %d points)" % (out / "benchmark-47-delivered.txt", counts["B"]))
    print("Key written to %s - do not read it before marking." % (out / "_key.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
