"""Score a returned marking sheet from `export_marking_round2.py`.

    python scripts/score_marking_round2.py <marks.txt>

Reads lines of the form

    R01: y n y n n | grounded

and works out from the identifiers which set they belong to. R-lines are scored as the
TC2 consistency check, against the labels the same answers were given on 22 September.
B-lines decide the two hand-marked acceptance tests for the delivered system.

A label is the band an answer falls in: correct at 80% of the marking guide's points or
more, partly correct at 40% or more, incorrect below that, and capped at partly correct
when the answer is marked unfaithful. Both markings are banded the same way, so the
comparison measures the marking and not the banding.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ym07.config import load_config          # noqa: E402
from ym07.items import load_items            # noqa: E402

LINE = re.compile(r"^\s*([RB]\d{2})\s*[::]\s*([^|]+)(?:\|\s*([A-Za-z]+))?", re.I)
YES = {"y", "yes", "1", "true", "t"}
NO = {"n", "no", "0", "false", "f"}
FAITH = {"grounded", "partly", "unfaithful"}


def band(score: float, faithfulness: str) -> str:
    if score >= 0.8:
        label = "correct"
    elif score >= 0.4:
        label = "partly"
    else:
        label = "incorrect"
    if faithfulness == "unfaithful" and label == "correct":
        label = "partly"
    return label


def parse(path: Path) -> tuple[dict[str, tuple[list[bool], str]], list[str]]:
    marks, problems = {}, []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = LINE.match(raw)
        if not match:
            continue
        marker_id, body, faithfulness = match.group(1).upper(), match.group(2), match.group(3)
        tokens = [t for t in re.split(r"[,\s]+", body.strip()) if t]
        values = []
        for token in tokens:
            low = token.lower()
            if low in YES:
                values.append(True)
            elif low in NO:
                values.append(False)
            else:
                problems.append("line %d (%s): could not read the mark %r" % (number, marker_id, token))
                break
        else:
            faith = (faithfulness or "").strip().lower()
            if faith and faith not in FAITH:
                problems.append("line %d (%s): %r is not grounded, partly or unfaithful"
                                % (number, marker_id, faith))
                faith = ""
            if marker_id in marks:
                problems.append("line %d: %s appears more than once" % (number, marker_id))
            marks[marker_id] = (values, faith)
    return marks, problems


def first_marks(items_by_id) -> dict[tuple[str, str], tuple[float, str]]:
    """Score and faithfulness per (item, label) from the first marking."""
    rows = list(csv.DictReader(open("marking_results.csv", encoding="utf-8-sig")))
    per = defaultdict(lambda: {"made": 0, "points": 0, "faith": ""})
    for row in rows:
        entry = per[(row["item_id"], row["label"])]
        entry["points"] += 1
        if (row.get("made") or "").strip().lower() in YES:
            entry["made"] += 1
        faith = (row.get("faithfulness") or "").strip().lower()
        if faith:
            entry["faith"] = faith
    return {k: (v["made"] / v["points"] if v["points"] else 0.0, v["faith"])
            for k, v in per.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("marks", help="the returned text file")
    parser.add_argument("--key", default="grading/round2/_key.json")
    parser.add_argument("--items", default="bench/eee4114f.jsonl")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    args = parser.parse_args()

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    config = load_config(args.config)
    items = {item.id: item for item in load_items(args.items, known_topics=config.topic_ids)}

    marks, problems = parse(Path(args.marks))
    if not marks:
        print("No lines of the form 'R01: y n y n n | grounded' were found.")
        return 1

    # Check every answer has the right number of marks before scoring anything.
    expected = {}
    for group, entries in (("remark", key["remark"]), ("benchmark", key["benchmark"])):
        for marker_id, meta in entries.items():
            expected[marker_id] = (group, len(items[meta["item_id"]].must_include), meta)
    for marker_id, (values, _faith) in sorted(marks.items()):
        if marker_id not in expected:
            problems.append("%s is not an identifier in either set" % marker_id)
        elif len(values) != expected[marker_id][1]:
            problems.append("%s has %d marks but %d points"
                            % (marker_id, len(values), expected[marker_id][1]))
    if problems:
        print("Problems found - nothing scored:\n")
        for p in problems:
            print("  " + p)
        return 1

    # ---- TC2 -------------------------------------------------------------------
    remark = {k: v for k, v in marks.items() if k.startswith("R")}
    if remark:
        before = first_marks(items)
        same, rows = 0, []
        for marker_id in sorted(remark):
            values, faith = remark[marker_id]
            meta = key["remark"][marker_id]
            now_score = sum(values) / len(values)
            now = band(now_score, faith)
            was_score, was_faith = before[(meta["item_id"], meta["first_label"])]
            was = band(was_score, was_faith)
            agree = now == was
            same += agree
            rows.append((marker_id, was, now, "same" if agree else "CHANGED"))
        print("TC2, consistency re-mark")
        print("  %-5s %-10s %-10s %s" % ("id", "22 Sep", "today", ""))
        for row in rows:
            print("  %-5s %-10s %-10s %s" % row)
        n = len(rows)
        print("\n  %d of %d answers kept the same label (pass mark 17 of 20) - %s"
              % (same, n, "PASS" if same >= 17 else "MISS"))
        if n < 20:
            print("  note: only %d of the 20 were marked" % n)
        print()

    # ---- the delivered system's benchmark ---------------------------------------
    bench = {k: v for k, v in marks.items() if k.startswith("B")}
    if bench:
        scores, faiths = [], defaultdict(int)
        for marker_id, (values, faith) in bench.items():
            scores.append(sum(values) / len(values))
            faiths[faith or "unrecorded"] += 1
        mean = sum(scores) / len(scores)
        unfaithful = faiths.get("unfaithful", 0)
        n = len(scores)
        print("Benchmark, delivered system (%d of 47 answers marked)" % n)
        print("  AT2  mean hand-marked correctness   %.3f   (pass mark 0.75)  %s"
              % (mean, "pass" if mean >= 0.75 else "MISS"))
        print("  AT8  marked unfaithful              %d/%d = %.0f%%   (pass mark <= 10%%)  %s"
              % (unfaithful, n, 100 * unfaithful / n,
                 "pass" if unfaithful / n <= 0.10 else "MISS"))
        print("  answers at 80%% of points or more    %d/%d" % (sum(1 for s in scores if s >= 0.8), n))
        print("  faithfulness                        " +
              ", ".join("%s %d" % (k, v) for k, v in sorted(faiths.items())))
        print("\n  For comparison, the frozen run of 21 September scored 0.598 with 9 unfaithful.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
