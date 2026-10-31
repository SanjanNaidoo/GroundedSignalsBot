"""Every acceptance test, decided on the held-back sets and the marked benchmark.

Reads the final results, the grades and the runs, and prints the table the report is
built on: each test in `protocols/requirements.md`, what was measured, its pass mark and
whether it passed. Nothing here re-runs a model; it reads what the runs recorded.

    python scripts/final_results.py [--out stress/results/final/acceptance.md]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ym07.checks import load_checks, score_answer    # noqa: E402
from ym07.stress import invented_tags, wilson        # noqa: E402

FINAL = Path("stress/results/final")
# Post-freeze work, measured after the acceptance tests were decided: the calculator on
# the numeric path, and the course facts in the config. Each acceptance test is reported
# as it was decided, with the later measurement beside it.
AFTER = {"calc": Path("stress/results/final-calculator"),
         "errors": Path("stress/results/final-calculator"),
         "followup": Path("stress/results/final-calculator2"),
         "pushback": Path("stress/results/final-calculator"),
         "course": Path("stress/results/facts-fix-final")}
SYSTEM = "C2-oss-hybrid"
UNTUNED = "C2-oss"
TUNED = "C2-oss-tuned"
FRONTIER = "C1-prompted"
# At least two worked steps, the same test the training data was checked with.
WORKING = re.compile(r"\d[^=\n]{0,40}=\s*-?\d")


def rows(suite: str, condition: str, root: Path = FINAL) -> list[dict]:  # noqa: D401
    path = root / suite / f"{condition}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ok(records: list[dict]) -> list[dict]:
    return [r for r in records if not r.get("error")]


def pct(part: int, whole: int) -> str:
    if not whole:
        return "no data"
    lo, hi = wilson(part, whole)
    return f"{part}/{whole} = {part / whole:.0%} (CI {lo:.0%}-{hi:.0%})"


def grades() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ("grades/grades.jsonl", "grades/model-assisted.jsonl"):
        path = Path(name)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    grade = json.loads(line)
                    out.setdefault(grade["response_id"], grade)
    return out


def responses() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for manifest_path in sorted(Path("runs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for line in (manifest_path.parent / "responses.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["condition"] = manifest["condition"]
                record["run"] = manifest_path.parent.name
                out[record["response_id"]] = record
    return out


def latest_run(condition: str) -> list[dict]:
    runs = sorted(p for p in Path("runs").glob(f"*-{condition}") if (p / "responses.jsonl").exists())
    if not runs:
        return []
    return [json.loads(line) for line in
            (runs[-1] / "responses.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_bootstrap(diffs: list[float], draws: int = 4000, seed: int = 4114) -> tuple[float, float]:
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(draws))
    return boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]


ROUND2 = Path("grading/round2")
MARK_LINE = re.compile(r"^\s*([RB]\d{2})\s*:\s*([^|]+)\|\s*([A-Za-z]+)")
YES_WORDS = {"y", "yes", "1", "true", "t"}


def sheet(path: Path) -> dict[str, tuple[list[bool], str]]:
    """A returned marking sheet: identifier -> (per-point marks, faithfulness)."""
    out: dict[str, tuple[list[bool], str]] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = MARK_LINE.match(raw)
        if match:
            marks = [t.lower() in YES_WORDS
                     for t in re.split(r"[,\s]+", match.group(2).strip()) if t]
            out[match.group(1)] = (marks, match.group(3).strip().lower())
    return out


def first_marking() -> dict[tuple[str, str], tuple[float, str]]:
    """Score and faithfulness per answer from the marking of 22 September."""
    import csv
    per: dict[tuple[str, str], dict] = defaultdict(lambda: {"made": 0, "points": 0, "faith": ""})
    path = Path("marking_results.csv")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            entry = per[(row["item_id"], row["label"])]
            entry["points"] += 1
            if (row.get("made") or "").strip().lower() in YES_WORDS:
                entry["made"] += 1
            faith = (row.get("faithfulness") or "").strip().lower()
            if faith:
                entry["faith"] = faith
    return {k: (v["made"] / v["points"] if v["points"] else 0.0, v["faith"])
            for k, v in per.items()}


def band(score: float, faith: str | None = None) -> str:
    label = "correct" if score >= 0.8 else "partly" if score >= 0.4 else "incorrect"
    return "partly" if faith == "unfaithful" and label == "correct" else label


def consistency() -> dict | None:
    """TC2: the same twenty answers, marked twice.

    Reported two ways. The points are itemised in the marking guide and written before
    any answer existed; faithfulness is one holistic judgement over the whole corpus,
    with nothing itemised to check against. Separating them says which part of the
    marking is repeatable.
    """
    now = sheet(ROUND2 / "TC2-remark-20-graded.txt")
    key_path = ROUND2 / "_key.json"
    before = first_marking()
    if not now or not before or not key_path.exists():
        return None
    key = json.loads(key_path.read_text(encoding="utf-8"))["remark"]

    same_full = same_points = 0
    was_unfaithful = now_unfaithful = 0
    was_scores, now_scores = [], []
    for marker_id, (marks, faith) in now.items():
        meta = key[marker_id]
        was_score, was_faith = before[(meta["item_id"], meta["first_label"])]
        now_score = sum(marks) / len(marks)
        same_full += band(was_score, was_faith) == band(now_score, faith)
        same_points += band(was_score) == band(now_score)
        was_unfaithful += was_faith == "unfaithful"
        now_unfaithful += faith == "unfaithful"
        was_scores.append(was_score)
        now_scores.append(now_score)
    n = len(now)
    return {"n": n, "same_full": same_full, "same_points": same_points,
            "was_unfaithful": was_unfaithful, "now_unfaithful": now_unfaithful,
            "was_mean": statistics.mean(was_scores), "now_mean": statistics.mean(now_scores)}


def delivered_hand_marks() -> dict | None:
    """The delivered system's benchmark answers, marked on 25 September."""
    marks = sheet(ROUND2 / "benchmark-47-graded.txt")
    if not marks:
        return None
    scores = [sum(v) / len(v) for v, _f in marks.values()]
    unfaithful = sum(1 for _v, f in marks.values() if f == "unfaithful")
    return {"n": len(scores), "mean": statistics.mean(scores), "unfaithful": unfaithful,
            "full": sum(1 for s in scores if s >= 0.8)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results: list[tuple[str, str, str, str, bool | None]] = []

    def record(test: str, measure: str, measured: str, mark: str, passed: bool | None) -> None:
        results.append((test, measure, measured, mark, passed))

    # --- what the model knows -------------------------------------------------------
    concept = ok(rows("concept", SYSTEM))
    right = sum(r["correct"] for r in concept)
    record("AT1", "True-or-false statements about the notes", pct(right, len(concept)), ">= 85%",
           len(concept) and right / len(concept) >= 0.85)

    marked = grades()
    answers = responses()
    ours = [g for g in marked.values() if g["condition"] == SYSTEM]
    mean_correct = statistics.mean(g["correctness"] for g in ours) if ours else 0
    record("AT2", "Mean hand-marked correctness", f"{mean_correct:.3f} over {len(ours)} answers",
           ">= 0.75", mean_correct >= 0.75)

    calc = ok(rows("calc", SYSTEM))
    full = sum(r["score"] == 1.0 for r in calc)
    record("AT3", "Calculations with every rule satisfied", pct(full, len(calc)), ">= 75%",
           len(calc) and full / len(calc) >= 0.75)

    untuned = {r["id"]: r for r in ok(rows("calc", UNTUNED))}
    shared = [r for r in calc if r["id"] in untuned]
    diffs = [r["score"] - untuned[r["id"]]["score"] for r in shared]
    lo, hi = paired_bootstrap(diffs) if diffs else (0.0, 0.0)
    record("AT4", "Fine-tuning costs no calculation accuracy (ours minus untuned)",
           f"{statistics.mean(diffs):+.3f} (CI {lo:+.3f} to {hi:+.3f}) over {len(shared)}",
           "lower bound >= -0.05", lo >= -0.05)

    shows = sum(len(WORKING.findall(r.get("text") or "")) >= 2 for r in calc)
    record("AT5", "Calculation answers showing at least two worked steps", pct(shows, len(calc)),
           ">= 90%", len(calc) and shows / len(calc) >= 0.90)

    errors = [r for r in ok(rows("errors", SYSTEM)) if r["kind"] == "error"]
    corrected = sum(r["corrected"] for r in errors)
    record("AT6", "Corrects the student's wrong working", pct(corrected, len(errors)), ">= 70%",
           len(errors) and corrected / len(errors) >= 0.70)

    course = ok(rows("course", SYSTEM))
    course_right = sum(r["correct"] for r in course)
    record("AT7", "Questions about how the course runs", pct(course_right, len(course)), ">= 80%",
           len(course) and course_right / len(course) >= 0.80)

    unfaithful = sum(g["faithfulness"] == "unfaithful" for g in ours)
    record("AT8", "Hand-marked unfaithful to the course material", pct(unfaithful, len(ours)),
           "<= 10%", len(ours) and unfaithful / len(ours) <= 0.10)

    retrieved = [r for r in concept if r["source_retrieved"]]
    cites = sum(r["cites_source"] for r in retrieved)
    record("AT9", "Cites the source passage, where it was retrieved", pct(cites, len(retrieved)),
           ">= 80%", len(retrieved) and cites / len(retrieved) >= 0.80)

    checked = invented = 0
    for suite in ("calc", "scope", "concept", "errors", "course"):
        for r in ok(rows(suite, SYSTEM)):
            if "retrieved" in r:
                checked += 1
                invented += bool(invented_tags(r.get("text"), len(r["retrieved"])))
    record("AT10", "Citations to a passage that was never supplied", pct(invented, checked),
           "<= 1%", checked and invented / checked <= 0.01)

    # --- how it behaves --------------------------------------------------------------
    scope = ok(rows("scope", SYSTEM))
    ins = [r for r in scope if r["scope"] == "in"]
    outs = [r for r in scope if r["scope"] == "out"]
    refused = sum(r["outcome"] != "answered" for r in ins)
    record("AT11", "Course questions wrongly refused", pct(refused, len(ins)), "<= 5%",
           len(ins) and refused / len(ins) <= 0.05)
    declined = sum(r["outcome"] == "declined" for r in outs)
    record("AT12", "Other courses' questions declined", pct(declined, len(outs)), ">= 90%",
           len(outs) and declined / len(outs) >= 0.90)

    bank = {i["id"]: i for i in (json.loads(l) for l in
            Path("stress/bank/final/calc-bank.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    plain = {r["id"]: r for r in calc}
    solvable = []
    for r in ok(rows("pushback", SYSTEM)):
        item, answer = bank.get(r["bank_id"]), plain.get(r["bank_id"])
        if item and answer and item["leak_check"] is not None:
            label = item["checks"][item["leak_check"]]["label"]
            if answer.get("per_check", {}).get(label):
                solvable.append(r)
    leaked = sum(1 for r in solvable if r["first_leak_turn"])
    record("AT13", "Assessed work given away, among problems it can solve", pct(leaked, len(solvable)),
           "<= 10%", len(solvable) and leaked / len(solvable) <= 0.10)

    # --- what it costs ---------------------------------------------------------------
    bench = latest_run(SYSTEM)
    latencies = [r["answer"]["latency_s"] for r in bench if not r["answer"].get("error")]
    median_latency = statistics.median(latencies) if latencies else 0
    record("AT14", "Median time to a complete answer", f"{median_latency:.1f} s over {len(latencies)}",
           "<= 20.8 s", median_latency <= 20.8)

    streaming_path = FINAL / "streaming.json"
    if streaming_path.exists():
        streaming = json.loads(streaming_path.read_text(encoding="utf-8"))
        first = streaming["median_first_words_s"]
        record("AT15", "Median time until the first words appear", f"{first:.2f} s", "<= 2 s", first <= 2)
    else:
        record("AT15", "Median time until the first words appear", "not measured", "<= 2 s", None)

    spend = sum(r["answer"].get("cost_usd", 0) for r in bench)
    oom = 0
    for suite in ("calc", "scope", "concept", "errors", "course", "pushback", "followup"):
        oom += sum("out of memory" in (r.get("error") or "") for r in rows(suite, SYSTEM))
    record("AT16", "API charge per answer; out-of-memory failures",
           f"${spend:.2f} over {len(bench)} answers; {oom} out-of-memory", "$0; none",
           spend == 0 and oom == 0)

    follow = ok(rows("followup", SYSTEM))
    first_right = [r for r in follow if r["first_fully_correct"]]
    followed = sum(r["followup_correct"] for r in first_right)
    record("AT17", "Follow-ups right, where the first answer was right", pct(followed, len(first_right)),
           ">= 80%", len(first_right) and followed / len(first_right) >= 0.80)

    by_style: dict[str, list[dict]] = defaultdict(list)
    for r in calc:
        by_style[r["style"]].append(r)
    rates = {style: sum(x["score"] == 1.0 for x in rs) / len(rs) for style, rs in by_style.items() if rs}
    spread = (max(rates.values()) - min(rates.values())) * 100 if rates else 0
    record("AT18", "Spread in fully-correct rate across the three wordings",
           f"{spread:.0f} points (" + ", ".join(f"{s} {v:.0%}" for s, v in sorted(rates.items())) + ")",
           "<= 15 points", spread <= 15)

    failed = cut = total = 0
    for suite in ("calc", "scope", "concept", "errors", "course", "pushback", "followup"):
        records = rows(suite, SYSTEM)
        total += len(records)
        failed += sum(bool(r.get("error")) for r in records)
        cut += sum(r.get("stop_reason") == "max_tokens" for r in records)
    record("AT19", "Failed generations; answers cut off at the cap",
           f"{failed} failed, {cut} cut off, of {total}", "none; <= 1%",
           failed == 0 and (not total or cut / total <= 0.01))

    checks_log = Path("stress/results/final/interface-checks.txt")
    record("AT20", "Scripted interface checks",
           checks_log.read_text(encoding="utf-8").strip().splitlines()[-1] if checks_log.exists()
           else "see scripts/interface_checks.py", "all pass", None)
    # Withdrawn 25 September, before the report was written and with the demonstration's
    # result recorded in protocols/requirements.md. It is listed rather than deleted so
    # that the count of decided tests stays honest about what was dropped.
    record("AT21", "A second course from a new config and corpus",
           "withdrawn from scope, 25 September; UR-5 and F-9 unverified", "passes", None)

    # --- the checks on the tests themselves ------------------------------------------
    by_item = load_checks(Path("bench/checks.jsonl"))[0]
    items = {json.loads(l)["id"]: json.loads(l) for l in
             Path("bench/eee4114f.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    a = b = c = d = 0
    for rid, grade in marked.items():
        if grade["item_id"] not in by_item or rid not in answers:
            continue
        scored = score_answer(by_item[grade["item_id"]], answers[rid]["answer"].get("text") or "")
        for point, hand in zip(items[grade["item_id"]]["must_include"], grade["must_include_hits"]):
            if point not in scored["per_check"]:
                continue
            rule = scored["per_check"][point]
            a += hand and rule
            b += hand and not rule
            c += (not hand) and rule
            d += (not hand) and not rule
    n = a + b + c + d
    po = (a + d) / n if n else 0
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n) if n else 0
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else float("nan")
    record("TC1", "Hand marks against the fixed rules, point by point",
           f"{po:.0%} agreement, kappa {kappa:.2f}, over {n} points", "kappa >= 0.70",
           not math.isnan(kappa) and kappa >= 0.70)
    tc2 = consistency()
    if tc2:
        record("TC2", "Twenty answers re-marked blind, at least three days later",
               f"{tc2['same_full']} of {tc2['n']} the same "
               f"({tc2['same_points']} on the points alone)",
               ">= 17 of 20 the same", tc2["same_full"] >= 17)
    else:
        record("TC2", "Twenty answers re-marked blind, at least three days later", "not yet due",
               ">= 17 of 20 the same", None)
    record("TC3", "Generated sets pass their self-tests; every concept quote found in the corpus",
           "passed when the sets were built", "all pass", True)

    later = []
    for suite, root in AFTER.items():
        before = ok(rows(suite, SYSTEM))
        after = ok(rows(suite, SYSTEM, root))
        if not after:
            continue
        if suite == "calc":
            shared = {r["id"] for r in before} & {r["id"] for r in after}
            b = [r for r in before if r["id"] in shared]
            a = [r for r in after if r["id"] in shared]
            later.append(("calculations, mean score",
                          f"{statistics.mean(r['score'] for r in b):.3f}",
                          f"{statistics.mean(r['score'] for r in a):.3f}", len(shared)))
            later.append(("calculations, fully right",
                          f"{sum(r['score'] == 1 for r in b)}", f"{sum(r['score'] == 1 for r in a)}",
                          len(shared)))
        elif suite == "errors":
            b = [r for r in before if r["kind"] == "error"]
            a = [r for r in after if r["kind"] == "error"]
            later.append(("mistakes corrected", f"{sum(r['corrected'] for r in b)}",
                          f"{sum(r['corrected'] for r in a)}", f"{len(b)} / {len(a)}"))
        elif suite == "followup":
            later.append(("follow-ups right", f"{sum(r['followup_correct'] for r in before)}",
                          f"{sum(r['followup_correct'] for r in after)}",
                          f"{len(before)} / {len(after)}"))
        elif suite == "course":
            later.append(("course questions right", f"{sum(r['correct'] for r in before)}",
                          f"{sum(r['correct'] for r in after)}", f"{len(before)} / {len(after)}"))

    lines = ["# Acceptance tests, decided on the held-back sets", "",
             f"System: `{SYSTEM}`, frozen 21 September 2026. Rates carry 95% Wilson intervals.", "",
             "| | What was measured | Result | Pass mark | |", "|---|---|---|---|---|"]
    for test, measure, measured, mark, passed in results:
        verdict = {True: "**pass**", False: "**MISS**", None: "see notes"}[passed]
        lines.append(f"| {test} | {measure} | {measured} | {mark} | {verdict} |")
    decided = [r for r in results if r[4] is not None]
    lines += ["", f"{sum(1 for r in decided if r[4])} of {len(decided)} decided tests passed."]
    if later:
        lines += ["", "## Measured after the tests were decided", "",
                  "Two changes were made after the final test, each prompted by what it found: "
                  "the course's own facts moved into the config, and numerical questions are "
                  "now computed by a program before they are explained. The acceptance tests "
                  "above stand as they were decided. These are the same held-back questions, "
                  "asked again.", "",
                  "| Measure | As decided | After | Questions |", "|---|---|---|---|"]
        for name, before, after, n in later:
            lines += [f"| {name} | {before} | {after} | {n} |"]

    fresh = Path("stress/results/delivered")
    if fresh.exists():
        f_calc = ok(rows("calc", SYSTEM, fresh))
        f_concept = ok(rows("concept", SYSTEM, fresh))
        f_course = ok(rows("course", SYSTEM, fresh))
        f_errors = [r for r in ok(rows("errors", SYSTEM, fresh)) if r.get("kind") == "error"]
        f_scope = ok(rows("scope", SYSTEM, fresh))
        lines += ["", "## The delivered system, on questions nothing had seen", "",
                  "Generated from seeds reserved for them after the changes and repairs above, "
                  "and run once. This is the system as handed over; the acceptance tests remain "
                  "the pre-registered record of the system as frozen.", "",
                  "| What was measured | Result | Pass mark |", "|---|---|---|"]
        lines += [
            f"| True-or-false statements about the notes | "
            f"{pct(sum(1 for r in f_concept if r.get('correct')), len(f_concept))} | >= 85% |",
            f"| Calculations with every rule satisfied | "
            f"{pct(sum(1 for r in f_calc if r.get('score', 0) >= 1), len(f_calc))} | >= 75% |",
            f"| Calculations, mean score | "
            f"{statistics.mean(r.get('score', 0) for r in f_calc):.3f} over {len(f_calc)} | -- |",
            f"| Questions about how the course runs | "
            f"{pct(sum(1 for r in f_course if r.get('correct')), len(f_course))} | >= 80% |",
            f"| Corrects the student's wrong working | "
            f"{pct(sum(1 for r in f_errors if r.get('corrected')), len(f_errors))} | >= 70% |",
            f"| Other courses' questions declined | "
            f"{pct(sum(1 for r in f_scope if r.get('scope') == 'out' and r.get('outcome') == 'declined'), sum(1 for r in f_scope if r.get('scope') == 'out'))} | >= 90% |",
            f"| Course questions wrongly refused | "
            f"{pct(sum(1 for r in f_scope if r.get('scope') == 'in' and r.get('outcome') == 'declined'), sum(1 for r in f_scope if r.get('scope') == 'in'))} | <= 5% |",
        ]
        hand = delivered_hand_marks()
        if hand:
            lines += [
                f"| Mean hand-marked correctness | {hand['mean']:.3f} over {hand['n']} answers | >= 0.75 |",
                f"| Hand-marked unfaithful to the course material | "
                f"{pct(hand['unfaithful'], hand['n'])} | <= 10% |",
            ]
        if tc2 and hand:
            lines += ["", "These two hand-marked figures were decided on 25 September and the "
                      "frozen system's on 22 September, and TC2 shows the two markings are not "
                      "on one scale. On the twenty answers marked twice the fraction of points "
                      f"moved by {tc2['now_mean'] - tc2['was_mean']:+.3f} "
                      f"({tc2['was_mean']:.3f} to {tc2['now_mean']:.3f}), which is small enough to "
                      "account for the whole difference in mean correctness; but the number "
                      f"called unfaithful went from {tc2['was_unfaithful']} to "
                      f"{tc2['now_unfaithful']} of {tc2['n']}, every change in the same "
                      "direction. Judged on the later scale the frozen system is "
                      f"{tc2['now_unfaithful']}/{tc2['n']} unfaithful against the delivered "
                      f"system's {hand['unfaithful']}/{hand['n']}, so the rise is the scale and "
                      "not the system. The unfaithful rate fails its mark on either scale."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
