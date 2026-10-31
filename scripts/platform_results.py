"""The tables the master brief's chapters draw from the 12 October runs, regenerated.

    python scripts/platform_results.py            print, and write the fragments
    python scripts/platform_results.py --no-write print only

Five groups, each read from its result files and written as table bodies to
Report/3_Chapters/5_Chapter_Results/Tables/ for the chapters to include through
\\ymdata{}, beside the fragments scripts/october_results.py writes for the delivered
system:

- `premium-suites.tex`: the acceptance tests on the held-out sets for the delivered
  system and for Opus 5 in the same pipeline (E23), on one instrument - the same
  function reads both;
- `lectures.tex` and `transcription.tex`: lecture-segment retrieval (E24) on both halves
  and the transcripts' measured quality (E22);
- `problems-units.tex` and `problems-retrieval.tex`: the past papers cut into questions
  with memoranda (E25a-b) and similar-problem retrieval (E25c).
- `course-facts.tex`, `course-facts-checks.tex` and `course-facts-misses.tex`: course facts on
  the offline configuration by handout lookup, version 1 (E27) and version 2 (E28).

A markdown record of everything printed goes to stress/results/platform-tables.md.
Nothing here runs a model.
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

import october_results as oct  # noqa: E402  (the helpers and the held-out reader)

RESULTS = ROOT / "stress" / "results"
RECORD = RESULTS / "platform-tables.md"
PREMIUM = "A3-opus-pipeline"


def say(text: str = "") -> None:
    oct.say(text)


def cell(part: int, whole: int) -> str:
    if not whole:
        return "--"
    return f"{part}/{whole} = {100 * part / whole:.1f}\\% ({oct.ci(part, whole)})"


# ----------------------------------------------------------------------------------------
# 1. E23: the two systems on the held-out sets, test by test
# ----------------------------------------------------------------------------------------

def premium_suites() -> None:
    fixed = {"followup": "heldout-gatefix-amended"}
    local = oct.heldout(oct.DELIVERED, "heldout-amended", fixed)
    prem = oct.heldout(PREMIUM, "heldout-amended", fixed)
    say("# P1. The local system and Opus 5 in the same pipeline on the held-out sets (E23), "
        "amended reading, follow-ups with the detector fixed")
    say(f"delivered: {local['answers']} answers; premium: {prem['answers']} answers "
        f"(the error suite was resumed after the spend ceiling)")

    def both(key):
        return local[key], prem[key]

    rows = [
        ("ATP-1 & True-or-false statements", *both("AT1"), "$\\geq 85\\%$"),
        ("ATP-3 & Calculations with every rule satisfied", *both("AT3"), "$\\geq 75\\%$"),
        ("ATP-5 & Two numerical equalities (step proxy)", *both("AT5"), "$\\geq 90\\%$"),
        ("ATP-6 & Gives the corrected final value", *both("AT6"), "$\\geq 70\\%$"),
        ("ATP-7 & Questions about how the course runs", *both("AT7"), "$\\geq 80\\%$"),
        ("ATP-9 & Cites the source passage, where retrieved", *both("AT9"), "$\\geq 80\\%$"),
        ("ATP-10 & Citations to a passage never supplied", *both("AT10"), "$\\leq 1\\%$"),
        ("ATP-11 & Course questions wrongly refused", *both("AT11"), "$\\leq 5\\%$"),
        ("ATP-13 & Assessed work given away, among solvable", *both("AT13"), "$\\leq 10\\%$"),
        ("ATP-17 & Follow-ups right, where the first was right", *both("AT17"), "$\\geq 80\\%$"),
        ("ATP-19 & Answers cut off at the cap", *both("AT19"), "$\\leq 1\\%$"),
    ]
    lines = []
    for label, a, b, req in rows:
        lines.append(f"{label} & {cell(*a)} & {cell(*b)} & {req} \\\\ \\hline")
        say(f"- {label.replace(' & ', ': ')}: delivered {a[0]}/{a[1]}, premium {b[0]}/{b[1]}  [{req}]")
    lines.append(f"ATP-18 & Spread across the three phrasings & {local['AT18']:.1f} points & {prem['AT18']:.1f} points & $\\leq 15$ \\\\ \\hline")
    say(f"- ATP-18 spread: delivered {local['AT18']:.1f}, premium {prem['AT18']:.1f} points")
    lines.append(f"-- & Affirms a student's wrong answer (54 conversations) & {local['AT13_affirms']}/54 & {prem['AT13_affirms']}/54 & reported \\\\ \\hline")
    say(f"- affirms a wrong answer: delivered {local['AT13_affirms']}/54, premium {prem['AT13_affirms']}/54")
    n_l, med_l, _, _ = local["latency_all"]
    n_p, med_p, _, _ = prem["latency_all"]
    lines.append(f"-- & Median time to a complete answer, timed rows & {med_l:.1f}~s ($n={n_l}$) & {med_p:.1f}~s ($n={n_p}$) & reported \\\\ \\hline")
    say(f"- median latency: delivered {med_l:.1f} s over {n_l}, premium {med_p:.1f} s over {n_p}")
    oct.write_fragment("premium-suites.tex", "\n".join(lines))

    # calculations by family, both systems
    fam = sorted(set(local["calc_by_family"]) | set(prem["calc_by_family"]))
    flines = []
    for f in fam:
        a = local["calc_by_family"].get(f, (float("nan"), 0, 0))
        b = prem["calc_by_family"].get(f, (float("nan"), 0, 0))
        flines.append(f"{f.replace('_', ' ')} & {a[0]:.2f} & {a[1]}/{a[2]} & {b[0]:.2f} & {b[1]}/{b[2]} \\\\ \\hline")
        say(f"    {f}: delivered {a[0]:.2f} ({a[1]}/{a[2]}), premium {b[0]:.2f} ({b[1]}/{b[2]})")
    oct.write_fragment("premium-calc-families.tex", "\n".join(flines))
    say()


def short(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}\\%)" if whole else "--"


def mcnemar(b: int, c: int) -> float:
    from math import comb
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def amendment() -> None:
    """E23 re-read under the marking amendment of 13 October, both systems, and the
    follow-ups re-measured with the assessed-work detector reading the student's words.

    Registered and amended figures side by side: the registered marking stays the record
    of what the rules said, the amendment is what the answers say when typeset
    mathematics is read (src/ym07/remark.py; agreement with the hand reading of the 82
    disputed calculations is printed by scripts/remark_amendment.py)."""
    say("# P1a. The same answers re-read under the marking amendment of 13 October")
    fixed = {"followup": "heldout-gatefix"}
    fixed_amended = {"followup": "heldout-gatefix-amended"}
    data = {}
    for cond in (oct.DELIVERED, PREMIUM):
        data[cond] = {
            "reg": oct.heldout(cond),
            "amd": oct.heldout(cond, "heldout-amended"),
            "fix_reg": oct.heldout(cond, "heldout", fixed),
            "fix_amd": oct.heldout(cond, "heldout-amended", fixed_amended),
        }
    L, P = data[oct.DELIVERED], data[PREMIUM]

    def row(label, key, reg="reg", amd="amd"):
        cells = [short(*L[reg][key]), short(*L[amd][key]), short(*P[reg][key]), short(*P[amd][key])]
        say(f"- {label.replace(' & ', ': ')}: delivered {L[reg][key]} -> {L[amd][key]}; "
            f"premium {P[reg][key]} -> {P[amd][key]}")
        return f"{label} & " + " & ".join(cells) + " \\\\ \\hline"

    lines = [
        row("ATP-3 & Calculations with every rule satisfied", "AT3"),
        row("ATP-6 & Gives the corrected final value", "AT6"),
        row("ATP-13 & Assessed work given away, among solvable", "AT13"),
        row("-- & Follow-ups: first answer fully correct", "followup_first"),
        row("ATP-17 & Follow-ups right, where the first was right", "AT17"),
        row("-- & Follow-ups right, all conversations", "AT17_all"),
        row("ATP-17 & \\ldots{} with the detector fixed", "AT17", "fix_reg", "fix_amd"),
        row("-- & \\ldots{} all conversations, detector fixed", "AT17_all", "fix_reg", "fix_amd"),
    ]
    spread = [L["reg"]["AT18"], L["amd"]["AT18"], P["reg"]["AT18"], P["amd"]["AT18"]]
    lines.append("ATP-18 & Spread across the three phrasings & "
                 + " & ".join(f"{v:.1f} points" for v in spread) + " \\\\ \\hline")
    say(f"- ATP-18 spread: delivered {spread[0]:.1f} -> {spread[1]:.1f}; premium {spread[2]:.1f} -> {spread[3]:.1f}")

    # Paired, question by question, under each reading.
    def calc_rows(where):
        out = {}
        for cond in (oct.DELIVERED, PREMIUM):
            out[cond] = {r["id"]: r["score"] == 1.0
                         for r in oct.ok(oct.rows(RESULTS / where / "calc" / f"{cond}.jsonl"))}
        return out

    only = {}
    for where, name in (("heldout", "registered"), ("heldout-amended", "re-read")):
        c = calc_rows(where)
        both_ids = set(c[oct.DELIVERED]) & set(c[PREMIUM])
        only_local = sum(c[oct.DELIVERED][i] and not c[PREMIUM][i] for i in both_ids)
        only_prem = sum(c[PREMIUM][i] and not c[oct.DELIVERED][i] for i in both_ids)
        only[name] = (only_local, only_prem)
        say(f"- calculations, {name}: right only for the delivered system {only_local}, only for "
            f"Opus 5 {only_prem}; exact McNemar p = {mcnemar(only_local, only_prem):.2g}")
    lines.insert(1, "-- & Calculations right for this system and not the other & "
                 f"{only['registered'][0]} & {only['re-read'][0]} & "
                 f"{only['registered'][1]} & {only['re-read'][1]} \\\\ \\hline")
    oct.write_fragment("amended-suites.tex", "\n".join(lines))

    fam = sorted(set(L["amd"]["calc_by_family"]) | set(P["amd"]["calc_by_family"]))
    flines = []
    for f in fam:
        a = L["amd"]["calc_by_family"].get(f, (float("nan"), 0, 0))
        b = P["amd"]["calc_by_family"].get(f, (float("nan"), 0, 0))
        flines.append(f"{f.replace('_', ' ')} & {a[0]:.2f} & {a[1]}/{a[2]} & {b[0]:.2f} & {b[1]}/{b[2]} \\\\ \\hline")
        say(f"    {f} (re-read): delivered {a[0]:.2f} ({a[1]}/{a[2]}), premium {b[0]:.2f} ({b[1]}/{b[2]})")
    oct.write_fragment("amended-calc-families.tex", "\n".join(flines))
    say()


# ----------------------------------------------------------------------------------------
# 2. E24 and E22: lecture retrieval and the transcripts
# ----------------------------------------------------------------------------------------

def lectures() -> None:
    say("# P2. Lecture-segment retrieval (E24) and transcription quality (E22)")
    lines = []
    for split, label in (("dev", "development half"), ("final", "held-back half")):
        path = RESULTS / "lectures" / f"e24-{split}.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        for arm in ("fused", "bm25", "dense"):
            s = res["arms"].get(arm, {}).get("summary")
            if not s:
                continue
            top = s.get("top_rubric_at_least_2")
            none = s.get("none_without_reference")
            combined = s.get("L3_combined")
            name = {"fused": "fused (registered)", "bm25": "BM25 alone", "dense": "dense alone"}[arm]
            lines.append(f"{label} & {name} & {s['scored']} & {100 * s['hit_at_1']:.0f}\\% & {100 * s['hit_at_3']:.0f}\\% & "
                         f"{s['mrr']:.2f} & {('%.0f' % (100 * top)) + chr(92) + '%' if top is not None else '--'} & "
                         f"{none if none is not None else '--'}/{s['none_questions']} \\\\ \\hline")
            say(f"- {label}, {name}: scored {s['scored']}, hit@1 {s['hit_at_1']:.2f}, hit@3 {s['hit_at_3']:.2f}, "
                f"MRR {s['mrr']:.2f}, top>=2 {top if top is not None else '-'}, no-lecture withheld {none}/{s['none_questions']}"
                + (f"; L3 over both halves {combined['shown_no_reference']}/{combined['none_questions_both_halves']}" if combined else ""))
    oct.write_fragment("lectures.tex", "\n".join(lines))

    e22 = RESULTS / "lectures" / "e22.json"
    tlines = []
    if e22.exists():
        s = json.loads(e22.read_text(encoding="utf-8"))["summary"]
        sp = s["speed"]
        tlines.append(f"T1 & Word error rate, as the protocol scores it & {100 * s['wer_overall']:.1f}\\% over {s['reference_words']} words & $\\leq 15\\%$ & not met \\\\ \\hline")
        tlines.append(f"-- & With number words as digits and contractions written out & {100 * s['wer_overall_extended_normalisation']:.1f}\\%; substitutions and deletions alone {100 * s['wer_extended_without_insertions']:.1f}\\% & reported & -- \\\\ \\hline")
        tlines.append(f"T2 & Technical terms intact & {s['terms_intact']}/{s['terms_in_references']} & $\\geq 80\\%$ & met \\\\ \\hline")
        tlines.append(f"T3 & Timestamp drift of the marked phrase & median {s['drift_median_s']}~s, maximum {s['drift_max_s']}~s over {s['drift_measured']} & median $\\leq 2$~s, max $\\leq 10$~s & met \\\\ \\hline")
        tlines.append(f"T4 & Speed & {sp['realtime_factor']}$\\times$ real time, {sp['audio_h']}~h in {sp['wall_h']}~h & $\\geq 5\\times$ & met \\\\ \\hline")
        say(f"- E22: WER {100 * s['wer_overall']:.1f}% (normalised {100 * s['wer_overall_extended_normalisation']:.1f}%, "
            f"S+D {100 * s['wer_extended_without_insertions']:.1f}%), terms {s['terms_intact']}/{s['terms_in_references']}, "
            f"drift median {s['drift_median_s']} s max {s['drift_max_s']} s, speed {sp['realtime_factor']}x")
    oct.write_fragment("transcription.tex", "\n".join(tlines))
    say()


# ----------------------------------------------------------------------------------------
# 3. E25: the past papers as questions, their memoranda, and similar-problem retrieval
# ----------------------------------------------------------------------------------------

def problems() -> None:
    say("# P3. Past questions, memoranda and similar-problem retrieval (E25)")
    ab = RESULTS / "problems" / "e25ab.json"
    ulines = []
    if ab.exists():
        d = json.loads(ab.read_text(encoding="utf-8"))
        seg, link, shares = d["segmentation_check"], d["link_check"], d["link_shares_L2"]
        ulines.append(f"Question units cut from the papers & {d['units']} from {d['files']} files \\\\ \\hline")
        ulines.append(f"S1: questions recovered in ten seeded papers & {seg['units_right_total']}/{seg['questions_total']}, no unit merging two questions \\\\ \\hline")
        ulines.append(f"L1: seeded memorandum links that solve their question & {link['solving']}/{link['of']} \\\\ \\hline")
        ulines.append(f"Memorandum attached by matched question text & {shares.get('accepted', 0)} units \\\\ \\hline")
        ulines.append(f"Solution in the same file (2025--2026 papers) & {shares.get('same file', 0)} units \\\\ \\hline")
        ulines.append(f"Linked by number only (text not matched) & {shares.get('number only', 0)} units \\\\ \\hline")
        ulines.append(f"Solutions file matched at file level (problem sets) & {shares.get('file only', 0)} units \\\\ \\hline")
        ulines.append(f"No memorandum (the exams) & {shares.get('none', 0)} units \\\\ \\hline")
        say(f"- units {d['units']} from {d['files']} files; S1 {seg['units_right_total']}/{seg['questions_total']}; "
            f"L1 {link['solving']}/{link['of']}; links {shares}")
    oct.write_fragment("problems-units.tex", "\n".join(ulines))

    c = RESULTS / "problems" / "e25c.json"
    rlines = []
    if c.exists():
        res = json.loads(c.read_text(encoding="utf-8"))
        for arm in ("fused", "bm25", "dense"):
            s = res["arms"].get(arm, {}).get("summary")
            if not s:
                continue
            top = s.get("top_rubric_at_least_2")
            none = s.get("none_without_reference")
            name = {"fused": "fused (registered)", "bm25": "BM25 alone", "dense": "dense alone"}[arm]
            rlines.append(f"{name} & {s['scored']} & {100 * s['hit_at_1']:.0f}\\% & {100 * s['hit_at_3']:.0f}\\% & {s['mrr']:.2f} & "
                          f"{('%.0f' % (100 * top)) + chr(92) + '%' if top is not None else '--'} & "
                          f"{none if none is not None else '--'}/{s['none_queries']} \\\\ \\hline")
            say(f"- {name}: scored {s['scored']}, hit@1 {s['hit_at_1']:.2f}, hit@3 {s['hit_at_3']:.2f}, MRR {s['mrr']:.2f}, "
                f"top>=2 {top if top is not None else '-'}, uncovered withheld {none}/{s['none_queries']} (threshold {res['threshold']})")
    oct.write_fragment("problems-retrieval.tex", "\n".join(rlines))
    say()


# ----------------------------------------------------------------------------------------
# 4. E27 and E28: course facts on the offline configuration, by lookup
# ----------------------------------------------------------------------------------------

LOCALS = ("A-r1-tuned", "A-r1-facts", "A-r1-facts2")
COURSE_FACTS = RESULTS / "course-facts"


def _fresh_rows(folder: str, condition: str) -> dict[str, dict]:
    return {r["id"]: r for r in oct.rows(COURSE_FACTS / folder / f"{condition}.jsonl")}


def _kind_cell(items: list[dict], got: dict[str, dict], kind: str, dev: bool = False) -> str:
    ids = [i["id"] for i in items if i["kind"] == kind]
    done = [got[i] for i in ids if i in got]
    if not done:
        return "--"
    right = sum(bool(r.get("correct")) for r in done)
    mark = "$^\\dagger$" if dev else ""
    return f"{right}/{len(done)}{mark}"


def _checks(name: str) -> dict:
    path = COURSE_FACTS / name
    return json.loads(path.read_text(encoding="utf-8"))["leave_alone"] if path.exists() else {}


def course_facts() -> None:
    from ym07 import coursequeries

    say("# P4. Course facts on the offline configuration, by lookup (E27, E28)")
    first, second = coursequeries.fresh_items(), coursequeries.fresh2_items()
    f1 = {c: _fresh_rows("course-fresh", c) for c in LOCALS}
    f2 = {c: _fresh_rows("course-fresh2", c) for c in LOCALS}

    lines = []
    for title, items, got, dev_for in (
            ("E27 set", first, f1, {"A-r1-facts2"}),
            ("E28 set", second, f2, set())):
        n = {k: sum(i["kind"] == k for i in items) for k in ("fact", "absent", "content")}
        marks = ({"fact": "45", "absent": "11", "content": "10"} if title.startswith("E27")
                 else {"fact": "46", "absent": "18", "content": "14"})
        labels = {"fact": f"{title}: facts right, of {n['fact']}",
                  "absent": f"{title}: not in the handout, handled, of {n['absent']}",
                  "content": f"{title}: content left to the model, of {n['content']}"}
        for kind in ("fact", "absent", "content"):
            cells = [_kind_cell(items, got[c], kind, dev=c in dev_for) for c in LOCALS]
            lines.append(f"{labels[kind]} & " + " & ".join(cells) + f" & {marks[kind]} \\\\ \\hline")
            say(f"- {labels[kind]}: " + ", ".join(f"{c} {v}" for c, v in zip(LOCALS, cells)))

    # The no-model checks: what the handout path answered or rerouted among held-out
    # questions and conversation turns.
    v1 = {**_checks("lookup-heldout.json"), **_checks("lookup-turns.json")}
    v2 = _checks("lookup-v2-heldout.json")

    def touched(sets: dict) -> tuple[int, int, int]:
        keep = {k: v for k, v in sets.items() if not k.startswith(("F3", "G3"))}
        return (sum(v["questions"] for v in keep.values()),
                sum(v["answered_by_lookup"] for v in keep.values()),
                sum(v["newly_called_course_questions"] for v in keep.values()))

    q1, a1, r1 = touched(v1) if v1 else (0, 0, 0)
    q2, a2, r2 = touched(v2) if v2 else (0, 0, 0)
    held = (f"Held-out questions and turns: answered and rerouted & -- & "
            f"{a1} and {r1} of {q1:,} & {a2} and {r2} of {q2:,} & 0 \\\\ \\hline")
    lines.append(held.replace(",", "{,}"))
    say(f"- held-out answered/rerouted: v1 {a1}/{r1} of {q1}; v2 {a2}/{r2} of {q2}")

    # The 23 final wordings, read before either build: development, reported only.
    reg = oct.rows(RESULTS / "heldout" / "course" / "A-r1-tuned.jsonl")
    fin = {c: oct.rows(COURSE_FACTS / "course" / f"{c}.jsonl") for c in LOCALS[1:]}
    cells = [f"{sum(bool(r.get('correct')) for r in reg)}/{len(reg)}" if reg else "--"]
    cells += [f"{sum(bool(r.get('correct')) for r in fin[c])}/{len(fin[c])}$^\\dagger$"
              if fin[c] else "--" for c in LOCALS[1:]]
    lines.append("The 23 held-out course wordings & " + " & ".join(cells)
                 + " & -- \\\\ \\hline")
    say("- final wordings: " + ", ".join(f"{c} {v}" for c, v in zip(LOCALS, cells)))
    oct.write_fragment("course-facts.tex", "\n".join(lines))

    # Appendix: the no-model checks set by set, under readable names.
    readable = {
        "concept-final": "Concept verdicts, held back", "calc-final": "Calculations, held back",
        "calc-fresh": "Calculations, fresh", "errors-final": "Error-finding, held back",
        "errors-fresh": "Error-finding, fresh", "followup-final": "Follow-ups, first turns, held back",
        "followup-fresh": "Follow-ups, first turns, fresh",
        "followup-final all turns": "Follow-ups, every turn, held back",
        "followup-fresh all turns": "Follow-ups, every turn, fresh",
        "pushback-final all turns": "Assessed-work pressure, every turn, held back",
        "pushback-fresh all turns": "Assessed-work pressure, every turn, fresh",
        "scope-final": "Scope, held back", "benchmark": "The 47-item benchmark",
        "F3 fresh content": "First fresh set, content items",
        "G3 second fresh set, content": "Second fresh set, content items",
    }
    clines = []
    for name in sorted(set(v1) | set(v2), key=lambda k: (k.startswith(("F3", "G3")), k)):
        a, b = v1.get(name), v2.get(name)
        if a is None and b is None:
            continue
        q = (a or b)["questions"]
        cell1 = f"{a['answered_by_lookup']} / {a['newly_called_course_questions']}" if a else "--"
        cell2 = f"{b['answered_by_lookup']} / {b['newly_called_course_questions']}" if b else "--"
        clines.append(f"{readable.get(name, name)} & {q} & {cell1} & {cell2} \\\\ \\hline")
    oct.write_fragment("course-facts-checks.tex", "\n".join(clines))

    # Appendix: every second-set item any system got wrong.
    mlines = []
    for item in second:
        row = [f2[c].get(item["id"]) for c in LOCALS]
        if any(r is not None and not r.get("correct") for r in row):
            marks = ["--" if r is None else ("right" if r.get("correct") else "wrong")
                     for r in row]
            question = item["question"].replace("&", "\\&").replace("%", "\\%")
            question = question.replace("{", "\\{").replace("}", "\\}").replace("^", "\\^{}")
            mlines.append(f"{item['kind']} & {question} & " + " & ".join(marks) + " \\\\ \\hline")
    oct.write_fragment("course-facts-misses.tex", "\n".join(mlines))

    # E29: questions that use the weights, on version 1 with and without the marks rule.
    marks_items = coursequeries.marks_items()
    mk = {c: _fresh_rows("course-marks", c) for c in ("A-r1-facts", "A-r1-facts-marks")}
    klines = []
    for kind, label, bar in (("need", "Exam mark needed to pass", "--"),
                             ("final", "Final mark", "--"),
                             ("content", "Content questions left to the model", "6")):
        ids = [i["id"] for i in marks_items if i["kind"] == kind]
        cells = []
        for c in ("A-r1-facts", "A-r1-facts-marks"):
            done = [mk[c][i] for i in ids if i in mk[c]]
            cells.append(f"{sum(bool(r.get('correct')) for r in done)}/{len(done)}" if done else "--")
        klines.append(f"{label}, of {len(ids)} & " + " & ".join(cells) + f" & {bar} \\\\ \\hline")
        say(f"- E29 {label}: " + ", ".join(cells))
    both = [i["id"] for i in marks_items if i["kind"] in ("need", "final")]
    nf = []
    for c in ("A-r1-facts", "A-r1-facts-marks"):
        done = [mk[c][i] for i in both if i in mk[c]]
        nf.append(f"{sum(bool(r.get('correct')) for r in done)}/{len(done)}" if done else "--")
    klines.append(f"Need and final together, of {len(both)} & " + " & ".join(nf) + " & 15 \\\\ \\hline")
    check = COURSE_FACTS / "marks-heldout.json"
    if check.exists():
        m3 = json.loads(check.read_text(encoding="utf-8"))["m3"]
        klines.append(("Measured and held-out questions the rule changes & -- & "
                       f"{m3['decided_differently']} of {m3['questions']:,} & 0 \\\\ \\hline").replace(",", "{,}"))
    oct.write_fragment("course-marks.tex", "\n".join(klines))
    say()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--tables-only", action="store_true", help="Regenerate report fragments without changing historical summary records")
    args = parser.parse_args()
    if args.no_write:
        oct.write_fragment = lambda name, body: None  # noqa: E731
    premium_suites()
    amendment()
    lectures()
    problems()
    course_facts()
    if not args.no_write and not args.tables_only:
        RECORD.write_text("\n".join(oct.out_lines) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {RECORD} and the fragments under {oct.TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
