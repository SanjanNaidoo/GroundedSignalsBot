"""E24 - does the right lecture and the right minute come back for a question?

    python scripts/lecture_eval.py --split dev
    python scripts/lecture_eval.py --split final --threshold 0.62

Builds the lecture index of ym07.lectures over every complete transcript, asks it the
registered questions of one split, and scores the answers against the relevant windows
labelled in the split's jsonl (written by scripts/lecture_label.py apply), under the
pass marks fixed in protocols/lectures.md before any index existed:

- a returned window is a hit when it is in the same recording as a relevant window and
  either starts within 30 s of it or overlaps it;
- L1: a hit in the top three for at least 70% of the questions that have a relevant window;
- L2: the top result scores at least 2 on the labelling rubric for at least 60% of those
  questions; a top result nobody has read is reported as unlabelled, not guessed;
- L3: for the questions the lectures do not cover, no reference is shown in at least 3
  of 4. The platform shows no reference when the dense similarity of the best window is
  below a threshold. The threshold is chosen on the development half from the numbers
  this script prints, written into the protocol, and then passed here unchanged for the
  held-back half.

The fused retriever is the registered arm. BM25 and the dense model alone are run beside
it from the same index objects, as diagnostics, because knowing which half of the fusion
carries a lecture question costs nothing once the index exists.

Everything is written to stress/results/lectures/e24-<split>.json: the settings, every
question's top ten windows with scores, and the summary, so the table in the report is
regenerated from the file and not retyped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.lectures import MIN_WORDS, STEP_S, WINDOW_S, LectureIndex, Window  # noqa: E402

BANK = ROOT / "stress" / "bank" / "lectures"
TRANSCRIPTS = ROOT / "lectures" / "transcripts"
RESULTS = ROOT / "stress" / "results" / "lectures"
TOLERANCE_S = 30.0
DEPTH = 10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def load_questions(split: str) -> list[dict]:
    path = BANK / f"lectures-{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def registered_digest(questions: list[dict]) -> str:
    """The digest of the questions as registered, before any window was labelled.

    scripts/lecture_sets.py registered each split with `relevant_windows` empty and no
    `labelled_candidates`; `lecture_label.py apply` fills those two fields in. Stripping
    them and serialising the rows the same way recovers the registered bytes, so the
    manifest's digest can be checked after labelling: the questions themselves must not
    have moved.
    """
    rows = []
    for q in questions:
        row = {k: v for k, v in q.items() if k != "labelled_candidates"}
        row["relevant_windows"] = []
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()[:16]


def is_hit(window: Window, relevant: Window) -> bool:
    if window.recording != relevant.recording:
        return False
    if abs(window.start - relevant.start) <= TOLERANCE_S:
        return True
    return window.start < relevant.end and relevant.start < window.end


def score_method(name: str, retriever, index: LectureIndex, questions: list[dict],
                 labels: dict, threshold: float | None, dense) -> dict:
    rows = []
    for q in questions:
        relevant = [index.windows[w] for w in q.get("relevant_windows", []) if w in index.windows]
        hits = retriever.search(q["question"], DEPTH)
        ranked = [(index.windows[h.chunk.chunk_id], h.score) for h in hits]
        first_hit = next((rank for rank, (w, _) in enumerate(ranked, start=1)
                          if any(is_hit(w, r) for r in relevant)), None)
        top = ranked[0][0] if ranked else None
        top_label = labels.get(q["id"], {}).get(top.window_id) if top else None
        dense_top = dense.search(q["question"], 1)[0].score if dense is not None else None
        shown = dense_top is None or threshold is None or dense_top >= threshold
        rows.append({
            "id": q["id"], "series": q["series"], "kind": q["kind"], "expect": q["expect"],
            "question": q["question"],
            "relevant_windows": len(relevant),
            "first_hit_rank": first_hit,
            "hit_at_1": first_hit is not None and first_hit <= 1,
            "hit_at_3": first_hit is not None and first_hit <= 3,
            "hit_at_5": first_hit is not None and first_hit <= 5,
            "reciprocal_rank": (1.0 / first_hit) if first_hit else 0.0,
            "top_window": top.window_id if top else None,
            "top_reference": top.reference() if top else None,
            "top_rubric": top_label,
            "dense_top_similarity": dense_top,
            "reference_shown": shown,
            "top10": [{"window": w.window_id, "reference": w.reference(), "start": round(w.start, 1),
                       "end": round(w.end, 1), "score": round(s, 5),
                       "hit": any(is_hit(w, r) for r in relevant)} for w, s in ranked],
        })

    scored = [r for r in rows if r["expect"] == "lecture" and r["relevant_windows"]]
    unlabelled = [r["id"] for r in rows if r["expect"] == "lecture" and not r["relevant_windows"]]
    none_rows = [r for r in rows if r["expect"] == "none"]
    top_read = [r for r in scored if r["top_rubric"] is not None]
    summary = {
        "method": name,
        "questions": len(rows),
        "scored": len(scored),
        "unlabelled_questions": unlabelled,
        "hit_at_1": sum(r["hit_at_1"] for r in scored) / len(scored) if scored else None,
        "hit_at_3": sum(r["hit_at_3"] for r in scored) / len(scored) if scored else None,
        "hit_at_5": sum(r["hit_at_5"] for r in scored) / len(scored) if scored else None,
        "mrr": sum(r["reciprocal_rank"] for r in scored) / len(scored) if scored else None,
        "top_rubric_read": len(top_read),
        "top_rubric_unread": [r["id"] for r in scored if r["top_rubric"] is None],
        "top_rubric_at_least_2": (sum(r["top_rubric"] >= 2 for r in top_read) / len(scored)
                                  if scored and len(top_read) == len(scored) else None),
        "none_questions": len(none_rows),
        "none_without_reference": (sum(not r["reference_shown"] for r in none_rows)
                                   if threshold is not None else None),
        "lecture_questions_losing_reference": (
            [r["id"] for r in scored if not r["reference_shown"]] if threshold is not None else None),
        "L1_pass": None, "L2_pass": None, "L3_pass": None,
    }
    if summary["hit_at_3"] is not None:
        summary["L1_pass"] = summary["hit_at_3"] >= 0.70
    if summary["top_rubric_at_least_2"] is not None:
        summary["L2_pass"] = summary["top_rubric_at_least_2"] >= 0.60
    # L3 is judged over the four no-lecture questions of both halves together; see main().
    return {"summary": summary, "rows": rows}


def combined_l3(results: dict, out_dir: Path) -> None:
    """L3 over both halves: the protocol's 'at least 3 of 4' spans the two splits.

    Reads the other split's results file if it exists and was run with the same
    threshold, adds the no-lecture counts, and records the verdict in both summaries.
    """
    other_name = "final" if results["split"] == "dev" else "dev"
    other_path = out_dir / f"e24-{other_name}.json"
    if results["threshold"] is None or not other_path.exists():
        return
    other = json.loads(other_path.read_text(encoding="utf-8"))
    if other.get("threshold") != results["threshold"]:
        return
    for name, arm in results["arms"].items():
        if name not in other["arms"]:
            continue
        a, b = arm["summary"], other["arms"][name]["summary"]
        if a["none_without_reference"] is None or b["none_without_reference"] is None:
            continue
        total = a["none_questions"] + b["none_questions"]
        kept = a["none_without_reference"] + b["none_without_reference"]
        a["L3_combined"] = {"none_questions_both_halves": total, "shown_no_reference": kept,
                            "threshold": results["threshold"], "pass": kept >= 3}
        a["L3_pass"] = kept >= 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--methods", nargs="+", default=["fused", "bm25", "dense"],
                        help="fused is the registered arm; the others are diagnostics from the same index")
    parser.add_argument("--threshold", type=float, default=None,
                        help="dense similarity below which no lecture reference is shown (L3)")
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()

    questions = load_questions(args.split)
    labels_path = BANK / f"labels-{args.split}.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    manifest = json.loads((BANK / "manifest.json").read_text(encoding="utf-8"))
    registered = manifest[args.split]["digest"]
    recovered = registered_digest(questions)
    if recovered != registered:
        print(f"WARNING: the {args.split} questions no longer match their registration "
              f"({recovered} vs {registered}); the labels may be fine but the questions moved",
              file=sys.stderr)
    else:
        print(f"questions match their registration ({registered})")

    print(f"building the lecture index over {TRANSCRIPTS} ...", flush=True)
    index = LectureIndex.from_dir(TRANSCRIPTS, method="fused")
    recordings = sorted({w.recording for w in index.windows.values()})
    print(f"  {len(index)} windows from {len(recordings)} recordings")
    fused = index.retriever
    bm25, dense = fused._retrievers  # the two halves of the registered fusion, reused as they are
    arms = {"fused": fused, "bm25": bm25, "dense": dense}

    results = {
        "experiment": "E24",
        "split": args.split,
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "questions_file": f"lectures-{args.split}.jsonl",
        "questions_digest": digest(BANK / f"lectures-{args.split}.jsonl"),
        "questions_registered_digest": registered,
        "questions_match_registration": recovered == registered,
        "labels_file": labels_path.name if labels_path.exists() else None,
        "labels_digest": digest(labels_path) if labels_path.exists() else None,
        "index": {"window_s": WINDOW_S, "step_s": STEP_S, "min_words": MIN_WORDS,
                  "windows": len(index), "recordings": len(recordings),
                  "method": "fused (BM25 + bge-small-en-v1.5, reciprocal rank fusion k=60, depth 50)"},
        "tolerance_s": TOLERANCE_S, "depth": DEPTH, "threshold": args.threshold,
        "arms": {},
    }
    for name in args.methods:
        results["arms"][name] = score_method(name, arms[name], index, questions, labels,
                                            args.threshold, dense)

    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_l3(results, out_dir)
    out = out_dir / f"e24-{args.split}.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    for name in args.methods:
        s = results["arms"][name]["summary"]
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"\n{name}: {s['scored']} scored questions"
              + (f" ({len(s['unlabelled_questions'])} still unlabelled: {', '.join(s['unlabelled_questions'])})"
                 if s["unlabelled_questions"] else ""))
        print(f"  hit@1 {fmt(s['hit_at_1'])}  hit@3 {fmt(s['hit_at_3'])}  hit@5 {fmt(s['hit_at_5'])}  "
              f"MRR {fmt(s['mrr'])}  L1 {'pass' if s['L1_pass'] else 'fail' if s['L1_pass'] is not None else '-'}")
        print(f"  top result scored >=2: {fmt(s['top_rubric_at_least_2'])}  "
              f"(top results read {s['top_rubric_read']}/{s['scored']}"
              + (f"; unread: {', '.join(s['top_rubric_unread'])}" if s["top_rubric_unread"] else "") + ")  "
              f"L2 {'pass' if s['L2_pass'] else 'fail' if s['L2_pass'] is not None else '-'}")
        if args.threshold is not None:
            both = s.get("L3_combined")
            verdict = (f"L3 over both halves {both['shown_no_reference']}/{both['none_questions_both_halves']} "
                       f"{'pass' if both['pass'] else 'fail'}" if both else
                       "L3 judged once both halves have run with this threshold")
            print(f"  no-lecture questions shown no reference in this half: {s['none_without_reference']}/{s['none_questions']}; "
                  f"{verdict}; lecture questions losing their reference: "
                  f"{s['lecture_questions_losing_reference'] or 'none'}")
        if name == "fused":
            print("  per question (rank of first hit, top result, dense similarity of the best window):")
            for r in results["arms"][name]["rows"]:
                rank = r["first_hit_rank"] if r["first_hit_rank"] else ("-" if r["relevant_windows"] else "n/a")
                rubric = r["top_rubric"] if r["top_rubric"] is not None else "?"
                print(f"    {r['id']}  {r['expect']:7s} rank {str(rank):>3s}  top {r['top_reference']:<52s} "
                      f"rubric {rubric}  dense {r['dense_top_similarity']:.3f}")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
