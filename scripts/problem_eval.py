"""E25c - does a past question of the same kind come back for a student's problem?

    python scripts/problem_eval.py
    python scripts/problem_eval.py --threshold 0.70

Builds the similar-problem index of ym07.problems over the question units, asks it the
twenty registered queries, and scores the answers against the units labelled relevant
(scripts/problem_label.py apply) under the pass marks of protocols/problems.md: a hit is
a labelled unit returned; R1 a hit in the top three for at least 70% of the queries that
have one; R2 the top result scoring at least 2 on the labelling rubric for at least 60%
of those, a top result nobody has read reported as unread; R3 nothing shown above the
no-reference rule for at least 2 of the 3 queries the papers do not cover, the rule a
threshold on the dense similarity of the best unit, set on these twenty and stated.

The fused retriever is the registered arm; BM25 and the dense model alone are run from
the same index as diagnostics. Everything goes to stress/results/problems/e25c.json.
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

from ym07.problems import ProblemIndex, load  # noqa: E402

BANK = ROOT / "stress" / "bank" / "problems"
UNITS = ROOT / "corpus" / "processed" / "problems.jsonl"
RESULTS = ROOT / "stress" / "results" / "problems"
DEPTH = 10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def registered_digest(queries: list[dict]) -> str:
    rows = []
    for q in queries:
        row = {k: v for k, v in q.items() if k != "labelled_candidates"}
        row["relevant_units"] = []
        rows.append(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()[:16]


def score_arm(name, retriever, index, queries, labels, threshold, dense):
    rows = []
    for q in queries:
        relevant = set(q.get("relevant_units", []))
        hits = retriever.search(q["question"], DEPTH)
        ranked = [(index.units[h.chunk.chunk_id], h.score) for h in hits]
        first_hit = next((rank for rank, (u, _) in enumerate(ranked, start=1) if u.unit_id in relevant), None)
        top = ranked[0][0] if ranked else None
        top_label = labels.get(q["id"], {}).get(top.unit_id) if top else None
        dense_top = dense.search(q["question"], 1)[0].score if dense is not None else None
        shown = dense_top is None or threshold is None or dense_top >= threshold
        rows.append({
            "id": q["id"], "kind": q["kind"], "expect": q["expect"], "question": q["question"],
            "relevant_units": len(relevant), "first_hit_rank": first_hit,
            "hit_at_1": first_hit is not None and first_hit <= 1,
            "hit_at_3": first_hit is not None and first_hit <= 3,
            "hit_at_5": first_hit is not None and first_hit <= 5,
            "reciprocal_rank": (1.0 / first_hit) if first_hit else 0.0,
            "top_unit": top.unit_id if top else None, "top_reference": top.reference() if top else None,
            "top_rubric": top_label, "dense_top_similarity": dense_top, "reference_shown": shown,
            "top10": [{"unit": u.unit_id, "reference": u.reference(), "score": round(s, 5), "hit": u.unit_id in relevant}
                      for u, s in ranked],
        })
    scored = [r for r in rows if r["expect"] == "unit" and r["relevant_units"]]
    none_rows = [r for r in rows if r["expect"] == "none"]
    top_read = [r for r in scored if r["top_rubric"] is not None]
    summary = {
        "method": name, "queries": len(rows), "scored": len(scored),
        "unlabelled_queries": [r["id"] for r in rows if r["expect"] == "unit" and not r["relevant_units"]],
        "hit_at_1": sum(r["hit_at_1"] for r in scored) / len(scored) if scored else None,
        "hit_at_3": sum(r["hit_at_3"] for r in scored) / len(scored) if scored else None,
        "hit_at_5": sum(r["hit_at_5"] for r in scored) / len(scored) if scored else None,
        "mrr": sum(r["reciprocal_rank"] for r in scored) / len(scored) if scored else None,
        "top_rubric_read": len(top_read), "top_rubric_unread": [r["id"] for r in scored if r["top_rubric"] is None],
        "top_rubric_at_least_2": (sum(r["top_rubric"] >= 2 for r in top_read) / len(scored)
                                  if scored and len(top_read) == len(scored) else None),
        "none_queries": len(none_rows),
        "none_without_reference": sum(not r["reference_shown"] for r in none_rows) if threshold is not None else None,
        "unit_queries_losing_reference": [r["id"] for r in scored if not r["reference_shown"]] if threshold is not None else None,
    }
    summary["R1_pass"] = summary["hit_at_3"] >= 0.70 if summary["hit_at_3"] is not None else None
    summary["R2_pass"] = summary["top_rubric_at_least_2"] >= 0.60 if summary["top_rubric_at_least_2"] is not None else None
    summary["R3_pass"] = (summary["none_without_reference"] >= 2) if threshold is not None and none_rows else None
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["fused", "bm25", "dense"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()

    queries = [json.loads(l) for l in (BANK / "problems.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels_path = BANK / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    manifest = json.loads((BANK / "manifest.json").read_text(encoding="utf-8"))
    recovered = registered_digest(queries)
    if recovered != manifest["digest"]:
        print(f"WARNING: the queries no longer match their registration ({recovered} vs {manifest['digest']})", file=sys.stderr)
    else:
        print(f"queries match their registration ({manifest['digest']})")

    units = load(UNITS)
    print(f"building the similar-problem index over {len(units)} question units ...", flush=True)
    index = ProblemIndex(units, method="fused")
    fused = index.retriever
    bm25, dense = fused._retrievers
    arms = {"fused": fused, "bm25": bm25, "dense": dense}
    results = {
        "experiment": "E25c", "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "queries_file": "problems.jsonl", "queries_digest": digest(BANK / "problems.jsonl"),
        "queries_registered_digest": manifest["digest"], "queries_match_registration": recovered == manifest["digest"],
        "labels_digest": digest(labels_path) if labels_path.exists() else None,
        "index": {"units": len(units), "units_digest": digest(UNITS),
                  "method": "fused (BM25 + bge-small-en-v1.5, reciprocal rank fusion k=60, depth 50)"},
        "depth": DEPTH, "threshold": args.threshold, "arms": {},
    }
    for name in args.methods:
        results["arms"][name] = score_arm(name, arms[name], index, queries, labels, args.threshold, dense)
    out_dir = Path(args.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "e25c.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    for name in args.methods:
        s = results["arms"][name]["summary"]
        print(f"\n{name}: {s['scored']} scored queries" + (f" (unlabelled: {s['unlabelled_queries']})" if s["unlabelled_queries"] else ""))
        print(f"  hit@1 {fmt(s['hit_at_1'])}  hit@3 {fmt(s['hit_at_3'])}  hit@5 {fmt(s['hit_at_5'])}  MRR {fmt(s['mrr'])}  "
              f"R1 {'pass' if s['R1_pass'] else 'fail' if s['R1_pass'] is not None else '-'}")
        print(f"  top result >= 2: {fmt(s['top_rubric_at_least_2'])} (read {s['top_rubric_read']}/{s['scored']}"
              + (f"; unread {s['top_rubric_unread']}" if s["top_rubric_unread"] else "") + f")  "
              f"R2 {'pass' if s['R2_pass'] else 'fail' if s['R2_pass'] is not None else '-'}")
        if args.threshold is not None:
            print(f"  no-coverage queries shown nothing: {s['none_without_reference']}/{s['none_queries']}  "
                  f"R3 {'pass' if s['R3_pass'] else 'fail'}; unit queries losing their reference: {s['unit_queries_losing_reference'] or 'none'}")
        if name == "fused":
            for r in results["arms"][name]["rows"]:
                rank = r["first_hit_rank"] if r["first_hit_rank"] else ("-" if r["relevant_units"] else "n/a")
                rubric = r["top_rubric"] if r["top_rubric"] is not None else "?"
                print(f"    {r['id']}  {r['expect']:4s} rank {str(rank):>3s}  top {r['top_reference'][:58]:<58s} rubric {rubric}  dense {r['dense_top_similarity']:.3f}")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
