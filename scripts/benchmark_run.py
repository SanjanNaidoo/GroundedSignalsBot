"""The 47 benchmark items, answered by the delivered pipeline.

`ym07 run` answers the benchmark through the plain runner, which is the right thing for
the hosted arm and the September conditions but is not the system handed over in
October: it has no memory fallback, no citation check, and it answers a question about
the course from the transcribed facts rather than from the handout's pages. The
held-out suites of 9 and 10 October went through `stress.LocalRunner`, which is the
pipeline the browser also runs. This asks the benchmark questions through that runner,
so the delivered system's benchmark answers are produced by the same code as every
other October figure.

What it writes is a run directory of the same shape `ym07 run` writes - `responses.jsonl`
and `manifest.json` - so that `ym07 grade`, `scripts/export_for_review.py` and
`scripts/final_results.py` read it unchanged. Three fields are added to each record
beside the answer: the route the question took, the rung of the fallback ladder that
served it, and any citation the check removed.

    python scripts/benchmark_run.py --conditions A-r1-tuned A-r1-distill-7b

Settings are the held-out run's: a 15.5 GB cap on the card and a 6,144-token
generation cap. Each item is asked once, alone, with no conversation, as the benchmark
has always been asked.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07 import __version__                      # noqa: E402
from ym07.conditions import get_condition         # noqa: E402
from ym07.config import load_config               # noqa: E402
from ym07.items import load_items                 # noqa: E402
from ym07.retrieval import build_retriever        # noqa: E402
from ym07.runner import adapter_summary, looks_degenerate  # noqa: E402
from ym07.stress import LocalRunner               # noqa: E402
from ym07.cli import _load_env                    # noqa: E402

_load_env()  # the API key, if any, lives in .env, which git ignores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["A-r1-tuned"])
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--items", default="bench/eee4114f.jsonl")
    parser.add_argument("--chunks", default="corpus/processed/chunks.jsonl")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--cap-gb", type=float, default=15.5)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--allow-paid", action="store_true",
                        help="run hosted conditions through ym07.hosted; refused otherwise")
    parser.add_argument("--max-usd", type=float, default=10.0, help="spend ceiling per hosted condition")
    parser.add_argument("--dry-run", action="store_true",
                        help="hosted conditions use the stub client: nothing sent, nothing spent")
    args = parser.parse_args()

    config = load_config(args.config)
    item_set = load_items(args.items, known_topics={t.id for t in config.topics})
    retrievers: dict[str, object] = {}

    for name in args.conditions:
        condition = get_condition(name)
        hosted = condition.resolved_provider(config) != "local"
        if hosted and not args.allow_paid:
            print(f"{name} runs on a paid API; pass --allow-paid and --max-usd to run it.",
                  file=sys.stderr)
            return 1
        corpus = (condition.chunks or args.chunks) if condition.use_retrieval else None
        retriever = None
        if corpus is not None:
            if corpus not in retrievers:
                retrievers[corpus] = build_retriever(corpus, config.retrieval.method)
            retriever = retrievers[corpus]

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(args.runs) / f"{stamp}-{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{name} -> {run_dir}")

        if hosted:
            from ym07.hosted import HostedRunner

            runner = HostedRunner(config=config, condition_name=name, retriever=retriever,
                                  corpus=corpus, max_usd=args.max_usd, dry_run=args.dry_run)
        else:
            runner = LocalRunner(config=config, condition_name=name, retriever=retriever,
                                 cap_gb=args.cap_gb, max_new_tokens=args.max_new_tokens,
                                 corpus=corpus)
        records: list[dict] = []
        started = time.monotonic()
        try:
            for done, item in enumerate(item_set.items, start=1):
                answer = runner.answer(item.question)
                client = runner.client
                record = {
                    "response_id": f"{run_dir.name}::{item.id}",
                    "item_id": item.id,
                    "topic": item.topic,
                    "type": item.type,
                    "difficulty": item.difficulty,
                    "condition": name,
                    "question": item.question,
                    "retrieved": [
                        {"tag": f"S{i}", "chunk_id": chunk_id, "source": source}
                        for i, (chunk_id, source) in enumerate(
                            _sources(runner, retriever), start=1)
                    ],
                    "answer": answer.to_dict(),
                    "route": getattr(client, "last_route", None),
                    "served": runner.last_served,
                    "unsupported_citations_removed": runner.last_unsupported,
                }
                records.append(record)
                flag = " ERROR" if answer.error else ""
                print(f"  [{done}/{len(item_set)}] {item.id}  {answer.latency_s:6.1f}s  "
                      f"{record['route']}{flag}", flush=True)
        finally:
            runner.close()

        with (run_dir / "responses.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        errors = [r for r in records if r["answer"].get("error")]
        at_cap = [r for r in records if r["answer"].get("stop_reason") == "max_tokens"]
        degenerate = [r for r in at_cap if looks_degenerate(r["answer"].get("text") or "")]
        model_id = condition.resolved_model(config)
        manifest = {
            "condition": name,
            "description": condition.description,
            "purpose": condition.purpose,
            "started_utc": stamp,
            "ym07_version": __version__,
            "python": platform.python_version(),
            "config_path": str(config.path),
            "config_digest": config.digest,
            "items_path": str(item_set.path),
            "items_digest": item_set.digest,
            "item_count": len(item_set),
            "model": model_id,
            "adapter": adapter_summary(model_id),
            "provider": condition.resolved_provider(config),
            "dry_run": args.dry_run,
            "endpoint": None,
            # The hosted arms run at the condition's effort; the local arm has no such
            # setting. The two runs of 12 October were written before this line changed
            # and say "n/a" although they ran at "high" (conditions.py).
            "effort": condition.resolved_effort(config) if hosted else "n/a",
            "max_tokens": args.max_new_tokens,
            "cap_gb": args.cap_gb,
            "runner": ("ym07.hosted.HostedRunner (the delivered pipeline around a hosted "
                       "model: handout pages, gate, citation check)" if hosted else
                       "stress.LocalRunner (the delivered pipeline: fallback ladder, "
                       "citation check, handout pages, gate)"),
            "use_course_prompt": condition.use_course_prompt,
            "use_retrieval": condition.use_retrieval,
            "top_k": condition.resolved_top_k(config) if condition.use_retrieval else None,
            "retriever": config.retrieval.method if condition.use_retrieval else None,
            "corpus_chunks": len(retriever) if retriever else 0,
            "gate": condition.gate,
            "calculate": condition.calculate,
            "route_adapter": condition.route_adapter,
            "overturn": condition.overturn,
            "check_citations": condition.check_citations,
            "course_source": condition.course_source,
            "errors": len(errors),
            "refusals": 0,
            "truncated": len(at_cap) - len(degenerate),
            "degenerate": len(degenerate),
            "degenerate_items": [r["item_id"] for r in degenerate],
            "answers_served_degraded": runner.trimmed,
            "wall_clock_s": round(time.monotonic() - started, 1),
            "total_cost_usd": round(sum(r["answer"].get("cost_usd", 0.0) for r in records), 4),
            "total_input_tokens": sum(r["answer"]["input_tokens"] for r in records),
            "total_output_tokens": sum(r["answer"]["output_tokens"] for r in records),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
        if runner.system:
            (run_dir / "system_prompt.txt").write_text(runner.system, encoding="utf-8",
                                                      newline="\n")
        latencies = sorted(r["answer"]["latency_s"] for r in records if not r["answer"].get("error"))
        median = latencies[len(latencies) // 2] if latencies else float("nan")
        print(f"  done: {len(records)} answers, {len(errors)} errors, "
              f"{manifest['truncated']} cut off, median {median:.1f} s, "
              f"{runner.trimmed} served degraded")
    return 0


def _sources(runner: LocalRunner, retriever) -> list[tuple[str, str]]:
    """The chunks the answering attempt was given, with their citations."""
    if retriever is None:
        return []
    by_id = getattr(retriever, "_by_id", None)
    if by_id is None:
        by_id = {chunk.chunk_id: chunk for chunk in retriever.chunks}
        try:
            retriever._by_id = by_id
        except Exception:
            pass
    out = []
    for chunk_id in runner.last_retrieved:
        chunk = by_id.get(chunk_id)
        out.append((chunk_id, chunk.citation() if chunk is not None else chunk_id))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
