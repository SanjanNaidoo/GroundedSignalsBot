"""Generate the worked-solution training pairs and the held-out calculation set.

Two passes over the twenty-seven solution files, split by the recorded seed so that
no file feeds both:

  validation.jsonl  questions with known answer values, from the nine held-out files.
                    These measure whether the fix worked, on questions the model was
                    not trained on and that are not in the frozen benchmark.
  pairs.jsonl       worked question-and-answer pairs from the other eighteen, plus
                    the eligible notes chunks that carry worked examples, to be mixed
                    with the existing short expository pairs.

Every generated question passes the same benchmark filter the original training data
did, so nothing resembling a benchmark item enters either set. Every pair records the
model that wrote it: DeepSeek wrote the first batch and ran out of balance partway,
so the remainder carries a different writer and the report has to say so.

Resumable. Chunks that already have pairs are skipped, so an interrupted run is
finished by running it again.

usage:  python scripts/gen_worked.py [--writer grok] [--validation] [--notes]
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "src")
from ym07.cli import _load_env                                    # noqa: E402
from ym07.config import load_config                               # noqa: E402
from ym07.items import load_items                                 # noqa: E402
from ym07.model import build_client                               # noqa: E402
from ym07.retrieval import dedupe_chunks, load_chunks             # noqa: E402
from ym07.trainingdata import BenchmarkGuard, skip_reason         # noqa: E402
from ym07.worked import (                                         # noqa: E402
    EXTRACT_SYSTEM, WORKED_SYSTEM, checks_from_values, parse_extraction,
    parse_worked, shows_working, solution_chunks, write_validation_set,
)

STEP = re.compile(r"\d[^=\n]{0,40}=\s*-?\d")

parser = argparse.ArgumentParser()
parser.add_argument("--writer", default="deepseek",
                    help="which frontier model writes the pairs; recorded on each one")
parser.add_argument("--validation", action="store_true",
                    help="(re)build the held-out set as well as the training pairs")
parser.add_argument("--notes", action="store_true",
                    help="also generate from eligible notes chunks that work arithmetic out")
args = parser.parse_args()

_load_env()
OUT = Path("training/worked")
config = load_config("configs/eee4114f.yaml")
entry = config.frontier_by_id(args.writer)
client = build_client(False, entry.provider, config, entry)
items = load_items("bench/eee4114f.jsonl", known_topics=config.topic_ids)
chunks = dedupe_chunks(load_chunks("corpus/processed/chunks.jsonl"))
guard = BenchmarkGuard(items, [c.text for c in chunks])

split = json.loads((OUT / "split.json").read_text(encoding="utf-8"))
by_file = solution_chunks(chunks)
held = set(split["validate_files"])
cost = 0.0


def ask(system, chunk):
    reply = client.answer(system=system, user=chunk.text, model=entry.model,
                          max_tokens=config.model.max_tokens, effort="default")
    return chunk, reply


def run(system, targets, parse, label):
    global cost
    rows, failures = [], 0
    print(f"\n{label}: {len(targets)} chunks, written by {entry.model}", flush=True)
    if not targets:
        return rows
    with ThreadPoolExecutor(max_workers=4) as pool:
        for chunk, reply in pool.map(lambda c: ask(system, c), targets):
            cost += (reply.input_tokens * entry.price_in
                     + reply.output_tokens * entry.price_out) / 1e6
            if reply.error:
                failures += 1
                print(f"  failed {chunk.chunk_id}: {reply.error[:90]}", flush=True)
                continue
            if (reply.text or "").strip().upper().startswith("NONE"):
                continue
            for row in parse(reply.text):
                row["source"] = chunk.source
                row["chunk_id"] = chunk.chunk_id
                row["writer"] = entry.model
                rows.append(row)
    print(f"  {len(rows)} produced, {failures} failed requests", flush=True)
    return rows


if args.validation:
    raw = run(EXTRACT_SYSTEM, [c for f in split["validate_files"] for c in by_file[f]],
              parse_extraction, "held-out calculation set")
    validation = []
    for n, row in enumerate(raw, start=1):
        if guard.check(row["question"], ""):
            continue
        item_id = f"worked-{n:03d}"
        validation.append({
            "item_id": item_id, "question": row["question"], "values": row["values"],
            "source": row["source"], "chunk_id": row["chunk_id"], "writer": row["writer"],
            "checks": checks_from_values(item_id, row["values"]),
        })
    print("\nvalidation set:", json.dumps(write_validation_set(validation, OUT / "validation.jsonl")))

# --- training pairs, resumable ------------------------------------------------
path = OUT / "pairs.jsonl"
existing = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] \
    if path.exists() else []
done = {row["chunk_id"] for row in existing}

targets = [c for f in split["train_files"] for c in by_file[f] if c.chunk_id not in done]
if args.notes:
    targets += [c for c in chunks
                if skip_reason(c) is None and c.source not in held
                and c.source not in by_file and c.chunk_id not in done
                and len(STEP.findall(c.text)) >= 3]

raw = run(WORKED_SYSTEM, targets, parse_worked, "worked training pairs")
kept, dropped, thin = [], 0, 0
for row in raw:
    if guard.check(row["question"], row["answer"]):
        dropped += 1
    elif not shows_working(row["answer"]):
        thin += 1          # a pair that works nothing out is what this pass exists
    else:                  # to avoid, whatever its length
        kept.append(row)

with path.open("w", encoding="utf-8", newline="\n") as fh:
    for row in existing + kept:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

total = existing + kept
steps = sorted(len(STEP.findall(r["answer"])) for r in total)
print(f"\nworked pairs: {len(total)} ({len(kept)} new this run)")
print(f"  dropped by the benchmark filter: {dropped}")
print(f"  dropped for showing no working:  {thin}")
if steps:
    print(f"  median arithmetic steps shown:   {steps[len(steps)//2]} "
          f"(the existing training answers show 0)")
writers = {}
for row in total:
    writers[row["writer"]] = writers.get(row["writer"], 0) + 1
print(f"  written by: {writers}")
print(f"\ncost this run ${cost:.2f}")
