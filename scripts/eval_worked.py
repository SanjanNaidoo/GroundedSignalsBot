"""Measure an adapter on the held-out calculation set.

Two numbers, because the set is small and one of them does not depend on its size:

  accuracy       the fraction of known answer values the model reproduced. Nine
                 questions and twenty-four rules is a coarse instrument - it will
                 catch a large change and miss a small one - so it is reported with
                 its n and not leaned on further than that.
  working shown  the median number of arithmetic steps the model evaluates. This
                 needs no ground truth at all, so the small set does not weaken it,
                 and it measures the thing the retraining actually changes: v2 shows
                 none, and not showing the working is not doing it.

Answers are produced exactly as C2-oss-tuned produces them - same prompt, same fused
retriever, same four passages - so the only difference between a before and an after
run is the adapter.

usage:  python scripts/eval_worked.py --adapter models/qwen2.5-7b-eee4114f --label v2
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:256")
import torch  # noqa: E402

# 15 of the card's 16 GB. The last two of nine questions failed at 14: not because
# they are longer - they are not - but because the allocator has fragmented by then
# and cannot find a contiguous block. Releasing the cache between questions did not
# recover it. The cap stays below the card so that an oversized request fails here
# rather than being backed by system RAM, which is what Windows does otherwise and
# what once turned a 37-second training step into twenty minutes.
CAP_GB = float(os.environ.get("YM07_VRAM_CAP_GB", "15.0"))
torch.cuda.set_per_process_memory_fraction(
    min(1.0, CAP_GB * 1e9 / torch.cuda.get_device_properties(0).total_memory), 0)

sys.path.insert(0, "src")
from ym07.checks import Check, score_answer                       # noqa: E402
from ym07.conditions import get_condition                         # noqa: E402
from ym07.config import load_config                               # noqa: E402
from ym07.model import build_client                               # noqa: E402
from ym07.prompts import system_prompt, user_prompt               # noqa: E402
from ym07.retrieval import build_retriever                        # noqa: E402

STEP = re.compile(r"\d[^=\n]{0,40}=\s*-?\d")
OUT = Path("training/worked/results")

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", required=True)
parser.add_argument("--label", required=True, help="v2, v3 - names the result file")
parser.add_argument("--set", default="training/worked/validation.jsonl")
args = parser.parse_args()

config = load_config("configs/eee4114f.yaml")
condition = get_condition("C2-oss-tuned")
retriever = build_retriever("corpus/processed/chunks.jsonl", config.retrieval.method)
client = build_client(False, "local", config)
system = system_prompt(config, grounded=True)
top_k = condition.resolved_top_k(config)

rows = [json.loads(l) for l in Path(args.set).read_text(encoding="utf-8").splitlines() if l.strip()]
results = []
for n, row in enumerate(rows, start=1):
    # Prompt lengths vary and the allocator's cache fragments across them; two of the
    # nine ran out of memory at the 14 GB cap on the first pass. Releasing the cache
    # between questions costs a little speed and makes the run complete.
    torch.cuda.empty_cache()
    hits = retriever.search(row["question"], top_k)
    answer = client.answer(system=system, user=user_prompt(row["question"], hits),
                           model=args.adapter, max_tokens=config.model.max_tokens,
                           effort=condition.resolved_effort(config))
    if answer.error:
        print(f"  {row['item_id']}: ERROR {answer.error[:90]}", flush=True)
        continue
    checks = [Check(item_id=row["item_id"], **c) for c in row["checks"]]
    scored = score_answer(checks, answer.text)
    steps = len(STEP.findall(answer.text))
    results.append({"item_id": row["item_id"], "correctness": scored["correctness"],
                    "steps": steps, "tokens": answer.output_tokens,
                    "per_check": scored["per_check"], "text": answer.text})
    print(f"  {n}/{len(rows)} {row['item_id']}  {scored['correctness']:.2f}  "
          f"{steps} steps  {answer.output_tokens} tokens", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{args.label}.json").write_text(
    json.dumps({"adapter": args.adapter, "set": args.set, "results": results}, indent=2),
    encoding="utf-8", newline="\n")

acc = sum(r["correctness"] for r in results) / len(results)
steps = sorted(r["steps"] for r in results)
tokens = sorted(r["tokens"] for r in results)
print(f"\n{args.label}  ({args.adapter})")
print(f"  accuracy on {len(results)} held-out questions   {acc:.3f}")
print(f"  median arithmetic steps shown            {steps[len(steps)//2]}")
print(f"  median tokens                            {tokens[len(tokens)//2]}")
