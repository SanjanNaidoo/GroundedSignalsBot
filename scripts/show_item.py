"""One question, end to end: what was asked, what came back, and how it was marked.

The comparison tables say a model got 83% of the rules. This says why, for a single
question, in the terms the marking actually used - the question as the model received
it, the passages retrieved for it, the route it took through the pipeline, what the
calculator computed, the answer, and each rule with the pattern it was looking for and
whether the answer satisfied it.

    python scripts/show_item.py partial_fractions-000
    python scripts/show_item.py convolution-000 --models C2-oss-7b A-r1-distill-7b
    python scripts/show_item.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

BANK = ROOT / "stress" / "bank" / "arena" / "calc-bank.jsonl"
RESULTS = ROOT / "stress" / "results" / "arena" / "calc"


def load_bank() -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(line) for line in
            BANK.read_text(encoding="utf-8").splitlines() if line.strip())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("item", nargs="?", help="an item id, e.g. convolution-000")
    parser.add_argument("--models", nargs="*", default=None,
                        help="which result files to show; default is all of them")
    parser.add_argument("--list", action="store_true", help="list the item ids")
    parser.add_argument("--full", action="store_true", help="do not truncate the answers")
    args = parser.parse_args()

    bank = load_bank()
    if args.list or not args.item:
        print(f"{len(bank)} items in {BANK.relative_to(ROOT)}:\n")
        for key, item in bank.items():
            print(f"  {key:26} {item['style']:12} {item['question'][:74]}")
        return 0

    if args.item not in bank:
        print(f"no item '{args.item}'. Use --list to see them.")
        return 1
    item = bank[args.item]

    print("=" * 78)
    print(f"ITEM  {item['id']}        family={item['family']}  phrasing={item['style']}")
    print("=" * 78)
    print("\nTHE QUESTION, exactly as every model received it:\n")
    print("  " + item["question"].replace("\n", "\n  "))

    print("\nTHE ANSWER KEY (computed by the generator, not written by hand):\n")
    print("  " + json.dumps(item["key"]))

    print("\nTHE RULES, and what each looks for:\n")
    for check in item["checks"]:
        print(f"  [{check['kind']}] {check['label']}")
        if check.get("pattern"):
            print(f"      pattern: {check['pattern'][:150]}")
        elif check.get("value") is not None:
            print(f"      value  : {json.dumps(check['value'])}")

    wanted = args.models or sorted(p.stem for p in RESULTS.glob("*.jsonl"))
    for name in wanted:
        path = RESULTS / f"{name}.jsonl"
        if not path.exists():
            continue
        row = next((r for r in (json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines() if line.strip())
                    if r["id"] == args.item), None)
        print("\n" + "-" * 78)
        print(f"{name}")
        print("-" * 78)
        if row is None:
            print("  not run")
            continue
        if row.get("error"):
            print(f"  UNANSWERED - {row['error'][:200]}")
            continue
        print(f"  route     : {row.get('route')}")
        if row.get("served"):
            # Only present when the first attempt ran the card out of memory, so its
            # absence means the answer got everything it was meant to have.
            print(f"  ran short : answered at '{row['served']}' after the card filled up")
        print(f"  passages  : {len(row.get('retrieved') or [])} sent to the model")
        for hit in (row.get("retrieved") or []):
            label = hit.get("source") if isinstance(hit, dict) else str(hit)
            print(f"      {label}")
        if row.get("computation"):
            print(f"  computed  : {row['computation'][:200]}")
        if row.get("reasoning"):
            print(f"  thought   : {len(row['reasoning'])} characters, not marked")
        print(f"  tokens    : {row.get('tokens')}   latency {row.get('latency_s')}s")
        print(f"\n  MARKED    : score {row.get('score')}")
        for label, passed in (row.get("per_check") or {}).items():
            print(f"      {'PASS' if passed else 'FAIL'}  {label}")
        body = row.get("text") or ""
        shown = body if args.full else body[:1200]
        print("\n  THE ANSWER THAT WAS MARKED:\n")
        print("    " + shown.replace("\n", "\n    "))
        if len(body) > len(shown):
            print(f"    [... {len(body) - len(shown)} more characters; --full to see them]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
