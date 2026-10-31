"""Cut the past papers into question units with their memoranda, and summarise the result.

    python scripts/build_problems.py            # writes corpus/processed/problems.jsonl
    python scripts/build_problems.py --show classtest2019.pdf   # print one paper's units

The units are what the similar-problem index of E25 is built on (protocols/problems.md).
The summary printed here - units per kind, links accepted, number-only, file-only and
none - is the L2 figure the protocol reports but does not decide on.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.problems import RAW, build, save  # noqa: E402

OUT = ROOT / "corpus" / "processed" / "problems.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", default=None, help="print the units of this paper file")
    parser.add_argument("--text", type=int, default=240, help="characters of each unit to print with --show")
    args = parser.parse_args()

    units = build(ROOT / RAW)
    if args.show:
        for u in units:
            if u.source == args.show or (u.memo or {}).get("file") == args.show:
                memo = u.memo or {}
                print(f"[{u.unit_id}] p.{u.page} q{u.number} marks {u.marks} memo {memo.get('link', 'none')}"
                      + (f" ({memo.get('file')} p.{memo.get('page')}, overlap {memo.get('overlap')})" if memo.get("file") else ""))
                print("   " + u.text[: args.text].replace("\n", " | "))
        return 0

    save(units, OUT)
    kinds = collections.Counter(u.kind for u in units)
    links = collections.Counter((u.memo or {}).get("link", "none") for u in units)
    files = collections.Counter(u.source for u in units)
    print(f"{len(units)} question units from {len(files)} files -> {OUT}")
    print("  by kind:", dict(kinds))
    print("  memorandum links:", dict(links))
    for name, n in sorted(files.items()):
        print(f"    {n:2d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
