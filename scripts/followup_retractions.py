"""How often a follow-up reply disowns the first reply's citations, in the saved suites.

Found in E30 (16 October): asked a follow-up that changes one number, Opus 5 opened with
"a correction to my previous answer", said its earlier citations were wrong, and closed
with the citation check's note that no course material covered the answer. Neither was
true. The follow-up is retrieved on its own words ("what if the coefficient were 0.6"),
so it gets unrelated passages, numbered [S1]-[S4] again; the first reply, tags and all,
is in the history; the model checks the old tags against the new passages and retracts
them, and the citation check, finding nothing in the new passages that supports the
answer, says it came from general knowledge. The follow-up suite marks the new final
value only, so none of this was counted.

This counts the two signs in the second replies of the held-out follow-up suite, as
re-measured with the assessed-work detector fixed (heldout-gatefix). Each match was read.

    .venv\\Scripts\\python.exe scripts\\followup_retractions.py
    .venv\\Scripts\\python.exe scripts\\followup_retractions.py --run stress/results/followup-fix-amended

The second form counts E33's run (protocols/followup-fix.md, FX1) into that run's folder.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.citations import NOT_COURSE_MATERIAL  # noqa: E402

RUN = ROOT / "stress" / "results" / "heldout-gatefix" / "followup"
OUT = ROOT / "stress" / "results" / "platform-live" / "run-2026-10-16" / "followup-retractions.json"
RETRACTS = re.compile(
    r"(?i)correction (to|first|before)|correction to (my|what)|flag something about my previous"
    r"|I cited .{0,80}(previous|earlier)|in my (previous|earlier) (message|answer|turn|reply) I (cited|attached|tagged)")


def main() -> int:
    global RUN, OUT
    if "--run" in sys.argv:
        RUN = ROOT / sys.argv[sys.argv.index("--run") + 1] / "followup"
        OUT = RUN.parent / "followup-retractions.json"
    result = {}
    for path in sorted(RUN.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        second = {r["id"]: (r.get("replies") or ["", ""])[-1] or "" for r in rows}
        retracts = sorted(i for i, t in second.items() if RETRACTS.search(t))
        disclaims = sorted(i for i, t in second.items() if NOT_COURSE_MATERIAL in t)
        either = sorted(set(retracts) | set(disclaims))
        right = {r["id"]: bool(r.get("followup_correct")) for r in rows}
        family = {r["id"]: r.get("family") for r in rows}
        result[path.stem] = {
            "conversations": len(rows), "retracts_own_citations": len(retracts),
            "not_course_material_note": len(disclaims), "either": len(either),
            "either_with_followup_value_right": sum(right[i] for i in either),
            "first_replies_with_note": sum(NOT_COURSE_MATERIAL in ((r.get("replies") or [""])[0] or "")
                                           for r in rows),
            "by_family": dict(Counter(family[i] for i in either)), "ids": either,
        }
        print(f"{path.stem}: {len(either)} of {len(rows)} second replies retract the first reply's "
              f"citations ({len(retracts)}) or carry the not-course-material note ({len(disclaims)}); "
              f"{result[path.stem]['either_with_followup_value_right']} of them still give the new value")
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
