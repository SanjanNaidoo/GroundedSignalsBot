"""E33's FX4: nothing asked without a conversation before it changes (protocols/followup-fix.md).

For every single-turn held-out item, every first turn of a conversation and every
benchmark item, the search text and the user prompt are built with the E33 switches on
and off and compared byte for byte. The search text decides the passages, so equal search
texts and an equal prompt builder mean an equal prompt; the prompt is built here on no
passages to show the builder itself is unchanged. For the second turns of the 54 held-out
follow-ups the same comparison counts what the change does change, using the first
replies saved in heldout-gatefix.

    .venv\\Scripts\\python.exe scripts\\followup_fix_check.py
"""

from __future__ import annotations

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

from ym07 import prompts, retrieval, stress           # noqa: E402
from ym07.items import load_items                     # noqa: E402

OUT = ROOT / "stress" / "results" / "followup-fix" / "fx4-check.json"


def built(message: str, history) -> tuple[str, str]:
    query = retrieval.conversation_query(message, history)
    prompt = prompts.user_prompt(message, [], history or None)
    return query, prompt


def both(message: str, history=None) -> tuple[tuple[str, str], tuple[str, str]]:
    retrieval.CONVERSATION_QUERY, prompts.HISTORY_WITHOUT_TAGS = False, False
    before = built(message, history)
    retrieval.CONVERSATION_QUERY, prompts.HISTORY_WITHOUT_TAGS = True, True
    after = built(message, history)
    return before, after


def main() -> int:
    os_cwd = Path.cwd()
    import os

    os.chdir(ROOT)
    single: list[tuple[str, str]] = []
    for suite, split in (("calc", "fresh"), ("concept", "final"), ("errors", "final"),
                         ("scope", "final"), ("course", "final")):
        bank = stress.suite_items(suite, split, stress.calcsuites.load_set(stress.BANKS["fresh"]["calc"])
                                  if suite == "calc" else [])
        for item in bank:
            single.append((f"{suite}:{item['id']}", item.get("question") or item.get("message")))
    calc_final = stress.calcsuites.load_set(stress.BANKS["final"]["calc"])
    for item in stress.pushback_items(calc_final):
        single.append((f"pushback-first:{item['id']}", item["turns"][0]))
    followups = stress.suite_items("followup", "final", [])
    for item in followups:
        single.append((f"followup-first:{item['id']}", item["turns"][0]))
    for item in load_items(ROOT / "bench" / "eee4114f.jsonl"):
        single.append((f"benchmark:{item.id}", item.question))

    changed = [name for name, message in single if len(set(both(message))) != 1]

    saved = {json.loads(line)["id"]: json.loads(line) for line in
             (ROOT / "stress/results/heldout-gatefix/followup/A3-opus-pipeline.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip()}
    second_query = second_prompt = 0
    for item in followups:
        history = [("Student", item["turns"][0]), ("Tutor", saved[item["id"]]["replies"][0])]
        (q0, p0), (q1, p1) = both(item["turns"][1], history)
        second_query += q0 != q1
        second_prompt += p0 != p1

    result = {"single_turn_items": len(single), "single_turn_changed": len(changed),
              "changed_ids": changed, "followup_second_turns": len(followups),
              "second_turn_search_text_changed": second_query,
              "second_turn_prompt_changed (Opus 5's saved first replies)": second_prompt}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"FX4: {len(changed)} of {len(single)} single-turn and first-turn items change "
          f"({'met' if not changed else 'MISSED'})")
    print(f"second turns: search text changes on {second_query} of {len(followups)}, "
          f"prompt on {second_prompt} of {len(followups)}")
    print(f"written {OUT}")
    os.chdir(os_cwd)
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
