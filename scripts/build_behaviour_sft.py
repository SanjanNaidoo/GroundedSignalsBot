"""The training set for the behaviour adapter: what it teaches and what it leaves out.

Three behaviours are wanted, and the examples for each come from a different place:

  refuse assessed work   47 existing examples where the student says the work is for
                         marks. The deterministic gate already withholds the answer in
                         18 of 18, so this is not what makes the policy hold - it is so
                         the model declines of its own accord instead of being overruled,
                         which on the last measurement it was in 12 of 18.

  course information     190 generated examples, built by scripts/build_course_training.py
                         with the handout values scrambled, so what is learned is reading
                         the supplied block rather than the 2026 dates.

  cite the passages      383 existing examples whose answers carry [S1]-style markers.
                         The chosen model cites in 1 answer out of 132 where the model it
                         replaces cited in 83%, and a citation is the only visible
                         evidence a student has that an answer came from the course.

What it leaves out is the 457 examples whose questions are numerical. The router turns
the adapter off for those, so training on them cannot help at inference - and they are
short, direct arithmetic answers, which is the most plausible mechanism for the damage
the first adapter did to arithmetic. Training only on what the adapter is actually
applied to makes the training distribution match the inference distribution, which it
did not before.

    python scripts/build_behaviour_sft.py --out training/behaviour-sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.config import load_config  # noqa: E402
from ym07.hybrid import is_assessed, question_of, route  # noqa: E402
from ym07.prompts import system_prompt  # noqa: E402

# A completion that turns a question away because it belongs to another course. Seventeen
# of these were being trained, and the scope restriction they teach was withdrawn on
# 6 October - the acceptance test first, then the switch in the config on 8 October. They
# have to come out of the training set too, or the behaviour stays in the weights however
# the config is set: the reply that prompted this reproduced fourteen words of one of
# them verbatim on a question the handout answers.
#
# Four separate patterns were tried against the 715 and all four select the same 17, so
# the set is not sensitive to how it is described.
SCOPE_REFUSAL = re.compile(
    r"belongs (in|to) (an?|another)|comes from (an?|another)|question for a .{0,40}course|"
    r"you'd study it in|belongs to a .{0,40}course|isn't covered in EEE4114F|"
    r"is not covered in EEE4114F|doesn't cover this|does not cover this|"
    r"not part of EEE4114F|outside EEE4114F", re.I)


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def asked(row: dict) -> str:
    users = [m for m in row["prompt"] if m["role"] == "user"]
    return question_of(users[-1]["content"]) if users else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", default="training/sft-v3.jsonl")
    parser.add_argument("--course", default="training/course/pairs.jsonl")
    parser.add_argument("--out", default="training/behaviour-sft.jsonl")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    existing = load(ROOT / args.existing)
    course = load(ROOT / args.course)
    if not existing:
        print(f"no examples at {args.existing}", file=sys.stderr)
        return 1

    # Only what the adapter is applied to at inference.
    kept = [r for r in existing if route(asked(r)) == "tuned"]
    dropped = len(existing) - len(kept)

    # Out, with the restriction they teach.
    refusing_scope = [r for r in kept if SCOPE_REFUSAL.search(r["completion"][0]["content"])]
    kept = [r for r in kept if r not in refusing_scope]

    # The system prompt is rebuilt from the config rather than taken from the stored
    # examples, so that what the adapter is trained under is what the server sends. The
    # two differed by exactly one line - the scope clause - and training under a clause
    # the server no longer sends is how a withdrawn behaviour survives in the weights.
    system = system_prompt(load_config(ROOT / args.config), grounded=True)
    for row in kept:
        row["prompt"] = [{"role": "system", "content": system} if m["role"] == "system"
                         else m for m in row["prompt"]]

    # pair_id is what the trainer splits on, and chunk_id - where there is one - is what
    # keeps every example drawn from the same passage on the same side of that split, so
    # a near-duplicate cannot sit in training and evaluation at once. Both are carried
    # through rather than reduced to prompt and completion.
    out = [{k: v for k, v in r.items()
            if k in ("prompt", "completion", "pair_id", "chunk_id", "tokens")}
           for r in kept]
    # The course examples have no passage behind them, so the fact they ask about plays
    # that part: the four scrambled copies of one phrasing differ only in their values,
    # and splitting them apart would put a near-identical example on each side. Grouping
    # by fact also makes the evaluation side a fact the adapter never trained on, which
    # is the stronger check - reading the block has to generalise, not just be recalled.
    for n, r in enumerate(course):
        out.append({"prompt": r["prompt"], "completion": r["completion"],
                    "pair_id": f"course-{r['fact']}-{n:03d}",
                    "chunk_id": f"course-fact-{r['fact']}"})

    rng = random.Random(args.seed)
    rng.shuffle(out)

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                    encoding="utf-8", newline="\n")

    refusals = sum(1 for r in kept if is_assessed(asked(r)))
    citing = sum(1 for r in kept if "[S" in r["completion"][0]["content"])
    print(f"wrote {len(out)} examples to {path.relative_to(ROOT)}\n")
    print(f"  from {args.existing}      {len(kept)}  (of {len(existing)}; "
          f"{dropped} numerical and {len(refusing_scope)} scope refusals left out)")
    print(f"    of which refuse assessed work {refusals}")
    print(f"    of which cite a passage       {citing}")
    print(f"  from {args.course}  {len(course)}  (course information)")
    still = [r for r in out if SCOPE_REFUSAL.search(r["completion"][0]["content"])]
    clause = [r for r in out if "belongs to another course" in json.dumps(r["prompt"])]
    print(f"\n  scope refusals remaining        {len(still)}")
    print(f"  carrying the scope clause       {len(clause)}")
    if still or clause:
        print("the scope restriction is still in the training set", file=sys.stderr)
        return 1
    kinds: dict[str, int] = {}
    for r in course:
        kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
    print(f"  course examples by kind         {kinds}")
    longest = max(len(json.dumps(r)) for r in out)
    print(f"\n  longest example about {longest // 4} tokens, so --max-length should be "
          f"at least that")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
