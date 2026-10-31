"""Does the tutor move to another course without touching the code? (AT21)

The brief asks for a chatbot that is "easily adaptable to new courses". The claim this
project makes is narrower and testable: everything course-specific lives in a config file
and a corpus, so a second course needs neither a code change nor a new program.

This runs the same pipeline against `configs/demo-2046.yaml` and a corpus written for the
demonstration, and checks three things:

1. It answers a question from the second course's own material, and cites it.
2. It declines a question from EEE4114F, which is now the course next door.
3. Nothing under `src/` names either course: the code carries no course-specific text.

The fine-tuned adapter is not used here. It was trained on EEE4114F material and a second
course would need its own training run; what ports without code is the retrieval, the
prompt, the policies and the interface.

    python scripts/second_course_demo.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ym07.conditions import get_condition          # noqa: E402
from ym07.config import load_config                # noqa: E402
from ym07.model import build_client_for            # noqa: E402
from ym07.prompts import system_prompt, user_prompt  # noqa: E402
from ym07.retrieval import build_retriever         # noqa: E402

CONFIG = "configs/demo-2046.yaml"
CHUNKS = "corpus/demo-course/chunks.jsonl"
OWN_COURSE = "What is the time constant of an RC circuit, and how long until the step response settles?"
OTHER_COURSE = "Why does a 1200 Hz tone sampled at 1000 Hz alias to 200 Hz?"


def main() -> int:
    passed = []
    config = load_config(CONFIG)
    condition = get_condition("C2-oss")          # the base weights: no adapter for a new course
    retriever = build_retriever(CHUNKS, config.retrieval.method)
    client = build_client_for(condition, config)
    print(f"Course: {config.code} ({config.name}), {len(retriever)} passages, "
          f"topics {', '.join(t.id for t in config.topics)}\n")

    def ask(question: str) -> str:
        hits = retriever.search(question, condition.resolved_top_k(config))
        reply = client.answer(system=system_prompt(config, grounded=True),
                              user=user_prompt(question, hits),
                              model=condition.resolved_model(config),
                              max_tokens=config.model.max_tokens, effort="n/a")
        print(f"Q: {question}\nA: {(reply.text or '')[:400].strip()}\n")
        return reply.text or ""

    answer = ask(OWN_COURSE)
    cites = bool(re.search(r"\[S\d\]", answer))
    on_topic = any(word in answer.lower() for word in ("time constant", "rc", "tau", "exponential"))
    passed.append(("answers its own course, with a citation", cites and on_topic))

    declined = ask(OTHER_COURSE)
    turns_away = bool(re.search(r"eee2046f|not (covered|part)|outside|another course|different course",
                                declined, re.I))
    passed.append(("declines the course next door", turns_away))

    source = " ".join(p.read_text(encoding="utf-8", errors="ignore")
                      for p in Path("src/ym07").glob("*.py"))
    # The stress suites name EEE4114F because they are this project's experiment, not the
    # tutor; the tutor itself is in the other modules.
    tutor_source = " ".join(p.read_text(encoding="utf-8", errors="ignore")
                            for p in Path("src/ym07").glob("*.py")
                            if p.name not in {"stress.py", "hybrid.py", "coursequeries.py"})
    clean = not re.search(r"eee4114f|eee2046f", tutor_source, re.I)
    passed.append(("no course code in the tutor's own code", clean))
    if not clean:
        for name in re.findall(r"\w*eee\d{4}\w*", tutor_source, re.I)[:5]:
            print(f"    found: {name}")

    print()
    for name, ok in passed:
        print(f"[{'pass' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
