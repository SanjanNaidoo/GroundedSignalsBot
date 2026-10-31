"""The interactive tutor — the deliverable a student would actually use.

Deliberately built on the same parts as the benchmark: the same course config,
the same system prompt, the same retriever, the same model clients. A demo that
reimplemented any of those would be a different system from the one the report
has numbers for, and the numbers would no longer describe the thing being shown.
So `ym07 chat` with a corpus is condition C2; without one it is C1.

Two things the batch harness does not need, and this does:

- **Memory.** Tutoring is a conversation — the scaffolding policy asks the
  student what they have tried, which is worthless if the reply is not
  remembered. History is folded into the user message by the shared
  `user_prompt`, so it travels through the unmodified client interface.
- **Visible cost.** Every turn on the hosted arm spends money. The running total
  is printed so it is never a surprise, and so the difference between the hosted
  and local arms is felt rather than merely reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .conditions import OSS_MODEL
from .config import CourseConfig, load_config
from .model import Answer, build_client
from .retrieval import about_the_course
from .prompts import system_prompt, user_prompt
from .retrieval import LexicalRetriever, build_retriever

BANNER = """\
{code} tutor — {model}
{grounding}

  /sources   what the last answer was grounded in
  /cost      spend so far this session
  /reset     forget the conversation, keep the session
  /quit      leave
"""

# Enough turns to sustain a scaffolded exchange, few enough that the resent
# transcript does not dominate the context (or, on the hosted arm, the bill).
MAX_HISTORY_TURNS = 6


def _wrap(text: str, width: int = 88) -> str:
    import textwrap

    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(out)


def chat(
    *,
    config: CourseConfig,
    retriever: LexicalRetriever | None,
    client,
    model_id: str,
    provider: str,
    out=print,
) -> int:
    grounded = retriever is not None
    system = system_prompt(config, grounded=grounded)

    if grounded:
        grounding = (
            f"Grounded in {len(retriever)} passages of course material "
            f"(top {config.retrieval.top_k} retrieved per question)."
        )
    else:
        grounding = (
            "NO COURSE MATERIAL LOADED — answering from the model's general "
            "knowledge only. Run `ym07 ingest` first for the grounded tutor."
        )

    out(BANNER.format(code=config.code, model=model_id, grounding=grounding))

    history: list[tuple[str, str]] = []
    last_hits: list = []
    spend = 0.0
    turns = 0

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            break

        if not question:
            continue

        if question in ("/quit", "/exit", "/q"):
            break
        if question == "/cost":
            out(f"  {turns} turn(s), ${spend:.4f} this session\n")
            continue
        if question == "/reset":
            history.clear()
            out("  conversation forgotten\n")
            continue
        if question == "/sources":
            if not last_hits:
                out("  nothing retrieved yet\n")
            else:
                for idx, hit in enumerate(last_hits, start=1):
                    out(f"  [S{idx}] {hit.chunk.citation()}  (score {hit.score:.3f})")
                out("")
            continue

        hits = retriever.search(question, config.retrieval.top_k) if grounded else []
        last_hits = hits

        answer: Answer = client.answer(
            system=system,
            user=user_prompt(question, hits, history[-MAX_HISTORY_TURNS * 2:] or None,
                             facts=config.facts if about_the_course(question) else ()),
            model=model_id,
            max_tokens=config.model.max_tokens,
            effort=config.model.effort,
        )

        if answer.error:
            out(f"\n  [error] {answer.error}\n")
            continue

        out("")
        out(_wrap(answer.text))
        out("")

        if hits:
            cited = ", ".join(f"[S{i}]" for i in range(1, len(hits) + 1))
            out(f"  sources available: {cited} — /sources to list them")

        spend += answer.cost_usd
        turns += 1
        if answer.cost_usd:
            out(f"  ${answer.cost_usd:.4f} this turn, ${spend:.4f} total")
        out("")

        history.append(("Student", question))
        history.append(("Tutor", answer.text))

    if spend:
        out(f"\n{turns} turn(s), ${spend:.4f} spent.")
    return 0


def run_chat(args) -> int:
    config = load_config(args.config)
    provider = args.provider or config.model.provider
    model_id = args.model or (OSS_MODEL if provider == "local" else config.model.id)

    retriever = None if args.no_retrieval else build_retriever(args.chunks, config.retrieval.method)
    if retriever is None and not args.no_retrieval and Path(args.chunks).exists():
        print(f"warning: {args.chunks} is empty — running ungrounded.", file=sys.stderr)

    client = build_client(args.dry_run, provider, config)
    return chat(
        config=config,
        retriever=retriever,
        client=client,
        model_id=model_id,
        provider=provider,
    )
