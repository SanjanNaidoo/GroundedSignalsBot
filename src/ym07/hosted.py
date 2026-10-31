"""The delivered pipeline around a hosted model, for the premium-model arms.

The stress suites refuse paid conditions by design, because September's budget was
spent and a refusal in the software is safer than a rule in a file. The brief of
11 October approves Claude Opus 5, so this runner exists to put it through exactly the
pipeline the local system runs - the same retrieval and handout facts, the same prompt
builder, the same assessed-work gate, the same citation check - so that a difference
between the two arms is the model and nothing else. It is used only when the caller
passes `--allow-paid`, and it stops at a spend ceiling.

What is the same as `stress.LocalRunner`: `passages_for`, the parsed handout facts, the
`user_prompt` call with its handout and assessed-work blocks, the citation check against
the passages actually sent. What differs, and why:

- no memory cap and no fallback ladder, because the model is not on the card; every
  answer is served from the full context and the row says so;
- no router and no adapter, because there is nothing to switch; the gate still applies,
  as a wrapper that replaces a reply which answers assessed work anyway, since that
  guarantee was never the weights' to give;
- a running total of cost, from the token counts the client returns, with a ceiling the
  run will not cross. The suites append each row as it is produced and resume from the
  rows already written, so a run stopped at the ceiling loses nothing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from . import citations
from .conditions import get_condition
from .handout import read_facts
from .hybrid import HELD_BACK, held_back, is_assessed, question_of, student_said, student_turns
from .model import Answer, build_client
from .prompts import system_prompt, user_prompt
from .retrieval import about_the_course, conversation_query, passages_for

MAX_HISTORY_TURNS = 6
# September's Opus 5 means, with four 800-token passages added: what one answer costs
# before the run says anything, so the ceiling can be set with a number in front of it.
ESTIMATE_IN_TOKENS = 590 + 2400
ESTIMATE_OUT_TOKENS = 1610


def estimate_usd(answers: int, price_in: float = 5.0, price_out: float = 25.0) -> float:
    return answers * (ESTIMATE_IN_TOKENS * price_in + ESTIMATE_OUT_TOKENS * price_out) / 1e6


class GatedClient:
    """A hosted client with the deterministic assessed-work gate of the local system.

    Reads the student's own words, never the passages (that mistake cost the first
    system 125 calculations), and replaces a reply that answers work the student said
    was for marks with the same fixed reply the local system gives.
    """

    def __init__(self, client, *, gate: bool = True):
        self._client = client
        self.gate = gate
        self.last_route = "hosted"
        self.last_gate = None
        self.last_computation = None

    def answer(self, *, system, user, model, max_tokens, effort) -> Answer:
        asked = question_of(user)
        said = student_said(user)
        self.last_route = "hosted"
        reply = self._client.answer(system=system, user=user, model=model,
                                    max_tokens=max_tokens, effort=effort)
        if self.gate and not reply.error and is_assessed(asked, said) and not held_back(reply.text):
            self.last_route = "hosted (answer withheld)"
            return replace(reply, text=HELD_BACK)
        return reply


class HostedRunner:
    """One hosted condition's pipeline, with the surface `run_suite` expects."""

    def __init__(self, *, config, condition_name: str, retriever, cap_gb: float = 0.0,
                 adapter_scale: float | None = None, max_new_tokens: int | None = None,
                 corpus: str | None = None, max_usd: float = 0.0, dry_run: bool = False):
        condition = get_condition(condition_name)
        provider = condition.resolved_provider(config)
        if provider == "local":
            raise SystemExit(f"{condition_name} is a local condition; use LocalRunner")
        self.config = config
        self.condition = condition
        self.retriever = retriever if condition.use_retrieval else None
        self.model_id = condition.resolved_model(config)
        self.top_k = condition.resolved_top_k(config)
        self.system = (system_prompt(config, grounded=condition.resolved_grounded())
                       if condition.use_course_prompt else "")
        self.handout_facts = (read_facts(retriever.chunks, config.handout)
                              if retriever is not None else ())
        base = build_client(dry_run, provider, config)
        self.client = GatedClient(base, gate=condition.gate)
        self.adapter_scale = 1.0
        self.label = condition.name
        self.corpus = Path(corpus).stem if corpus else ""
        self.last_retrieved: list[str] = []
        self.last_served = ""          # always the full context: nothing is ever given up
        self.last_unsupported: list[int] = []
        self.last_ungrounded = False
        self.trimmed = 0
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self.answers = 0
        self.dry_run = dry_run

    def answer(self, message: str, history: list[tuple[str, str]] | None = None) -> Answer:
        if self.max_usd and self.spent_usd >= self.max_usd:
            raise SystemExit(f"spend ceiling ${self.max_usd:.2f} reached after {self.answers} "
                             f"answers (${self.spent_usd:.2f}); the rows written so far stand "
                             "and the run resumes from them")
        hits = passages_for(message, self.retriever, code=self.config.code,
                            top_k=self.top_k,
                            handout=self.condition.course_handout(self.config),
                            search_course=self.condition.course_search,
                            query=conversation_query(message, history))
        said = student_turns(history, message)
        reply = self.client.answer(
            system=self.system,
            user=user_prompt(message, hits, (history or [])[-MAX_HISTORY_TURNS * 2:] or None,
                             facts=(self.condition.course_facts(self.config, self.handout_facts)
                                    if about_the_course(message) else ()),
                             handout_pages=bool(hits) and all(
                                 h.chunk.source == self.config.handout for h in hits),
                             assessed=is_assessed(message, said)),
            model=self.model_id,
            max_tokens=self.config.model.max_tokens,
            effort=self.condition.resolved_effort(self.config),
        )
        self.answers += 1
        self.spent_usd += reply.cost_usd
        self.last_retrieved = [hit.chunk.chunk_id for hit in hits]
        self.last_unsupported = []
        self.last_ungrounded = False
        if self.condition.check_citations and not reply.error:
            text, removed, ungrounded = citations.checked(reply.text, hits)
            self.last_unsupported = removed
            self.last_ungrounded = ungrounded
            if removed:
                return replace(reply, text=text)
        return reply

    def close(self) -> None:
        print(f"  {self.label}: {self.answers} answers, ${self.spent_usd:.2f}"
              + (" (dry run, nothing spent)" if self.dry_run else ""))
