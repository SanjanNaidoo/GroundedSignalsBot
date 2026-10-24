"""The experimental conditions from protocols/evaluation.md.

One variable at a time. The model is held fixed across C0/C1/C2 — it is a
controlled variable, and if it changes between them the comparison means nothing.
C3 conditions each vary exactly one thing away from C2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import CourseConfig


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    purpose: str
    use_course_prompt: bool
    use_retrieval: bool
    model_id: str | None = None      # None = use the config's model
    provider: str | None = None      # None = use the config's provider
    effort: str | None = None        # None = use the config's effort
    top_k: int | None = None         # None = use the config's top_k
    frontier_id: str | None = None   # set = model/provider come from config `frontier:`

    def resolved_model(self, config: CourseConfig) -> str:
        if self.frontier_id:
            return config.frontier_by_id(self.frontier_id).model
        return self.model_id or config.model.id

    def resolved_provider(self, config: CourseConfig) -> str:
        if self.frontier_id:
            return config.frontier_by_id(self.frontier_id).provider
        return self.provider or config.model.provider

    def resolved_effort(self, config: CourseConfig) -> str:
        return self.effort or config.model.effort

    def resolved_top_k(self, config: CourseConfig) -> int:
        return self.top_k if self.top_k is not None else config.retrieval.top_k


BASELINE = Condition(
    name="C0-baseline",
    description="Frontier model, no retrieval, no course prompt",
    purpose=(
        "The honest control. Every later design choice has to beat this number. "
        "If C0 already scores well, the project's contribution is the policy layer, "
        "not the retrieval — and the report must say so."
    ),
    use_course_prompt=False,
    use_retrieval=False,
)

PROMPTED = Condition(
    name="C1-prompted",
    description="Same model, course system prompt, still no retrieval",
    purpose=(
        "Isolates how much comes from prompting alone. The condition most projects "
        "skip and the one an examiner will ask about."
    ),
    use_course_prompt=True,
    use_retrieval=False,
)

RAG = Condition(
    name="C2-rag",
    description="Retrieval over the course corpus + policy prompt",
    purpose="The proposed system.",
    use_course_prompt=True,
    use_retrieval=True,
)

# C3 variants: one variable each, all else equal to C2.
MONEY = Condition(
    name="C3-money",
    description="C2 on a cheaper model tier",
    purpose=(
        "The cost/accuracy trade. Reports what correctness is lost per rand saved — "
        "the question anyone deploying this at a university actually has."
    ),
    use_course_prompt=True,
    use_retrieval=True,
    model_id="claude-sonnet-5",
)

EFFORT = Condition(
    name="C3-effort-low",
    description="C2 at low reasoning effort",
    purpose="Separates the cost of the model tier from the cost of thinking depth.",
    use_course_prompt=True,
    use_retrieval=True,
    effort="low",
)

TOPK = Condition(
    name="C3-topk-3",
    description="C2 retrieving 3 chunks instead of 8",
    purpose="Tests whether extra context helps or dilutes.",
    use_course_prompt=True,
    use_retrieval=True,
    top_k=3,
)

# ---------------------------------------------------------------------------
# The open-weights arm.
#
# A second arm, not a C3 variant. Swapping the model is not "one variable away
# from C2" — it changes the system under test, so it gets its own C0/C1/C2 with
# the same items, corpus, prompt and policy. Read down an arm for the effect of
# retrieval; read across the arms for the effect of the model. A C3 variant that
# changed the model would confound the two.
#
# Effort is recorded as "n/a": these models have no thinking-effort knob, and a
# manifest claiming effort=high for a run where nothing thought would be a lie.
# The pre-quantized checkpoint, not the bf16 one. Quantizing on load
# materialises the full-precision weights in system RAM first, which does not
# fit in 16 GB; this variant streams 4-bit weights straight to the GPU.
OSS_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# The fine-tuned checkpoint. Does not exist until the training run has produced
# it; the condition is declared here so the experiment design is legible now.
OSS_TUNED_MODEL = "models/qwen2.5-7b-eee4114f-v3"

OSS_BASELINE = Condition(
    name="C0-oss",
    description="Open-weights model, no retrieval, no course prompt",
    purpose=(
        "The open arm's control, and the floor the fine-tune has to beat. Pairs "
        "with C0-baseline: the gap between them is what the money buys before any "
        "of this project's work is applied."
    ),
    use_course_prompt=False,
    use_retrieval=False,
    model_id=OSS_MODEL,
    provider="local",
    effort="n/a",
)

OSS_PROMPTED = Condition(
    name="C1-oss",
    description="Open-weights model, course system prompt, no retrieval",
    purpose="Pairs with C1-prompted. Isolates prompting on the smaller model.",
    use_course_prompt=True,
    use_retrieval=False,
    model_id=OSS_MODEL,
    provider="local",
    effort="n/a",
)

OSS_RAG = Condition(
    name="C2-oss",
    description="Open-weights model, retrieval + policy prompt",
    purpose=(
        "The untuned open model in the full pipeline. This is the number the "
        "fine-tune is measured against — without it, any improvement could just "
        "be the retrieval doing the work."
    ),
    use_course_prompt=True,
    use_retrieval=True,
    model_id=OSS_MODEL,
    provider="local",
    # Four passages, not the config's eight. The adapter can only be trained on
    # examples of about 4,000 tokens on the 16 GB GPU, and an eight-passage
    # example is nearly twice that. C2-oss uses the same four as C2-oss-tuned,
    # so the pair still differs only by the adapter.
    top_k=4,
    effort="n/a",
)

OSS_TUNED = Condition(
    name="C2-oss-tuned",
    description="Fine-tuned open-weights model, retrieval + policy prompt",
    purpose=(
        "The proposed cheap system, and the project's primary claim: a fine-tuned "
        "7B with the course corpus beats a frontier model that does not have it. "
        "The bar is C1-prompted, not C2-rag. C2-rag is the ceiling reference — "
        "what the same pipeline does with an unlimited model budget."
    ),
    use_course_prompt=True,
    use_retrieval=True,
    model_id=OSS_TUNED_MODEL,
    provider="local",
    # The depth the adapter was trained at; see C2-oss.
    top_k=4,
    effort="n/a",
)

# ---------------------------------------------------------------------------
# The cross-model ungrounded baseline.
#
# The same bare question — no system prompt, no retrieval — sent to other
# frontier models at each provider's default settings. It is what a student
# gets by pasting the question into a chat box, and it answers a question the
# single-model C0 cannot: which frontier model is the strongest ungrounded
# baseline? That bounds how much of the C1/C2 gap is the course prompt and the
# corpus, versus simply the choice of model.
#
# Effort is recorded as "default" because that is what was used. Every provider
# defaults differently, and overriding it would measure our settings rather
# than their model.
#
# Model ids, endpoints, key names and prices live in configs/<course>.yaml
# under `frontier:`. To add a model, add it there AND to this tuple.
FRONTIER_IDS = ("gpt", "gemini", "grok", "deepseek")


def _frontier_baseline(frontier_id: str) -> Condition:
    return Condition(
        name=f"C0-{frontier_id}",
        description=f"Frontier model '{frontier_id}', no retrieval, no course prompt",
        purpose=(
            "Pairs with C0-baseline across providers. If another model's bare "
            "answer already beats the grounded Claude pipeline, the project's "
            "contribution is smaller than it looks — and the report must say so."
        ),
        use_course_prompt=False,
        use_retrieval=False,
        frontier_id=frontier_id,
        effort="default",
    )


def _frontier_prompted(frontier_id: str) -> Condition:
    return Condition(
        name=f"C1-{frontier_id}",
        description=f"Frontier model '{frontier_id}', course system prompt, no retrieval",
        purpose=(
            "The C0-{id} pair. Prompting is the cheapest intervention this "
            "project makes, and on one model it cannot be told apart from that "
            "model happening to like the prompt. Running it across providers "
            "says whether the course prompt buys anything general."
        ).replace("{id}", frontier_id),
        use_course_prompt=True,
        use_retrieval=False,
        frontier_id=frontier_id,
        # Held at the provider default, exactly as C0-{id} is. The prompt is the
        # only thing allowed to differ between the pair; changing effort as well
        # would leave the gap unattributable to either.
        effort="default",
    )


FRONTIER_BASELINES = tuple(_frontier_baseline(f) for f in FRONTIER_IDS)
FRONTIER_PROMPTED = tuple(_frontier_prompted(f) for f in FRONTIER_IDS)

CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in (
        BASELINE, PROMPTED, RAG, MONEY, EFFORT, TOPK,
        OSS_BASELINE, OSS_PROMPTED, OSS_RAG, OSS_TUNED,
        *FRONTIER_BASELINES,
        *FRONTIER_PROMPTED,
    )
}

# Conditions where a citation-validity grade is meaningful.
CITED_CONDITIONS = {c.name for c in CONDITIONS.values() if c.use_retrieval}


def get_condition(name: str) -> Condition:
    if name not in CONDITIONS:
        raise KeyError(
            f"Unknown condition '{name}'. Known: {', '.join(sorted(CONDITIONS))}"
        )
    return CONDITIONS[name]


def repeat(condition: Condition, index: int) -> Condition:
    """Label a repeat run of the same condition (for variance measurement)."""
    return replace(condition, name=f"{condition.name}-r{index}")
