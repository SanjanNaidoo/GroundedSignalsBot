"""The experimental conditions from protocols/evaluation.md.

One variable at a time. The model is held fixed across C0/C1/C2 — it is a
controlled variable, and if it changes between them the comparison means nothing.
C3 conditions each vary exactly one thing away from C2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import CourseConfig

# The behaviour adapter every tuned condition loads. Named once because repointing it
# at a retrained adapter used to mean editing ten conditions, and missing one would
# compare two different models under a single name. v1 is kept addressable by
# A-r1-tuned-nogate-v1; every result directory written before 8 October is v1.
# v2 is the delivered adapter, frozen 8 October. Chosen on the two behaviours that are
# stable across training seeds - 5 of 5 absent facts and 10 of 10 out-of-course questions
# answered, both reproduced on an independent seed - and not on concept accuracy or the
# course-fact count, where the seed alone moves the numbers further than any of the
# training-set changes did. v2 rather than v2b because it was trained first and was
# already partly measured, not because it scored higher: picking the better of two draws
# from an eleven-point band is choosing noise and calling it a result.
ADAPTER = "models/r1-distill-7b-eee4114f-behaviour-v2"
ADAPTER_V2 = "models/r1-distill-7b-eee4114f-behaviour-v2"
# v2 trained a second time: identical examples, identical settings, the shuffle seed
# changed from 0 to 1. Nothing about the model it should produce is different, so the
# distance between this and v2 is how far two training runs drift apart on their own -
# and therefore how much of any difference between v1, v2 and v3 was ever attributable
# to the training data. Run because the three disagreed in a way the data could not
# explain: v3 carried more examples that open with a denial than v1 and answered "false"
# on 6% of the concept suite against v1's 26%, which is not a direction the change could
# produce.
ADAPTER_V2B = "models/r1-distill-7b-eee4114f-behaviour-v2b"
ADAPTER_V1 = "models/r1-distill-7b-eee4114f-behaviour"


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
    # How strongly the fine-tuned adapter is applied, where one is used. 1.0 is the
    # adapter as trained; lower values interpolate towards the base model. It lives on
    # the condition so the benchmark, the stress suites and the browser demo cannot
    # disagree about which system they are running.
    adapter_scale: float = 1.0
    # A pipeline that turns the adapter on and off per question (see ym07.hybrid).
    hybrid: bool = False
    # The chunked corpus to retrieve from, where it is not the run's default. Chunk
    # size and retrieval depth decide each other - twelve 200-token passages are the
    # same prompt length as four 800-token ones - so a condition that sets `top_k`
    # without saying which corpus it means is only half a condition. Naming it here
    # also keeps the pairing out of the command line, where getting it wrong would
    # file a run under a name that misdescribes it and leave nothing to catch it.
    chunks: str | None = None
    # Whether the assessed-work gate replaces a reply that answers work the student
    # said is for marks. On in the delivered system, because a policy that holds most
    # of the time is not a policy. Off when the question is what the model does by
    # itself: with the gate on, the pushback suite reads 18 of 18 whatever the weights
    # have learned, so it cannot see the training it is supposed to be measuring.
    gate: bool = True
    # Whether a numerical question is computed by a program before it is explained.
    # Off since 9 October. Measured on all 264 calculation questions: 87.5% without it
    # against 86.4% with, discordant 10-7 at p = 0.63, and it costs 13% of the latency
    # and 6% of the output tokens. Three measurements, and the largest is null - 1-2 at
    # n=33, 8-3 at n=99, 10-7 at n=264 - so the n=99 reading of +5 points was the outlier.
    # A-r1-final-nocalc is now the arm that puts it back.
    calculate: bool = False
    # Whether the router switches the adapter off for numerical questions. That rule
    # exists because the first adapter cost arithmetic accuracy; the second does not -
    # 78.8% either way - so the rule may now be removable, and a component that cannot
    # show it earns its place should not ship.
    route_adapter: bool = True
    # Whether the system prompt tells the model to ground its claims in the supplied
    # passages and cite them. Normally this follows retrieval, which is why "retrieval
    # off" has always changed two things at once. Set it independently and the two can
    # be told apart: passages with the instruction, passages without it, neither.
    grounded: bool | None = None
    # Where an answer about how the course runs comes from.
    #
    #   "handout"       the facts parsed out of the handout, as a labelled list. Ships.
    #   "handout-pages" the same list, with the handout's pages supplied under it too.
    #   "pages"         the handout's four pages alone, nothing parsed.
    #   "facts"         the values transcribed into the config by hand.
    #   "search"        a ranked search of the whole corpus, which is what came first.
    #
    # Five arms over the same 23 questions, so where the answers come from is reported
    # as a measurement rather than asserted.
    course_source: str = "handout-pages"
    # Whether a question plainly asking for one of the handout's facts is answered by
    # looking the fact up, cited to its page, with no model asked (E27, 14 October; see
    # ym07.coursefacts and protocols/course-facts.md). The same lookup also widens what
    # counts as a question about the course, by similarity, so that one the word list
    # misses still gets the handout. Off everywhere except A-r1-facts, so every system
    # measured before it is unchanged.
    #   0  off
    #   1  version 1 (E27, A-r1-facts)
    #   2  version 2 (E28, A-r1-facts2): also replies itself when the handout does not give
    #      what is asked, and no longer widens routing; see ym07.coursefacts
    handout_lookup: int = 0
    # Whether the lookup also answers a question that uses the handout's weights with marks
    # of the student's own, by arithmetic, the pass mark stated as an assumption (E29,
    # protocols/course-marks.md). Only A-r1-facts-marks.
    handout_marks: bool = False
    # Whether a citation whose passage shares no wording with the answer is removed
    # after the fact. The restriction on answering other courses' questions is
    # withdrawn, and six of ten such answers came back tagged [S1] to a passage of
    # signal processing notes - a citation is the only evidence a student has that an
    # answer came from the course, and there it said the opposite of the truth. On, with
    # A-r1-final-nocheck as the arm that prices it. See ym07.citations.
    check_citations: bool = True
    # Whether a decline is overturned by re-asking the base weights. Both paths exist
    # only because the scope clause misfired - they were rescuing 13 of 77 in-scope
    # questions sent to other courses. The clause is now out of the config and out of all
    # 715 training prompts, and the delivered model answers 10 of 10 out-of-course
    # questions, so there may be nothing left for them to catch.
    # Off since 9 October. Not one of 160 scope outcomes changes with it removed, and
    # concept is 0-0. It was built to rescue the 13 of 77 in-scope questions the scope
    # clause wrongly sent to other courses, and that clause is gone from the config and
    # from all 715 training prompts. The code is left in place behind this switch rather
    # than deleted, so the measurement stays reproducible.
    overturn: bool = False

    def resolved_grounded(self) -> bool:
        return self.use_retrieval if self.grounded is None else self.grounded

    def course_facts(self, config: CourseConfig,
                     parsed: tuple[tuple[str, str], ...] = ()) -> tuple[tuple[str, str], ...]:
        """The labelled list of facts to put in front of the model.

        `parsed` is what handout.read_facts got out of the document. The transcribed
        copy in the config is only sent by the arm that is measuring it.
        """
        if self.course_source in ("handout", "handout-pages"):
            return parsed
        return config.facts if self.course_source == "facts" else ()

    def course_handout(self, config: CourseConfig) -> str:
        """The handout to supply as passages, empty for the arms not using it."""
        return config.handout if self.course_source in ("pages", "handout-pages") else ""

    @property
    def course_search(self) -> bool:
        """Whether a question about the course is answered by ranking the whole corpus."""
        return self.course_source == "search"

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

OSS_HYBRID = Condition(
    name="C2-oss-hybrid",
    description="One local model, adapter on or off per question, retrieval + policy prompt",
    purpose=(
        "What round one forces: the adapter improves explanation, citation and the "
        "course boundary, and damages arithmetic. This condition answers numerical "
        "work from the base weights and everything else from the fine-tuned ones, "
        "with the fine-tuned weights gating the numerical path so assessed work is "
        "still held back. It is the proposed system from 20 September onwards."
    ),
    use_course_prompt=True,
    use_retrieval=True,
    model_id=OSS_TUNED_MODEL,
    provider="local",
    top_k=4,
    effort="n/a",
    hybrid=True,
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

# --- the size ladder ------------------------------------------------------------
# One variable: how big the model is. Same corpus, same four passages, same policy
# prompt, same program-aided calculation, no adapter anywhere - on the calculation
# suite every question routes to the base weights, so the adapter never fires and
# these differ from the delivered system only in parameter count.
#
# The point is the slope, not any one score. The project is bounded by a 16 GB card
# and 15 GB of host memory; what a reader needs to know is whether the architecture
# is near its ceiling or merely near this machine's ceiling, and that is a question
# about the curve rather than the endpoint.
SIZE_LADDER = {
    "C2-oss-1.5b": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "C2-oss-3b": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "C2-oss-7b": OSS_MODEL,
    "C2-oss-14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
}

SIZES = tuple(
    Condition(
        name=name,
        description=f"Size ladder: {model.rsplit('/', 1)[-1]}, retrieval + policy prompt",
        purpose=(
            "One rung of the model-size ladder. Everything but the parameter count is "
            "held at the delivered system's settings, so the difference between rungs "
            "is the size and nothing else."
        ),
        use_course_prompt=True,
        use_retrieval=True,
        model_id=model,
        provider="local",
        top_k=4,
        effort="n/a",
        hybrid=True,
    )
    for name, model in SIZE_LADDER.items()
)


# The delivered system, reading more of the corpus. Retrieval puts the right passage
# in the top four for 77% of the questions written from a known chunk and in the top
# eight for 84%, so a quarter of the questions at the delivered depth are answered
# without the passage that would have grounded them. The config already asks for
# eight; the conditions above cut it to four because the adapter could only be
# trained on four-passage examples inside 16 GB, which is a limit on training and not
# on answering. This is that limit lifted, and nothing else.
OSS_HYBRID_K8 = replace(OSS_HYBRID, name="C2-oss-hybrid-k8", top_k=8,
                        description="The delivered pipeline, eight passages instead of four")

# Eight passages mostly does not run on this card: 37 of 41 concept questions failed,
# out of memory. The prompt roughly doubles, and the logits computed across it during
# prefill - 152k vocabulary by every position - ask for allocations above five
# gigabytes, which on a 16 GB card already holding a 7B model and its cache is the end
# of it.
#
# An earlier version of this comment said six passages is the most that fits. The
# evidence beside it does not support that: the six-passage probe failed 6 of the 12
# questions it reached, and its failing allocation was 892 MB with 1.26 GB free, so it
# died of fragmentation rather than absence. Both probes also ran on the delivered
# model at a 14 GB cap, where runs now use 15.5 GB, and neither touched the
# calculation suite. Where the ceiling actually sits is not yet measured - it needs
# the chosen model, the current cap, and the fallback switched off so a failure is
# recorded as one instead of being absorbed - and it should be stated in tokens of
# retrieved material rather than a passage count, since that is what the card
# responds to and it is the form another chunk size can be compared against.
OSS_HYBRID_K6 = replace(OSS_HYBRID, name="C2-oss-hybrid-k6", top_k=6,
                        description="The delivered pipeline, six passages instead of four")


# --- the candidate replacements ------------------------------------------------
# Same size class as the delivered model, same card, same pipeline: what changes is
# the weights. The size ladder showed accuracy still climbing at 7B and the next rung
# unreachable on 16 GB, so the remaining lever is not more parameters but better ones -
# a newer generation, or weights trained to reason before answering.
#
# All three emit their working inside <think> tags, which is stripped from the answer
# before marking; see ym07.model.split_reasoning for why that is not optional.
ARENA_MODELS = {
    "A-qwen3-8b": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "A-r1-distill-7b": "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
    "A-qwen3-4b-thinking": "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
}

ARENA = tuple(
    Condition(
        name=name,
        description=f"Candidate: {model.rsplit('/', 1)[-1]}, retrieval + policy prompt",
        purpose=(
            "One candidate for the delivered model. Everything but the weights is held "
            "at the delivered system's settings, so a difference between candidates is "
            "the model and nothing else."
        ),
        use_course_prompt=True,
        use_retrieval=True,
        model_id=model,
        provider="local",
        top_k=4,
        effort="n/a",
        hybrid=True,
    )
    for name, model in ARENA_MODELS.items()
)


# The same candidate on half the context. Nothing about the model changes; only how
# much of the corpus is put in front of it.
#
# This exists to settle whether the 17 questions it cannot answer are a limit of the
# model or of the card. The failures are all out of memory, which looks like a hardware
# ceiling, but an error message is not evidence - if the identical model answers the
# identical questions once the prompt is halved, then memory was the binding constraint
# and the capability was there all along. If it still fails, the explanation is
# something else and the "more resources would fix this" claim does not hold.
R1_K2 = replace(ARENA[1], name="A-r1-distill-7b-k2", top_k=2,
                description="R1-Distill-7B on two passages instead of four")


# --- what the retrieval grid proposes, tested on answers rather than recall ---------
#
# The grid (scripts/chunk_grid.py, 673 questions) says the delivered setting is the
# worst of the four chunk sizes it compared. At a matched prompt budget, 200-token
# chunks twelve deep put the right passage in front of the model 91.4% of the time
# against 76.7% for the 800-token chunks four deep that this project ships, and they
# do it in slightly fewer tokens. The ordering holds at every overlap threshold tried.
#
# That is a measurement of retrieval, not of answers, and the two can disagree here
# for a specific reason: twelve 200-token passages are twelve fragments, where four
# 800-token passages are four whole ones. A derivation that sits intact inside an
# 800-token chunk is cut across three 200-token chunks, so the right material can
# arrive in pieces too short to work from. Recall cannot see that and these conditions
# can, which is why the grid does not get to decide this on its own.
#
# The control is ARENA[1] itself - 800 tokens, four passages - already run on the
# calculation suite, so what these add is the comparison and not a re-measurement.
R1_C200 = replace(ARENA[1], name="A-r1-c200-k12", top_k=12,
                  chunks="corpus/processed/chunks-200.jsonl",
                  description="R1-Distill-7B, 200-token chunks twelve deep",
                  purpose=("What the retrieval grid recommends. Same prompt length as "
                           "the delivered setting, 15 points more retrieval recall, and "
                           "twelve fragments instead of four whole passages."))
# --- the chosen model, taught the course's behaviour ------------------------------
#
# The adapter the delivered system carried was fitted to Qwen2.5 and cannot be moved to
# other weights, so the chosen model arrived with no course training at all. What that
# costs is measured: it declined 0 of 27 out-of-scope questions where the delivered model
# declined 80 of 80, named assessed work 0 times in 18 where the delivered model named it
# 50 times in 50, and cited a retrieved passage in 1 answer of 132 against 83%.
#
# This is that gap closed, on 715 examples and three behaviours: refusing assessed work,
# answering from the course handout block supplied in the prompt, and citing the passages
# an answer used. The 457 numerical examples of the original set are left out, because the
# router switches the adapter off for numerical questions - so training on them cannot
# reach inference, and they are the short direct arithmetic answers that are the likeliest
# explanation for the arithmetic the first adapter damaged. For the first time the
# training distribution is the distribution the adapter is applied to.
#
# The course facts are taught with their values scrambled, so what is learned is reading
# the block rather than the 2026 dates. A model that memorised them would answer next
# year's students with last year's timetable, and a date change would mean retraining.
R1_TUNED = replace(ARENA[1], name="A-r1-tuned",
                   model_id=ADAPTER,
                   description="R1-Distill-7B with the course behaviour adapter",
                   purpose=("The chosen model taught the three behaviours it arrived "
                            "without: withholding assessed work, answering about the "
                            "course from the supplied handout, and citing what it used. "
                            "Compared against A-r1-distill-7b, which is the same weights "
                            "with no adapter."))

# The delivered local system with the handout lookup in front of the model (E27). Asked
# for by Sanjan on 14 October: on the 23 held-out course questions A-r1-tuned misread six
# facts that were in its prompt, and in the browser it answered "How are we assessed in
# this course?" with "The course does not cover this". Everything else is A-r1-tuned's,
# so on any question the lookup does not answer the two are the same system.
R1_FACTS = replace(R1_TUNED, name="A-r1-facts", handout_lookup=1,
                   description="the delivered local system with the handout lookup",
                   purpose=("Course facts answered from the handout by lookup, cited to "
                            "the page, rather than read by the 7B model. Compared against "
                            "A-r1-tuned on the fresh course set of 14 October."))

# Version 1 with the marks rule (E29, protocols/course-marks.md): asked what exam mark a
# student needs given their class test and project marks, A-r1-facts replied with the
# weights alone, and the model, asked instead, said 82% where 48.6% is right at a 50%
# pass mark. Everything else is A-r1-facts.
R1_FACTS_MARKS = replace(R1_FACTS, name="A-r1-facts-marks", handout_marks=True,
                         description="the delivered local system with version 1 of the handout "
                                     "lookup and the marks rule",
                         purpose=("Questions that use the course's weights answered by arithmetic "
                                  "from the handout, the pass mark stated as an assumption. "
                                  "Compared against A-r1-facts on the marks set of 14 October."))

# Version 2 of the lookup (E28, protocols/course-facts-2.md). E27 found the local model
# inventing what the handout does not give (an exam start time, a project deadline), a
# similarity guard holding back five right matches, and the wider routing catching
# assessed-work messages. On every question its handout path leaves alone, this is
# A-r1-tuned.
R1_FACTS2 = replace(R1_TUNED, name="A-r1-facts2", handout_lookup=2,
                    description="the delivered local system with version 2 of the handout lookup",
                    purpose=("Course facts from the handout by lookup, and a fixed reply when "
                             "the handout does not give what is asked. Compared against "
                             "A-r1-facts and A-r1-tuned on the second fresh course set."))

# --- what retrieval contributes, what training contributes, and whether they interact -
#
# Four cells over the same model and the same questions: retrieval on or off, the course
# adapter on or off. Measured one at a time, neither component's contribution can be
# separated from the other's, and nothing in this project has ever measured whether the
# retrieval helps at all - which for a retrieval-augmented tutor is the claim the whole
# thing rests on. It is a live question rather than a formality: three retrieval settings
# spanning fifteen points of recall produced no difference in answers on any suite.
#
# Turning retrieval off changes two things, not one. The course system prompt carries
# instructions to ground every claim in the supplied passages and cite them, and keeping
# those while supplying nothing would instruct the model to cite what it has not got. So
# the off cells use the ungrounded prompt, as C0/C1 against C2 always have. The contrast
# is therefore "retrieval and its instructions" against neither, which is the comparison
# a reader cares about but is not a single variable, and saying so here is cheaper than
# having it inferred from the numbers.
#
# The gate is off in all four, because it and the training aim at the same behaviour: it
# forces the pushback suite to 18 of 18 whatever the weights have learned, so with it on
# the measurement is blind to the thing being measured.
R1_NORAG = replace(ARENA[1], name="A-r1-norag", use_retrieval=False,
                   use_course_prompt=True, gate=False,
                   description="R1-Distill-7B, no retrieval, no course adapter",
                   purpose="The model alone: neither component, as the floor to measure from.")
R1_TUNED_NORAG = replace(ARENA[1], name="A-r1-tuned-norag", use_retrieval=False,
                         use_course_prompt=True, gate=False,
                         model_id=ADAPTER,
                         description="R1-Distill-7B with the course adapter, no retrieval",
                         purpose="Training without retrieval: what the adapter knows unaided.")
R1_RAG_NOGATE = replace(ARENA[1], name="A-r1-rag-nogate", gate=False,
                        description="R1-Distill-7B, retrieval, no adapter, gate off",
                        purpose="Retrieval without training, with the gate out of the way.")
R1_TUNED_NOGATE = replace(ARENA[1], name="A-r1-tuned-nogate", gate=False,
                          model_id=ADAPTER,
                          description="R1-Distill-7B with retrieval and the course adapter",
                          purpose=("Both components. Against the other three it says what "
                                   "each contributes and whether they add up."))

# --- does each remaining component earn its place -----------------------------------
#
# The delivered system against itself with one piece removed. A component that cannot
# show a difference is weight the design does not need, and the honest answer may be
# that the system is the model with retrieval and an adapter and little else.
#
# All three keep the gate off, since it only touches assessed work and is measured
# separately, and all three are the tuned model with retrieval - the configuration the
# four-cell run picked out.
R1_NOCALC = replace(ARENA[1], name="A-r1-final-withcalc", gate=False, calculate=True,
                    model_id=ADAPTER,
                    description="the delivered system with the calculator put back",
                    purpose=("The calculator is off in the delivered system since "
                             "9 October, so this arm is the one that puts it back. The "
                             "result files under the old name A-r1-final-nocalc, in "
                             "stress/results/v2-final-calc99, are the same comparison "
                             "taken while the default was the other way round."))
R1_NOROUTE = replace(ARENA[1], name="A-r1-final-noroute", gate=False, route_adapter=False,
                     model_id=ADAPTER,
                     description="the delivered system with the adapter left on throughout",
                     purpose=("The router answers numerical questions from the base "
                              "weights because the first adapter cost arithmetic. The "
                              "second does not - 78.8% either way - so this asks whether "
                              "the rule is still buying anything."))
R1_NOCITE = replace(ARENA[1], name="A-r1-final-nocite", gate=False, grounded=False,
                    model_id=ADAPTER,
                    description="the delivered system, passages supplied, citing not asked for",
                    purpose=("Retrieval has always been measured against having neither "
                             "the passages nor the instruction to cite them. This supplies "
                             "the passages and drops the instruction, so the two can be "
                             "told apart."))

# --- where an answer about how the course runs comes from ----------------------------
#
# Three arms over the same 23 questions. The delivered system supplies the handout
# whole; these two are what it was doing before, so the choice is reported as a
# measurement. Both keep the gate off, like the ablations, and both are the tuned model
# with retrieval.
#
# All three now carry the routing fix. The arms measured before it was found sent 15 of
# the 23 questions to the base weights with the calculator attached, because the course
# code read as numerical work, so they cannot be compared with these directly.
# The adapter as it was before 8 October, kept addressable so the training-set changes
# can be measured rather than asserted. It was trained with the scope restriction in
# every system prompt and seventeen examples teaching it, with six absent-fact examples
# that shared one completion reciting the handout's contents, and with three of the 23
# course answers stating themselves in the block because two facts were never scrambled.
# Every result directory written before 8 October is this adapter.
# Experiment 1: the minimal system that still has the deterministic safeguards. Base
# weights, retrieval, the gate on, no adapter. BM25 and the shipped fusion are
# interchangeable at this chunk size - no difference at all over 673 questions - so the
# retrieval here is the delivered one and the only thing missing is the adapter. The
# question it answers: does the adapter contribute anything to assessed-work compliance
# that a deterministic gate over base weights does not already give?
R1_BASE_GATED = replace(ARENA[1], name="A-r1-base-gated", gate=True,
                        model_id=ARENA_MODELS["A-r1-distill-7b"],
                        description="base R1 with retrieval and the gate, no adapter",
                        purpose=("The minimal arm. If a gate over untrained weights also "
                                 "leaks 0 of 54, the adapter's contribution to assessed "
                                 "work is zero and the case for it rests on citation "
                                 "behaviour alone."))

# Experiment 4: the overturn machinery removed.
R1_NOOVERTURN = replace(ARENA[1], name="A-r1-final-withoverturn", gate=False,
                        overturn=True, model_id=ADAPTER,
                        description="the delivered system with the refusal overturns put back",
                        purpose=("The overturns are off in the delivered system since "
                                 "9 October - they changed none of 160 scope outcomes - "
                                 "so this arm puts them back. The files under the old "
                                 "name A-r1-final-nooverturn are the same comparison from "
                                 "the other side."))

R1_NOCHECK = replace(ARENA[1], name="A-r1-final-nocheck", gate=False,
                     check_citations=False,
                     model_id=ADAPTER,
                     description="the delivered system without the citation check",
                     purpose=("The check removes a citation whose passage shares no "
                              "wording with the answer. It caught all six false "
                              "citations on out-of-course answers and, hand-marked, "
                              "about one removal in five takes a citation that should "
                              "have stood. This is the arm that prices that trade."))

R1_TUNED_V2B = replace(ARENA[1], name="A-r1-tuned-nogate-v2b", gate=False,
                       model_id=ADAPTER_V2B,
                       description="v2's training set again, with a different shuffle seed",
                       purpose=("The control. Same examples, same settings, one "
                                "meaningless thing changed, so what separates it from v2 "
                                "is run-to-run variance and nothing else. Until this is "
                                "measured, no comparison between adapters means "
                                "anything, including the ones already in the metrics."))
R1_TUNED_V2 = replace(ARENA[1], name="A-r1-tuned-nogate-v2", gate=False,
                      model_id=ADAPTER_V2,
                      description="the adapter with 86 negative course examples",
                      purpose=("Between v1 and v3. It fixed the invented exam date and "
                               "the withdrawn scope refusals, and over-corrected: it "
                               "refused five course facts the handout carries."))
R1_TUNED_V1 = replace(ARENA[1], name="A-r1-tuned-nogate-v1", gate=False,
                      model_id=ADAPTER_V1,
                      description="the superseded adapter, trained with the scope restriction",
                      purpose=("What the rebuilt training set changed. Same questions, "
                               "same everything else, the adapter before it."))

R1_COURSE_PAGES = replace(ARENA[1], name="A-r1-course-pages", gate=False,
                          course_source="pages",
                          model_id=ADAPTER,
                          description="course questions answered from the handout's pages",
                          purpose=("The document itself, cited by page, with nothing "
                                   "parsed out of it. The adapter has no training for "
                                   "that format - all 190 of its course examples are a "
                                   "labelled list - and this is what it does without it."))
R1_COURSE_BOTH = replace(ARENA[1], name="A-r1-course-both", gate=False,
                         course_source="handout-pages",
                         model_id=ADAPTER,
                         description="the parsed facts with the handout's pages under them",
                         purpose=("Whether having the pages as well as the list is worth "
                                  "the tokens: it makes the citation instruction "
                                  "answerable with a page, and it is 1,624 tokens."))
R1_COURSE_FACTS = replace(ARENA[1], name="A-r1-course-facts", gate=False,
                          course_source="facts",
                          model_id=ADAPTER,
                          description="course questions answered from the transcribed facts",
                          purpose=("What the system did before: the handout's values typed "
                                   "into the config as a flat list, nothing retrieved. It "
                                   "leaves the citation instruction in the prompt with no "
                                   "passages under it, which is its own problem."))
R1_COURSE_SEARCH = replace(ARENA[1], name="A-r1-course-search", gate=False,
                           course_source="search",
                           model_id=ADAPTER,
                           description="course questions answered by ranking the whole corpus",
                           purpose=("The original design. It has to find the handout among "
                                    "316 chunks, and seventeen years of class test papers "
                                    "each carry their own date, so this is the arm that "
                                    "reads a date off a past paper."))

R1_C400 = replace(ARENA[1], name="A-r1-c400-k8", top_k=8,
                  chunks="corpus/processed/chunks-400.jsonl",
                  description="R1-Distill-7B, 400-token chunks eight deep",
                  purpose=("The middle case, and the one worth having if fragmentation "
                           "is what matters: 10 points more recall than the delivered "
                           "setting while keeping passages long enough to hold a short "
                           "derivation whole."))


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in (
        BASELINE, PROMPTED, RAG, MONEY, EFFORT, TOPK,
        OSS_BASELINE, OSS_PROMPTED, OSS_RAG, OSS_TUNED, OSS_HYBRID,
        *FRONTIER_BASELINES,
        *FRONTIER_PROMPTED,
        *SIZES,
        OSS_HYBRID_K8,
        OSS_HYBRID_K6,
        *ARENA,
        R1_K2,
        R1_C200,
        R1_C400,
        R1_TUNED, R1_FACTS, R1_FACTS2, R1_FACTS_MARKS,
        R1_NORAG, R1_TUNED_NORAG, R1_RAG_NOGATE, R1_TUNED_NOGATE,
        R1_NOCALC, R1_NOROUTE, R1_NOCITE,
        R1_COURSE_FACTS, R1_COURSE_SEARCH, R1_COURSE_PAGES, R1_COURSE_BOTH,
        R1_TUNED_V1, R1_TUNED_V2, R1_TUNED_V2B, R1_NOCHECK,
        R1_BASE_GATED, R1_NOOVERTURN,
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


# --- the premium-model arms, approved 11 October ------------------------------------
#
# The brief's A1 (premium model, no retrieval) is C1-prompted, run in September. These
# two put the same model through retrieval: A2 with the retrieval the first system had -
# fused, four passages, the course prompt, a ranked search for course questions, nothing
# checked afterwards - and A3 with the delivered system's pipeline around it: the handout
# parsed and supplied for course questions, the assessed-work gate, the citation check.
# Neither has a router or an adapter, because a hosted model has nothing to switch; the
# two differ from A-r1-tuned in the model and in that alone, which is the comparison the
# premium question needs. They run only through ym07.hosted, behind --allow-paid.
A2_OPUS = Condition(
    name="A2-opus-rag",
    description="Claude Opus 5 with the first system's retrieval: fused, four passages, course prompt",
    purpose=("The brief's A2: the premium model with the original retrieval and nothing "
             "else. Against C1-prompted it says what the corpus adds to the premium "
             "model; against A3-opus-pipeline, what the delivered pipeline adds."),
    use_course_prompt=True, use_retrieval=True,
    model_id="claude-opus-5", provider="anthropic", top_k=4, effort="high",
    hybrid=False, gate=False, calculate=False, route_adapter=False,
    course_source="search", check_citations=False, overturn=False,
)
A3_OPUS = Condition(
    name="A3-opus-pipeline",
    description="Claude Opus 5 inside the delivered pipeline: handout facts, gate, citation check",
    purpose=("The brief's A3: the premium model with the delivered system's retrieval "
             "side. Against A-r1-tuned the only difference is the model; against "
             "A2-opus-rag, only the pipeline."),
    use_course_prompt=True, use_retrieval=True,
    model_id="claude-opus-5", provider="anthropic", top_k=4, effort="high",
    hybrid=False, gate=True, calculate=False, route_adapter=False,
    course_source="handout-pages", check_citations=True, overturn=False,
)
CONDITIONS.update({c.name: c for c in (A2_OPUS, A3_OPUS)})

# The conditions measured through ym07.hosted. The browser serves them the same way,
# with the deterministic gate around the hosted model, so that a demonstration of A3
# is the system the results describe and not the bare model under its name.
HOSTED_PIPELINE = frozenset({A2_OPUS.name, A3_OPUS.name})
