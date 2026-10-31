"""The call to Claude, and the record of what it cost.

Deliberate choice: no server-side refusal fallback. The fallback feature silently
retries a refused request on a different model, which is exactly right for a
production app and exactly wrong for a benchmark — a recorded answer must be
attributable to one known model. A refusal is recorded as a refusal and shows up
in the report.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# USD per million tokens. Update from the pricing page if these drift; cost is a
# reported result (C3-money), so the numbers behind it belong in version control.
#
# Locally-hosted models are deliberately absent, so they price at zero. That is
# the honest number for *marginal API cost* and it is the number C3-money is
# about — but it is not the cost of running them. GPU time, electricity and the
# hardware itself are real, and the report must account for them separately
# rather than presenting the open arm as free.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Models where thinking is configured adaptively and `temperature` is rejected.
_ADAPTIVE_THINKING = {"claude-opus-5", "claude-sonnet-5"}


# A reasoning model's working, and the answer that follows it. Matched generously: the
# opening tag is sometimes absent because the chat template has already emitted it, so
# a reply can begin mid-thought and close with </think>.
_THINK = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_UNCLOSED_OPEN = re.compile(r"<think>(.*)\Z", re.S | re.I)
_LEADING_CLOSE = re.compile(r"\A(.*?)</think>", re.S | re.I)


def split_reasoning(completion: str) -> tuple[str, str]:
    """The answer a student would see, and the thinking that preceded it."""
    text = completion or ""
    thoughts: list[str] = []

    def take(match):
        thoughts.append(match.group(1))
        return ""

    # A stray closing tag with no opener: everything before it was thinking.
    if "<think>" not in text.lower() and "</think>" in text.lower():
        found = _LEADING_CLOSE.search(text)
        if found:
            thoughts.append(found.group(1))
            text = text[found.end():]
    text = _THINK.sub(take, text)
    # Thinking that ran to the token cap and never closed: there is no answer in it.
    unclosed = _UNCLOSED_OPEN.search(text)
    if unclosed:
        thoughts.append(unclosed.group(1))
        text = text[:unclosed.start()]
    return text.strip(), "\n".join(t.strip() for t in thoughts).strip()


@dataclass
class Answer:
    """One system-under-test response, with everything the report needs."""

    text: str
    model: str
    effort: str
    stop_reason: str | None
    refusal_category: str | None
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    latency_s: float
    error: str | None = None
    # What a reasoning model thought before answering, kept out of `text`.
    #
    # Qwen3 and the R1 distillations emit their working inside <think>...</think> and
    # then answer. That working is full of discarded values - a wrong sum it corrects
    # two lines later, the right answer reached and then abandoned - and every rule in
    # the calculation suite searches the answer for numbers. Scored against the whole
    # stream, a model would be marked on what it considered rather than on what it
    # said, and in both directions: credited for a right value it rejected, failed for
    # a wrong one it caught. So the reply is split, `text` carries what the student
    # would see, and this carries the rest for the record.
    reasoning: str = ""

    @property
    def cost_usd(self) -> float:
        rate_in, rate_out = PRICING.get(self.model, (0.0, 0.0))
        return (
            self.input_tokens * rate_in + self.output_tokens * rate_out
        ) / 1_000_000

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_usd"] = round(self.cost_usd, 6)
        return d


class DryRunClient:
    """Stands in for the API so the harness runs with no key and no spend.

    Produces obviously-fake output. It exercises the full path — prompting,
    retrieval, run directories, grading, reporting — without pretending to be a
    result. Never grade a dry run.
    """

    def answer(self, *, system, user, model, max_tokens, effort) -> Answer:
        time.sleep(0.01)
        preview = user.strip().splitlines()[-1][:80] if user.strip() else ""
        return Answer(
            text=(
                "[DRY RUN — no model was called. This is placeholder text so the "
                f"pipeline can be exercised end to end.] Question began: {preview!r}"
            ),
            model=model,
            effort=effort,
            stop_reason="end_turn",
            refusal_category=None,
            input_tokens=len(system or "") // 4 + len(user) // 4,
            output_tokens=64,
            cached_input_tokens=0,
            latency_s=0.01,
        )


class ClaudeClient:
    """Thin wrapper over the Anthropic SDK, with benchmark-appropriate settings."""

    def __init__(self, max_retries: int = 3):
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=max_retries)

    def answer(self, *, system, user, model, max_tokens, effort) -> Answer:
        anthropic = self._anthropic
        started = time.monotonic()

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": effort},
        }
        if system:
            kwargs["system"] = system
        if model in _ADAPTIVE_THINKING:
            # Adaptive thinking: the model decides how much to think. There is no
            # temperature knob on these models, so runs are not bit-identical —
            # measure that variance with repeat runs rather than assuming it away.
            kwargs["thinking"] = {"type": "adaptive"}

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.NotFoundError as exc:
            return self._failed(model, effort, started, f"unknown model or endpoint: {exc}")
        except anthropic.RateLimitError as exc:
            return self._failed(model, effort, started, f"rate limited: {exc}")
        except anthropic.APIStatusError as exc:
            return self._failed(model, effort, started, f"api error {exc.status_code}: {exc}")
        except anthropic.APIConnectionError as exc:
            return self._failed(model, effort, started, f"connection error: {exc}")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

        refusal_category = None
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            refusal_category = getattr(details, "category", None) or "unknown"

        usage = response.usage
        return Answer(
            text=text,
            model=model,
            effort=effort,
            stop_reason=response.stop_reason,
            refusal_category=refusal_category,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            latency_s=round(time.monotonic() - started, 3),
        )

    @staticmethod
    def _failed(model: str, effort: str, started: float, error: str) -> Answer:
        return Answer(
            text="",
            model=model,
            effort=effort,
            stop_reason=None,
            refusal_category=None,
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            latency_s=round(time.monotonic() - started, 3),
            error=error,
        )


class LocalClient:
    """Runs an open-weights model on local hardware, for the open arm.

    torch and transformers are an optional extra (`pip install -e '.[local]'`)
    and are imported lazily, so a machine that only drives the hosted arm never
    needs them installed.

    Greedy by default. Local inference exposes temperature and a seed, which the
    hosted models do not, so this arm is exactly reproducible — a re-run gives
    bit-identical output. That asymmetry is worth stating in the report: the
    variance runs the hosted arm needs are unnecessary here.
    """

    def __init__(
        self,
        *,
        max_new_tokens: int = 1024,
        load_in_4bit: bool = True,
        temperature: float = 0.0,
        seed: int = 0,
        adapter_scale: float = 1.0,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the machine
            # Distinguish "not installed" from "installed but mutually
            # incompatible". Blaming a missing extra when the real problem is a
            # version clash sends people to reinstall what they already have.
            if exc.name in (None, "torch", "transformers"):
                hint = (
                    "torch and transformers are not installed. Install the "
                    'optional extra:  pip install -e ".[local]"'
                )
            else:
                hint = (
                    "torch and transformers are both installed but disagree with "
                    "each other. This is a version mismatch, not a missing "
                    "package, and reinstalling the extra will not fix it.\n"
                    "The usual cause is a Python version that has no CUDA build "
                    "of torch — there are none for Python 3.14 — which silently "
                    "leaves a CPU-only wheel that transformers does not match. "
                    "Rebuild the environment on Python 3.12."
                )
            raise RuntimeError(f"The local provider could not start.\n{hint}\n"
                               f"(import failed: {exc})") from exc

        if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "torch is installed but reports no CUDA device, so this would run "
                "on the CPU: a 7B model would take hours per question and would "
                "not be the experiment the report describes.\n"
                "Either the driver is older than CUDA 12.8, or the installed torch "
                "is a CPU-only build. Check with:\n"
                '  python -c "import torch; print(torch.__version__, '
                'torch.version.cuda)"\n'
                "A CUDA build prints a version after the '+', e.g. 2.7.0+cu128; a "
                "CPU-only build prints None for torch.version.cuda."
            )

        self._torch = torch
        self._auto_model = AutoModelForCausalLM
        self._auto_tokenizer = AutoTokenizer
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.temperature = temperature
        self.seed = seed
        # How strongly the fine-tuned adapter is applied. 1.0 is the adapter as trained;
        # lower values interpolate towards the base model, which is a knob worth measuring
        # rather than a choice between all and nothing.
        self.adapter_scale = adapter_scale
        self._loaded: dict[tuple, tuple] = {}
        # Without this, concurrent callers all miss the cache at once and each
        # starts loading its own copy of a multi-gigabyte model, exhausting
        # memory before any of them finishes.
        self._load_lock = threading.Lock()

    def _load(self, model_id: str) -> tuple:
        """Load once and cache. Weights are big and conditions share them."""
        key = (model_id, self.adapter_scale)
        if key in self._loaded:
            return self._loaded[key]

        with self._load_lock:
            # Re-check inside the lock: several threads can pass the check above
            # before the first acquires it.
            if key in self._loaded:
                return self._loaded[key]
            return self._load_locked(model_id)

    def _load_locked(self, model_id: str) -> tuple:
        # The fine-tuned open arm is saved as a LoRA adapter rather than a whole
        # model: a directory of adapter weights that names the base it sits on.
        # The base is loaded exactly as C2-oss loads it and the adapter applied
        # on top, so the two conditions differ by the adapter and nothing else.
        adapter_config = Path(model_id) / "adapter_config.json"
        is_adapter = adapter_config.exists()
        base_id = (
            json.loads(adapter_config.read_text(encoding="utf-8"))["base_model_name_or_path"]
            if is_adapter else model_id
        )

        kwargs: dict = {"dtype": "auto", "device_map": "auto"}
        # A checkpoint that is already 4-bit carries its own quantization config;
        # supplying a second one conflicts with it.
        prequantized = "4bit" in base_id.lower() or "bnb" in base_id.lower()
        if self.load_in_4bit and not prequantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = self._auto_tokenizer.from_pretrained(model_id)
        model = self._auto_model.from_pretrained(base_id, **kwargs)
        if is_adapter:
            from peft import PeftModel
            from peft.tuners.lora import LoraLayer

            model = PeftModel.from_pretrained(model, model_id)
            if self.adapter_scale != 1.0:
                for module in model.modules():
                    if isinstance(module, LoraLayer):
                        for name in list(module.scaling):
                            module.scaling[name] *= self.adapter_scale
        model.eval()
        self._loaded[(model_id, self.adapter_scale)] = (tokenizer, model)
        return tokenizer, model

    def answer(self, *, system, user, model, max_tokens, effort, adapter: bool = True,
               max_new_tokens: int | None = None) -> Answer:
        # `max_tokens` and `effort` are hosted-arm concepts. The generation cap
        # comes from local_max_new_tokens instead, and effort has no analogue —
        # the conditions record it as "n/a" rather than implying a knob.
        #
        # `adapter=False` answers from the base weights with the fine-tuned adapter
        # switched off. It is the same loaded model either way, so a system can use
        # both without a second copy of a 7B on the card.
        started = time.monotonic()
        try:
            tokenizer, lm = self._load(model)
        except Exception as exc:  # pragma: no cover - depends on the machine
            return _failed(model, effort, started, f"could not load {model}: {exc}")

        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(lm.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        sampling = self.temperature > 0
        self._torch.manual_seed(self.seed)
        cap = max_new_tokens or self.max_new_tokens
        try:
            with self._torch.inference_mode(), self._without_adapter(lm, adapter):
                generated = lm.generate(
                    **inputs,
                    max_new_tokens=cap,
                    do_sample=sampling,
                    temperature=self.temperature if sampling else None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        except Exception as exc:  # pragma: no cover - depends on the machine
            return _failed(model, effort, started, f"generation failed: {exc}")

        new_tokens = generated[0][prompt_tokens:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        hit_cap = int(new_tokens.shape[0]) >= cap

        # The batch runner comes through here, not through `stream`, so the split has to
        # happen on both paths or the suites mark reasoning models on their working.
        answer, reasoning = split_reasoning(completion)
        return self._answer(answer, model, effort, started, prompt_tokens,
                            int(new_tokens.shape[0]), hit_cap, reasoning)

    @staticmethod
    def _without_adapter(lm, adapter: bool):
        """A context in which the adapter is off, when asked and when there is one."""
        import contextlib
        if adapter or not hasattr(lm, "disable_adapter"):
            return contextlib.nullcontext()
        return lm.disable_adapter()

    def stream(self, *, system, user, model, max_tokens=None, effort=None, adapter: bool = True):
        """Generate the same answer, yielding it as it is written.

        Yields text pieces, and finally an Answer with the whole reply and its counts.
        The tutor takes its time on a long derivation; a student watching a blank box
        cannot tell that from a crash. Nothing else differs: same prompt, same greedy
        decoding, same seed, so what streams here is what the batch runner records.
        """
        started = time.monotonic()
        tokenizer, lm = self._load(model)
        from transformers import TextIteratorStreamer

        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(lm.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        sampling = self.temperature > 0
        self._torch.manual_seed(self.seed)
        failure: list[str] = []

        def generate() -> None:
            try:
                with self._torch.inference_mode(), self._without_adapter(lm, adapter):
                    lm.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=sampling,
                        temperature=self.temperature if sampling else None,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        streamer=streamer,
                    )
            except Exception as exc:  # pragma: no cover - depends on the machine
                failure.append(str(exc))
                streamer.end()

        worker = threading.Thread(target=generate, daemon=True)
        worker.start()

        pieces = []
        for piece in streamer:
            pieces.append(piece)
            yield piece
        worker.join()

        if failure:
            yield _failed(model, effort, started, f"generation failed: {failure[0]}")
            return
        completion = "".join(pieces).strip()
        # Counted from the finished text rather than from the generation call, which a
        # streamer does not hand back. Same tokenizer, so the count matches within a token.
        # Counted on everything generated, thinking included, because that is what the
        # card and the clock actually paid for.
        output_tokens = len(tokenizer(completion).input_ids)
        answer, reasoning = split_reasoning(completion)
        yield self._answer(answer, model, effort, started, prompt_tokens, output_tokens,
                           output_tokens >= self.max_new_tokens, reasoning)

    @staticmethod
    def _answer(completion, model, effort, started, prompt_tokens, output_tokens, hit_cap,
                reasoning: str = "") -> Answer:
        return Answer(
            text=completion,
            reasoning=reasoning,
            model=model,
            effort=effort,
            stop_reason="max_tokens" if hit_cap else "end_turn",
            refusal_category=None,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=0,
            latency_s=round(time.monotonic() - started, 3),
        )


class OpenAICompatibleClient:
    """Any endpoint that speaks the OpenAI chat-completions protocol.

    Covers OpenAI itself and the providers that expose the same protocol —
    DeepSeek, xAI, Google's Gemini compatibility endpoint — which is why one
    class is enough for the whole cross-model baseline. Which endpoint, which
    model and which key come from the `frontier:` block of the course config.

    Sends the question at the provider's default settings: no temperature, no
    reasoning-effort knob, and no system prompt unless a condition supplies one.
    That is deliberate. The baseline is "what a student gets by pasting the
    question in", and every provider defaults differently; overriding those
    defaults would measure our settings, not their model.
    """

    def __init__(self, entry, max_retries: int = 3):
        import os

        import openai

        key = os.environ.get(entry.key_env)
        if not key:
            raise RuntimeError(
                f"{entry.key_env} is not set. The C0-{entry.id} condition sends the "
                f"question to '{entry.model}' and needs that key in the environment."
            )
        self._openai = openai
        self._entry = entry
        self._client = openai.OpenAI(
            api_key=key, base_url=entry.base_url, max_retries=max_retries
        )
        # The manifest's cost column reads from PRICING. Register this model's
        # rates so the number is reported rather than silently zero.
        PRICING[entry.model] = (entry.price_in, entry.price_out)

    def answer(self, *, system, user, model, max_tokens, effort) -> Answer:
        started = time.monotonic()
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        kwargs: dict = {"model": model, "messages": messages}
        kwargs[self._entry.max_tokens_param] = max_tokens

        try:
            response = self._client.chat.completions.create(**kwargs)
        except self._openai.APIStatusError as exc:
            return _failed(model, effort, started, f"api error {exc.status_code}: {exc}")
        except self._openai.APIConnectionError as exc:
            return _failed(model, effort, started, f"connection error: {exc}")

        choice = response.choices[0]
        message = choice.message
        refusal = getattr(message, "refusal", None)
        text = message.content or refusal or ""
        usage = response.usage
        return Answer(
            text=text,
            model=model,
            effort=effort,
            stop_reason=choice.finish_reason,
            refusal_category="refusal" if refusal else None,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_input_tokens=0,
            latency_s=round(time.monotonic() - started, 3),
        )


def _failed(model: str, effort: str, started: float, error: str) -> Answer:
    return Answer(
        text="",
        model=model,
        effort=effort,
        stop_reason=None,
        refusal_category=None,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        latency_s=round(time.monotonic() - started, 3),
        error=error,
    )


def build_client_for(condition, config, *, dry_run: bool = False, frontier=None):
    """The client a condition needs, including the pipeline it names.

    Every entry point goes through here, so the benchmark, the stress suites and the
    browser demo cannot end up running different systems under one condition name.
    """
    client = build_client(dry_run, condition.resolved_provider(config), config, frontier,
                          adapter_scale=condition.adapter_scale)
    if getattr(condition, "hybrid", False) and not dry_run:
        from .hybrid import HybridClient

        # All four switches, not only the gate. Passing one of them meant the browser
        # ran the calculator, the router and both refusal overturns whatever the
        # condition said, so a condition that ablates any of those described the batch
        # runner and not the served system. The suites have always passed all of them.
        client = HybridClient(client,
                              gate=getattr(condition, "gate", True),
                              calculate=getattr(condition, "calculate", True),
                              route_adapter=getattr(condition, "route_adapter", True),
                              overturn=getattr(condition, "overturn", True))
    return client


def build_client(dry_run: bool, provider: str = "anthropic", config=None, frontier=None,
                 adapter_scale: float = 1.0):
    """One client per provider (per endpoint, for the frontier baselines)."""
    if dry_run:
        return DryRunClient()
    if provider == "local":
        model_cfg = config.model if config else None
        return LocalClient(
            max_new_tokens=getattr(model_cfg, "local_max_new_tokens", 1024),
            load_in_4bit=getattr(model_cfg, "local_load_in_4bit", True),
            temperature=getattr(model_cfg, "local_temperature", 0.0),
            seed=getattr(model_cfg, "local_seed", 0),
            adapter_scale=adapter_scale,
        )
    if provider == "anthropic":
        return ClaudeClient()
    if provider == "openai":
        if frontier is None:
            raise ValueError("the openai provider needs a `frontier:` config entry")
        return OpenAICompatibleClient(frontier)
    raise ValueError(f"Unknown provider '{provider}'. Known: anthropic, local, openai")
