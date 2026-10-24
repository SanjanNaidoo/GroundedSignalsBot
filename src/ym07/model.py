"""The call to Claude, and the record of what it cost.

Deliberate choice: no server-side refusal fallback. The fallback feature silently
retries a refused request on a different model, which is exactly right for a
production app and exactly wrong for a benchmark — a recorded answer must be
attributable to one known model. A refusal is recorded as a refusal and shows up
in the report.
"""

from __future__ import annotations

import json
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
        self._loaded: dict[str, tuple] = {}
        # Without this, concurrent callers all miss the cache at once and each
        # starts loading its own copy of a multi-gigabyte model, exhausting
        # memory before any of them finishes.
        self._load_lock = threading.Lock()

    def _load(self, model_id: str) -> tuple:
        """Load once and cache. Weights are big and conditions share them."""
        if model_id in self._loaded:
            return self._loaded[model_id]

        with self._load_lock:
            # Re-check inside the lock: several threads can pass the check above
            # before the first acquires it.
            if model_id in self._loaded:
                return self._loaded[model_id]
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

            model = PeftModel.from_pretrained(model, model_id)
        model.eval()
        self._loaded[model_id] = (tokenizer, model)
        return tokenizer, model

    def answer(self, *, system, user, model, max_tokens, effort) -> Answer:
        # `max_tokens` and `effort` are hosted-arm concepts. The generation cap
        # comes from local_max_new_tokens instead, and effort has no analogue —
        # the conditions record it as "n/a" rather than implying a knob.
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
        try:
            with self._torch.inference_mode():
                generated = lm.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=sampling,
                    temperature=self.temperature if sampling else None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        except Exception as exc:  # pragma: no cover - depends on the machine
            return _failed(model, effort, started, f"generation failed: {exc}")

        new_tokens = generated[0][prompt_tokens:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        hit_cap = int(new_tokens.shape[0]) >= self.max_new_tokens

        return Answer(
            text=completion,
            model=model,
            effort=effort,
            stop_reason="max_tokens" if hit_cap else "end_turn",
            refusal_category=None,
            input_tokens=prompt_tokens,
            output_tokens=int(new_tokens.shape[0]),
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


def build_client(dry_run: bool, provider: str = "anthropic", config=None, frontier=None):
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
        )
    if provider == "anthropic":
        return ClaudeClient()
    if provider == "openai":
        if frontier is None:
            raise ValueError("the openai provider needs a `frontier:` config entry")
        return OpenAICompatibleClient(frontier)
    raise ValueError(f"Unknown provider '{provider}'. Known: anthropic, local, openai")
