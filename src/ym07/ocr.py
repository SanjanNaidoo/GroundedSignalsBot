"""Reading a question out of an image a student has uploaded.

A student stuck on a tutorial question usually has it on screen rather than in
their hands, and retyping a transform question to ask about it is enough friction
to stop them asking. This module turns an uploaded image into text, which then
joins the ordinary question path: the same retrieval, the same prompt builder, the
same routing. Nothing downstream knows the question arrived as a picture.

Three decisions shape what is here.

**Two engines, best first.** Classical optical character recognition reads a
screenshot of prose well and mathematics badly, because it recognises glyphs one
at a time with no idea what an expression is: measured on this course's own
material it turned `3\\delta[n-2]` into `34[n - 2]`, dropped the subscript from
`h_1[n]`, and read `LTI` as `LT!`. Those are not cosmetic errors. A tutor handed
`34[n - 2]` is being asked a different question from the one on the student's
screen, and it will answer the different one. A vision-language model reads the
image as a whole and writes the mathematics back out in LaTeX, so it is tried
first; Tesseract stays as the fallback for a machine with no graphics card, and
`EngineChoice` records which one actually ran.

**The reading is shown to the student before it is used.** Neither engine is
reliable enough to put between a student and an answer unsupervised, so the web
tutor posts the image to `/read`, puts the transcription in the composer where it
can be corrected, and sends it only once the student presses send. The student is
the last check on a machine reading, and they are the one person who can see both
the image and the text.

**What was read stays labelled.** Even confirmed, the text is marked in the prompt
as having come from an image, so the model treats it as a transcription rather
than as something typed from scratch. A misread question answered confidently is
the failure this project spends a chapter guarding against.

Screenshots of typed material are what this is for. Handwriting is harder, and the
vision-language engine is the one with any chance at it; Section 6.2 of the report
sets out what doing it properly would take.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field

# A screenshot of a tutorial question is well under this. The cap exists because the
# decoded image is held in memory on a machine whose graphics card is already the
# binding constraint, and because a multi-megabyte photograph is not a question.
MAX_IMAGE_BYTES = 6 * 1024 * 1024

# Past this the transcription has stopped being a question and started being a
# document, and it would crowd the retrieved passages out of the prompt.
MAX_EXTRACTED_CHARS = 4000

# Formats a browser produces from a paste, a drag or a file picker.
ACCEPTED = {"PNG", "JPEG", "GIF", "BMP", "WEBP", "TIFF"}


@dataclass
class OcrResult:
    """What was read, by what, and how much of it to believe."""

    text: str
    engine: str
    width: int = 0
    height: int = 0
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def empty(self) -> bool:
        return not self.text.strip()


class OcrUnavailable(RuntimeError):
    """No engine on this machine. Carries what to install, not just that it failed."""


class OcrFailed(ValueError):
    """The image arrived but could not be read. The message is shown to the student."""


def _load(data: bytes):
    """Decode the upload, or say why it is not an image this will read."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - Pillow ships with the harness
        raise OcrUnavailable("Pillow is not installed, so images cannot be decoded.") from exc

    if len(data) > MAX_IMAGE_BYTES:
        raise OcrFailed(
            f"that image is {len(data) / 1e6:.1f} MB; the tutor takes up to "
            f"{MAX_IMAGE_BYTES / 1e6:.0f} MB. A screenshot of the question is enough."
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise OcrFailed("that file could not be read as an image.") from exc

    if image.format not in ACCEPTED:
        raise OcrFailed(f"{image.format or 'that format'} is not one this reads "
                        f"({', '.join(sorted(ACCEPTED))}).")
    return image


# ---------------------------------------------------------------------------
# Characters an engine returns that are wrong in a mathematical context rather than
# merely unusual. A minus sign set in a PDF is read as an em dash about as often as
# not, and "n - 2" and "n - 2" are not the same expression to a reader or to a
# model. Mapping these is a transcription fix, not a guess at meaning: nothing here
# changes one symbol into a different symbol the student might have written.
_LITERAL = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ",
}


def _normalise(text: str) -> str:
    for wrong, right in _LITERAL.items():
        text = text.replace(wrong, right)
    return text


def _tidy(text: str) -> str:
    """Trailing space off every line, blank lines out, dashes normalised."""
    text = _normalise(text)
    lines = (line.rstrip() for line in text.splitlines())
    return "\n".join(line for line in lines if line.strip()).strip()


# ---------------------------------------------------------------------------
# LaTeX out, for the retrieval query
# ---------------------------------------------------------------------------
#
# The vision reader writes mathematics as LaTeX, which is the right thing to show a
# student and the right thing to put in front of the model: it is unambiguous about
# subscripts and the interface renders it. It is the wrong thing to search with. The
# course corpus is text pulled out of PDFs, and it spells mathematics in Unicode -
# 855 deltas, 3993 omegas, 2053 pis, and not one LaTeX command anywhere in 335
# passages. Searching `\delta` against a corpus that says `δ` matches on nothing, so
# the question that retrieves the worst would be the one transcribed most precisely.
# The query is built from this instead.

_SYMBOLS = {
    r"\delta": "δ", r"\Delta": "Δ", r"\omega": "ω", r"\Omega": "Ω",
    r"\theta": "θ", r"\Theta": "Θ", r"\sigma": "σ", r"\Sigma": "Σ",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\lambda": "λ",
    r"\mu": "μ", r"\phi": "φ", r"\tau": "τ", r"\rho": "ρ", r"\epsilon": "ε",
    r"\pi": "π", r"\infty": "∞",
    r"\sum": "Σ", r"\prod": "Π", r"\oint": "∮", r"\int": "∫",
    r"\partial": "∂", r"\nabla": "∇",
    r"\Longleftrightarrow": "⇐⇒", r"\longleftrightarrow": "⇐⇒",
    r"\Leftrightarrow": "⇐⇒", r"\leftrightarrow": "⇐⇒", r"\iff": "⇐⇒",
    r"\rightarrow": "→", r"\to": "→",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠",
    r"\approx": "≈", r"\equiv": "≡", r"\pm": "±",
    r"\times": "x", r"\cdot": "*", r"\ast": "*",
    r"\cos": "cos", r"\sin": "sin", r"\tan": "tan", r"\log": "log",
    r"\ln": "ln", r"\exp": "exp", r"\sqrt": "sqrt", r"\max": "max",
    r"\min": "min", r"\arg": "arg",
}

_MARKUP = (
    r"\begin{cases}", r"\end{cases}", r"\begin{array}", r"\end{array}",
    r"\begin{matrix}", r"\end{matrix}", r"\begin{bmatrix}", r"\end{bmatrix}",
    r"\begin{aligned}", r"\end{aligned}", r"\begin{equation}", r"\end{equation}",
    r"\displaystyle", r"\textstyle", r"\limits", r"\nolimits",
    r"\left", r"\right", r"\quad", r"\qquad", r"\thinspace", r"\!",
    r"\,", r"\;", r"\:", r"\\", "$$", "$", r"\(", r"\)", r"\[", r"\]",
)

_WRAPPED = re.compile(r"\\(?:text|mathrm|mathbf|mathit|operatorname|mbox)\s*\{")


def _closing(text: str, start: int) -> int:
    """Index just past the `}` that closes the `{` at `start`, or -1."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _unwrap(text: str) -> str:
    """`\\text{otherwise}` becomes `otherwise`."""
    for _ in range(20):
        found = _WRAPPED.search(text)
        if not found:
            break
        brace = found.end() - 1
        close = _closing(text, brace)
        if close < 0:
            text = text[:found.start()] + text[found.end():]
            continue
        text = text[:found.start()] + text[brace + 1:close - 1] + text[close:]
    return text


def unfrac(text: str) -> str:
    """`\\frac{A}{B}` becomes `(A)/(B)`, so a fraction survives as something searchable."""
    for _ in range(12):
        at = text.find("\\frac")
        if at < 0:
            break
        first = text.find("{", at)
        if first < 0:
            text = text.replace("\\frac", "/", 1)
            continue
        end_first = _closing(text, first)
        if end_first < 0:
            text = text.replace("\\frac", "/", 1)
            continue
        second = text.find("{", end_first)
        if second < 0 or text[end_first:second].strip():
            text = text[:at] + "(" + text[first + 1:end_first - 1] + ")/" + text[end_first:]
            continue
        end_second = _closing(text, second)
        if end_second < 0:
            text = text.replace("\\frac", "/", 1)
            continue
        text = (text[:at] + "(" + text[first + 1:end_first - 1] + ")/("
                + text[second + 1:end_second - 1] + ")" + text[end_second:])
    return text


def plain(text: str) -> str:
    """A LaTeX transcription, spelled the way the course corpus spells mathematics.

    Used for the retrieval query and nothing else: the student still sees the LaTeX,
    and the model is still sent the LaTeX. Readable rather than canonical, so the
    words and the line structure survive for the lexical half of the search.
    """
    text = text or ""
    text = _unwrap(text)
    text = unfrac(text)
    for command in sorted(_SYMBOLS, key=len, reverse=True):
        text = text.replace(command, _SYMBOLS[command])
    for markup in _MARKUP:
        text = text.replace(markup, " ")
    text = text.replace("{", "").replace("}", "").replace("&", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


# ---------------------------------------------------------------------------
# Tesseract: the fallback engine
# ---------------------------------------------------------------------------

# Where the engine is, when it is not on PATH. The Windows installer does not always
# put it there, and a server started from a shell opened before the install will not
# see it either - which would report the feature as unavailable on a machine where it
# is installed and working. An explicit path beats asking the student to fix their
# environment.
_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)


def binary() -> str:
    """The Tesseract executable, or an empty string if this machine has none."""
    override = os.environ.get("TESSERACT_CMD")
    if override and os.path.isfile(override):
        return override
    found = shutil.which("tesseract")
    if found:
        return found
    return next((path for path in _CANDIDATES if os.path.isfile(path)), "")


def _tesseract_ready() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return bool(binary())


def _prepare(image):
    """Greyscale, and scale small screenshots up.

    Tesseract is trained around roughly 300 dpi and does visibly worse on the small
    text a screenshot of a PDF viewer produces; enlarging a small image before
    reading it costs nothing and recovers characters that would otherwise be lost.
    The vision engine does its own resizing and does not use this.
    """
    from PIL import Image

    prepared = image.convert("L")
    shortest = min(prepared.size)
    if shortest < 1000:
        factor = min(3.0, 1000 / max(shortest, 1))
        prepared = prepared.resize(
            (int(prepared.width * factor), int(prepared.height * factor)),
            Image.LANCZOS,
        )
    return prepared


def _read_tesseract(image) -> tuple[str, list[str]]:
    import pytesseract

    # Point the wrapper at the executable found above, so a machine that has the
    # engine installed but not on PATH still works.
    pytesseract.pytesseract.tesseract_cmd = binary()
    try:
        text = pytesseract.image_to_string(_prepare(image))
    except Exception as exc:  # the binary is there but refused the image
        raise OcrFailed(f"the image could not be read: {exc}") from exc
    return text, ["read by character recognition, which misreads mathematics"]


# ---------------------------------------------------------------------------
# The vision-language engine: the primary
# ---------------------------------------------------------------------------

# Overridable so the evaluation can hold one engine against another without an edit.
#
# The full-precision checkpoint, quantized here at load rather than downloaded
# ready-quantized. That is not a preference: the community 4-bit build of this
# model family wraps the eight vision merger layers as 4-bit without packing their
# weights, and generation dies on the first image with a shape assertion inside
# bitsandbytes. Quantizing here also allows the vision tower to be left alone,
# which is the half of the model that has to resolve a subscript from a dozen
# pixels.
#
# 4B rather than 8B because of the host, not the card. The 8B checkpoint is four
# shards of about five gigabytes, and the loader holds all four open at once: that
# wants roughly 17 GB of Windows commit on a machine with 15 GB of memory, and the
# load fails before any of it reaches the GPU. The 4B loads in half that and leaves
# the graphics card with room for the tutor beside it, which the 8B never had on a
# 16 GB card shared with a desktop.
VLM_MODEL_ID = os.environ.get("YM07_VLM", "Qwen/Qwen3-VL-4B-Instruct")

# Kept in bfloat16 while the language half goes to 4-bit. Matched as substrings
# against each module's dotted name, so `visual` covers the whole vision tower
# including the deepstack mergers.
VLM_KEEP_FULL = ("visual", "lm_head", "embed_tokens")

# A question fits in far less than this. The cap is what stops a model that has
# started repeating itself from holding the request open. It is set well above what
# a question needs because the densest thing a student screenshots is the course
# information sheet - nine transform pairs, each a line of nested LaTeX - and a
# transcription cut off halfway is worse than a bad one: it looks complete.
VLM_MAX_NEW_TOKENS = 1200

# Vision tokens scale with pixels, and they land in the same KV cache as the
# tutor's own prompt on a card that is already the binding constraint. About a
# megapixel is more than a screenshot of a question needs and bounds both the
# memory and the latency.
VLM_MAX_PIXELS = 1_003_520

# Transcription, explicitly not answering. The rules are in the order they matter:
# the model's instinct on being shown an exam question is to solve it, so that is
# ruled out first. The notation clauses name the exact errors the character engine
# makes on this course's material, because naming them is what stops the vision
# model making them too - it will write a bare "d" for a delta if nothing asks it
# not to.
TRANSCRIBE_PROMPT = """Transcribe this image into text, exactly as it appears.

It is course material from a university digital signal processing course - most
often a tutorial, test or past examination question.

Rules:
- Transcribe only. Do not solve it, answer it, explain it or comment on it.
- Write mathematics in LaTeX. Put inline mathematics between single dollar signs
  and displayed equations between double dollar signs.
- Keep every subscript and superscript: write $h_1[n]$, not $h1[n]$ or $hy[n]$.
- Keep Greek letters as LaTeX commands: $\\delta[n-2]$, $\\omega$, $\\pi$,
  $\\sigma^2$. Never transcribe a Greek letter as the digit or Latin letter it
  looks like.
- Keep the numbering and lettering of parts, and the line structure of the
  original.
- Describe a figure, plot or circuit in square brackets on its own line, like
  [figure: a pole-zero plot with two poles inside the unit circle], and transcribe
  any axis labels and annotations you can read.
- If something is genuinely illegible, write [illegible] in its place rather than
  guessing.
- Output the transcription and nothing else. No preamble, no code fence."""

_vlm_lock = threading.Lock()
_vlm: tuple | None = None
# Why the load failed, if it did. Kept so that a machine which cannot run the
# engine falls through to Tesseract once instead of attempting a doomed load on
# every request a student makes.
_vlm_error = ""


def vlm_cached() -> bool:
    """Whether the weights are already on this machine.

    Checked before offering the engine, because the alternative is a student's
    first question hanging for the length of an eight-gigabyte download.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    if os.path.isdir(VLM_MODEL_ID):
        return True
    try:
        snapshot_download(VLM_MODEL_ID, local_files_only=True)
        return True
    except Exception:
        return False


def _vlm_ready() -> bool:
    if _vlm is not None:
        return True
    if _vlm_error:
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and vlm_cached()


def _load_vlm() -> tuple:
    """Load the reader once and keep it. Raises OcrUnavailable if it cannot."""
    global _vlm, _vlm_error

    if _vlm is not None:
        return _vlm
    with _vlm_lock:
        if _vlm is not None:
            return _vlm
        if _vlm_error:
            raise OcrUnavailable(_vlm_error)
        try:
            import torch
            from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                      BitsAndBytesConfig)

            processor = AutoProcessor.from_pretrained(
                VLM_MODEL_ID, max_pixels=VLM_MAX_PIXELS
            )
            kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
            # A checkpoint that is already quantized carries its own config and
            # rejects a second one.
            prequantized = "4bit" in VLM_MODEL_ID.lower() or "bnb" in VLM_MODEL_ID.lower()
            if not prequantized and torch.cuda.is_available():
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=list(VLM_KEEP_FULL),
                )
            model = AutoModelForImageTextToText.from_pretrained(VLM_MODEL_ID, **kwargs)
            model.eval()
        except Exception as exc:
            # Remember the reason rather than the fact, so the fallback note the
            # student sees can say what went wrong.
            detail = str(exc)
            if "paging file" in detail or "os error 1455" in detail.lower():
                # Not a GPU problem, and the message Windows gives does not say so.
                detail = ("this machine does not have enough memory to open the "
                          "checkpoint - close what else is running, or set YM07_VLM "
                          "to a smaller reader")
            _vlm_error = f"the vision reader could not be loaded ({detail})"
            raise OcrUnavailable(_vlm_error) from exc
        _vlm = (processor, model)
        return _vlm


def vlm_loaded() -> bool:
    """Whether the reader is on the card now."""
    return _vlm is not None


def free_vlm() -> None:
    """Release the reader's share of the card.

    The tutor and the reader compete for one 16 GB card. The server calls this before
    the local model generates (web.Pipeline, E30): with the reader resident, the local
    model's next answer did not fit, Windows paged 2.8 GB of it into system RAM, and an
    answer that takes 50 s took 839 s, for the rest of the session. The next screenshot
    pays about 16 s to load the reader again. A hosted model needs no card, so a
    session on Opus 5 keeps the reader loaded. The evaluation harness also loads several
    engines in one process and puts each one down with this.
    """
    global _vlm
    with _vlm_lock:
        _vlm = None
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _read_vlm(image) -> tuple[str, list[str]]:
    import torch

    processor, model = _load_vlm()
    # The engine wants RGB; a screenshot saved as a palette or RGBA PNG is common.
    picture = _enlarge(image.convert("RGB"))

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": picture},
                    {"type": "text", "text": TRANSCRIBE_PROMPT}],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    try:
        with torch.inference_mode():
            # Greedy, like every other generation in this project. A transcription
            # has one right answer and sampling one of several readings of the same
            # image would make the same screenshot produce different questions.
            out = model.generate(
                **inputs, max_new_tokens=VLM_MAX_NEW_TOKENS, do_sample=False,
            )
    except torch.cuda.OutOfMemoryError as exc:
        raise OcrFailed(
            "the card ran out of memory reading that image. A tighter screenshot "
            "of just the question should work."
        ) from exc

    fresh = out[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(fresh, skip_special_tokens=True)
    # A reading that stopped because it ran out of room looks exactly like a
    # complete one, which is the worst way for it to fail. Say so.
    notes = (["the image held more than the reader could transcribe in one go"]
             if len(fresh) >= VLM_MAX_NEW_TOKENS else [])
    return _unfence(text), notes


# Below this, a screenshot is enlarged before the reader sees it. The character
# engine has done this from the start and the vision engine did not, which was an
# oversight rather than a decision: a student screenshotting one question out of a
# PDF viewer produces something like 700x116, and at that size the difference
# between `u[n-5]` and `u[n+5]` is a few pixels of one glyph. That exact misreading
# turned a question about a non-causal response into a question about a causal one,
# and the tutor answered the wrong one correctly.
#
# The reading set this was measured on was cropped at 140 dpi, around 250k pixels,
# so it never exercised this: the 92% figure belongs to images about three times
# larger than the one that failed.
VLM_MIN_PIXELS = 450_000


def _enlarge(picture):
    """Scale a small screenshot up, bounded by what the reader will accept."""
    from PIL import Image

    pixels = picture.width * picture.height
    if pixels >= VLM_MIN_PIXELS:
        return picture
    factor = min((VLM_MIN_PIXELS / max(pixels, 1)) ** 0.5,
                 (VLM_MAX_PIXELS / max(pixels, 1)) ** 0.5, 4.0)
    if factor <= 1.01:
        return picture
    return picture.resize((int(picture.width * factor), int(picture.height * factor)),
                          Image.LANCZOS)


def _unfence(text: str) -> str:
    """Drop a markdown code fence the model wrapped its answer in despite being asked not to."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]  # the opening fence, with or without a language tag
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Choosing between them
# ---------------------------------------------------------------------------

# Best first. The order is the claim that the vision engine reads this course's
# material better, which Section 5 measures rather than assumes.
_ORDER = ("vision", "tesseract")

_READY = {"vision": _vlm_ready, "tesseract": _tesseract_ready}
_READER = {"vision": _read_vlm, "tesseract": _read_tesseract}

ENGINE_NAMES = {
    "vision": VLM_MODEL_ID.rsplit("/", 1)[-1],
    "tesseract": "Tesseract",
}


def engines() -> list[str]:
    """Which engines this machine can run, best first."""
    return [name for name in _ORDER if _READY[name]()]


def available() -> tuple[bool, str]:
    """Whether an image can be read here, and which engine would do it."""
    ready = engines()
    return (True, ready[0]) if ready else (False, "")


def extract(data: bytes, *, engine: str | None = None) -> OcrResult:
    """Read an uploaded image. Raises rather than returning silence.

    `engine` forces one of `_ORDER` instead of taking the best available, which is
    what the evaluation uses to put the two readings of the same image side by
    side. Left alone, a vision engine that fails at request time falls through to
    Tesseract rather than failing the student's question.
    """
    image = _load(data)
    wanted = [engine] if engine else engines()
    if engine and engine not in _READER:
        raise OcrFailed(f"{engine!r} is not an engine ({', '.join(_ORDER)}).")
    if not wanted:
        raise OcrUnavailable(
            "no way to read images is installed on this machine. Either fetch the "
            "vision reader with `ym07 fetch-reader`, or install Tesseract and "
            "`pip install -e '.[ocr]'`, then attach the screenshot again."
        )

    notes: list[str] = []
    started = time.perf_counter()
    for index, name in enumerate(wanted):
        try:
            text, engine_notes = _READER[name](image)
        except (OcrUnavailable, OcrFailed) as exc:
            remaining = wanted[index + 1:]
            if not remaining:
                raise
            # Say what was lost. A student whose mathematics came back garbled is
            # owed the reason, since the fallback is the engine that garbles it.
            notes.append(f"{ENGINE_NAMES[name]} was unavailable ({exc})")
            continue
        notes += engine_notes
        break
    else:  # pragma: no cover - the loop either breaks or raises
        raise OcrUnavailable("no engine could read the image.")

    text = _tidy(text)
    truncated = len(text) > MAX_EXTRACTED_CHARS
    if truncated:
        text = text[:MAX_EXTRACTED_CHARS]
        notes.append("only the first part of the image was read")
    # Area, not the shorter side. A question cropped out of a page is wide and thin -
    # the first version of this check measured the short side and warned that a
    # perfectly legible 1019x222 crop was too small to trust, which is the shape
    # every screenshot of a question has.
    if image.width * image.height < 150_000:
        notes.append("the image is small, so the reading may be unreliable")

    return OcrResult(text=text.strip(), engine=name,
                     width=image.width, height=image.height,
                     truncated=truncated, notes=notes,
                     latency_s=round(time.perf_counter() - started, 2))
