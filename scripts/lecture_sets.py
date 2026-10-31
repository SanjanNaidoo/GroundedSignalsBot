"""The two test sets of protocols/lectures.md, drawn and registered before any transcript is read.

    python scripts/lecture_sets.py

E22 - twenty thirty-second segments for hand transcription: twelve from the DSP series,
eight from the ML series, no two from the same recording, none in the first or last
ninety seconds, drawn with seed 20261011 from the recordings' own durations (read from
the MP4 headers, so no transcript is needed). Written to lectures/e22-segments.json with
empty fields for the hand transcription and the hand-read time of a distinctive phrase.

E24 - forty questions about lecture-taught concepts, in a student's wording: twenty
worded as concept questions and twenty as the calculation families, half DSP and half
ML, plus four about material the lectures should not cover. Split by the same seed
into a development half and a held-back half, written to stress/bank/lectures/, and
registered with a digest in the manifest beside them. The relevant windows are labelled
later, by reading the transcripts; the questions are fixed now so that the index cannot
be built around them.

Both files refuse to be overwritten once they exist: the sets are registered once.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "Lecture Videos"
E22_OUT = ROOT / "lectures" / "e22-segments.json"
E24_DIR = ROOT / "stress" / "bank" / "lectures"
SEED = 20261011
SEGMENT_S = 30.0
MARGIN_S = 90.0
DSP_NAME = re.compile(r"^L\d{2}[a-z]$")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def mp4_duration(path: Path) -> float | None:
    """Seconds, from the movie header, without decoding anything."""
    total = path.stat().st_size
    with path.open("rb") as fh:
        pos = 0
        while pos + 8 <= total:
            fh.seek(pos)
            head = fh.read(16)
            size, kind = struct.unpack(">I4s", head[:8])
            header = 8
            if size == 1:
                size = struct.unpack(">Q", head[8:16])[0]
                header = 16
            elif size == 0:
                size = total - pos
            if size < header:
                return None
            if kind == b"moov":
                inner, end = pos + header, pos + size
                while inner + 8 <= end:
                    fh.seek(inner)
                    isize, ikind = struct.unpack(">I4s", fh.read(8))
                    if isize < 8:
                        return None
                    if ikind == b"mvhd":
                        body = fh.read(32)
                        if body[0] == 1:
                            timescale, duration = struct.unpack(">IQ", body[20:32])
                        else:
                            timescale, duration = struct.unpack(">II", body[12:20])
                        return duration / timescale if timescale else None
                    inner += isize
                return None
            pos += size
    return None


def recordings() -> list[tuple[str, str, float]]:
    seen: dict[str, Path] = {}
    for p in sorted(VIDEOS.rglob("*")):
        if p.suffix.lower() in (".mp4", ".m4v") and p.is_file():
            seen.setdefault(p.name, p)
    out = []
    for name, p in sorted(seen.items()):
        d = mp4_duration(p)
        if d:
            out.append((name, "dsp" if DSP_NAME.match(p.stem) else "ml", d))
    return out


def draw_e22(recs: list[tuple[str, str, float]]) -> list[dict]:
    rng = random.Random(SEED)
    picks = []
    for series, n in (("dsp", 12), ("ml", 8)):
        pool = [r for r in recs if r[1] == series and r[2] > 2 * MARGIN_S + SEGMENT_S]
        for name, _, dur in rng.sample(pool, n):
            start = round(rng.uniform(MARGIN_S, dur - MARGIN_S - SEGMENT_S), 1)
            picks.append({"id": f"e22-{len(picks) + 1:02d}", "series": series, "recording": name,
                          "start_s": start, "end_s": round(start + SEGMENT_S, 1),
                          "start": f"{int(start // 60):02d}:{int(start % 60):02d}",
                          "reference_text": "", "phrase": "", "phrase_start_s": None})
    return picks


# The forty questions. `expect` says whether the lectures should hold an answer; the four
# marked none ask about material outside the course, so the right result is no reference.
QUESTIONS = [
    # DSP, concept wording
    ("dsp", "concept", "sampling", "Why does sampling a signal below twice its highest frequency cause aliasing, and what does the aliased frequency look like?"),
    ("dsp", "concept", "dt-signals-systems", "What does it mean for a discrete-time system to be causal, and how can I tell from its impulse response?"),
    ("dsp", "concept", "dt-signals-systems", "How does the absolute summability of h[n] relate to BIBO stability?"),
    ("dsp", "concept", "ztransform", "What is the region of convergence of the z-transform and why does it matter for stability?"),
    ("dsp", "concept", "dtft-dft", "What is the difference between the DTFT and the DFT?"),
    ("dsp", "concept", "spectrum-estimation", "Why does zero-padding a sequence before the DFT not improve the frequency resolution?"),
    ("dsp", "concept", "spectrum-estimation", "What is spectral leakage and how does a window function reduce it?"),
    ("dsp", "concept", "ztransform", "How do I read the frequency response of a system off its pole-zero plot?"),
    ("dsp", "concept", "dt-signals-systems", "How does convolution describe the output of an LTI system?"),
    ("dsp", "concept", "sampling", "What does the zero-order hold do when reconstructing a continuous signal from its samples?"),
    # DSP, calculation-family wording
    ("dsp", "calc", "convolution", "How do I compute the linear convolution of two short sequences by hand?"),
    ("dsp", "calc", "aliasing", "If a 6 kHz tone is sampled at 8 kHz, what frequency does it appear at?"),
    ("dsp", "calc", "dft_bins", "How do I work out which DFT bin a tone of a given frequency lands in?"),
    ("dsp", "calc", "first_order", "How do I find the impulse response of a first-order system from its difference equation?"),
    ("dsp", "calc", "partial_fractions", "How do I invert a z-transform using partial fractions?"),
    ("dsp", "calc", "record_length", "How long a record do I need to resolve two tones a given frequency apart?"),
    ("dsp", "calc", "dtft-dft", "How do I do circular convolution using the DFT?"),
    ("dsp", "calc", "dtft-dft", "Why is the FFT faster than computing the DFT directly?"),
    # ML, concept wording
    ("ml", "concept", "ml-evaluation", "What is the difference between overfitting and underfitting, and how does the bias-variance trade-off explain it?"),
    ("ml", "concept", "ml-evaluation", "Why do we hold out a test set, and what is three-way holdout?"),
    ("ml", "concept", "ml-evaluation", "How does k-fold cross-validation work and when should I use it?"),
    ("ml", "concept", "ml-supervised", "What does regularisation do to the weights of a regression model?"),
    ("ml", "concept", "neural-networks", "How does a ReLU activation work and why is it used?"),
    ("ml", "concept", "neural-networks", "What does backpropagation compute, and where does the chain rule come in?"),
    ("ml", "concept", "neural-networks", "Why does a convolutional layer share its parameters across the image?"),
    ("ml", "concept", "neural-networks", "What do padding and stride do to the output size of a convolution?"),
    ("ml", "concept", "reinforcement-learning", "What is a Markov decision process, and what are its states, actions and rewards?"),
    ("ml", "concept", "reinforcement-learning", "What is the difference between a policy and a value function?"),
    # ML, calculation-family wording
    ("ml", "calc", "knn", "How do I classify a point with k-nearest neighbours when k is 3?"),
    ("ml", "calc", "gradient_descent", "How do I take one gradient descent step on a simple cost function?"),
    ("ml", "calc", "forward_pass", "How do I compute the forward pass of a small network by hand?"),
    ("ml", "calc", "output_gradient", "How do I compute the gradient of the loss with respect to the output-layer weights?"),
    ("ml", "calc", "td_update", "How does the TD(0) update rule change the value estimate of a state?"),
    ("ml", "calc", "ml-supervised", "How do I fit a polynomial regression and choose its degree?"),
    ("ml", "calc", "neural-networks", "How do I count the parameters of a convolutional layer?"),
    ("ml", "calc", "ml-evaluation", "How does repeated holdout differ from a single train-test split?"),
    # Outside the course: the right answer is no lecture reference
    ("none", "none", "control", "How do I find the phase margin from a Bode plot?"),
    ("none", "none", "information-theory", "How does Huffman coding assign shorter codes to more frequent symbols?"),
    ("none", "none", "transformers", "How does self-attention work in a transformer?"),
    ("none", "none", "circuits", "How do I compute the Laplace transform of a step input to an RC circuit?"),
]


def build_e24() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    items = [{"id": f"lec-{i + 1:02d}", "series": s, "kind": k, "topic": t, "question": q,
              "expect": "none" if s == "none" else "lecture", "relevant_windows": []}
             for i, (s, k, t, q) in enumerate(QUESTIONS)]
    dev, final = [], []
    for group in sorted({(i["series"], i["kind"]) for i in items}):
        members = [i for i in items if (i["series"], i["kind"]) == group]
        rng.shuffle(members)
        half = len(members) // 2
        dev.extend(members[:half])
        final.extend(members[half:])
    dev.sort(key=lambda i: i["id"])
    final.sort(key=lambda i: i["id"])
    return dev, final


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; the set is registered and is not rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    recs = recordings()
    print(f"{len(recs)} recordings, {sum(r[2] for r in recs) / 3600:.1f} h")

    picks = draw_e22(recs)
    write_once(E22_OUT, json.dumps({"seed": SEED, "segment_s": SEGMENT_S, "margin_s": MARGIN_S,
                                    "segments": picks}, indent=1, ensure_ascii=False) + "\n")
    print(f"E22: {len(picks)} segments -> {E22_OUT.relative_to(ROOT)}")
    for p in picks:
        print(f"   {p['id']}  {p['series']}  {p['recording']:<48} {p['start']}")

    dev, final = build_e24()
    for name, items in (("lectures-dev.jsonl", dev), ("lectures-final.jsonl", final)):
        write_once(E24_DIR / name, "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    manifest = {"seed": SEED, "written": "2026-10-11",
                "dev": {"file": "lectures-dev.jsonl", "items": len(dev), "digest": digest(E24_DIR / "lectures-dev.jsonl")},
                "final": {"file": "lectures-final.jsonl", "items": len(final), "digest": digest(E24_DIR / "lectures-final.jsonl")},
                "note": "Questions fixed before any transcript was read or any index built; "
                        "relevant_windows are labelled afterwards from the transcripts."}
    write_once(E24_DIR / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    print(f"E24: {len(dev)} development and {len(final)} held-back questions -> {E24_DIR.relative_to(ROOT)}; "
          f"digests {manifest['dev']['digest']} / {manifest['final']['digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
