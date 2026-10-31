"""E22 - how good are the transcripts, against twenty segments transcribed by hand?

    python scripts/e22_score.py                 # scores the segments whose reference is filled in
    python scripts/e22_score.py --self-check    # the machine text as its own reference: must give 0% WER

The references live in lectures/e22-segments.json: for each segment, `reference_text`
(the words spoken, typed from the clip), `phrase` (one distinctive technical phrase in
it) and the time that phrase begins, either `phrase_start_s` in recording time or
`phrase_start_in_clip_s` read off the clip player (the clip starts `clip_lead_s` before
the segment). The machine's words are taken from the recording's transcript by their
timestamps: every word whose midpoint lies inside the segment.

Metrics and pass marks are the protocol's (protocols/lectures.md, E22), fixed before any
transcript was read:

- word error rate, per segment and overall, after both texts are lower-cased, punctuation
  stripped and hyphens made spaces (T1: at most 15% overall);
- technical-term error rate against the protocol's list (T2: at least 80% intact);
- timestamp drift between the hand-read start of the phrase and the transcript's word
  timestamp for its first word (T3: median at most 2 s, maximum at most 10 s);
- speed from the transcription manifest (T4: at least 5x real time).

The result is written to stress/results/lectures/e22.json with every segment's counts,
so the report's table comes from the file.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "lectures" / "e22-segments.json"
TRANSCRIPTS = ROOT / "lectures" / "transcripts"
RESULTS = ROOT / "stress" / "results" / "lectures" / "e22.json"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# The protocol's list, verbatim, normalised the same way as the texts.
TERMS = [
    "discrete-time", "continuous-time", "sampling", "aliasing", "Nyquist", "impulse response",
    "convolution", "causal", "BIBO", "stable", "DTFT", "DFT", "FFT", "z-transform",
    "region of convergence", "pole", "zero", "unit circle", "frequency response", "window",
    "leakage", "spectrum", "periodogram", "zero-order hold", "reconstruction", "linear",
    "time-invariant", "difference equation", "partial fractions", "regression",
    "gradient descent", "learning rate", "overfitting", "bias", "variance", "holdout",
    "cross-validation", "k-fold", "regularisation", "neural network", "backpropagation",
    "activation", "ReLU", "convolutional", "kernel", "stride", "padding", "pooling", "Markov",
    "policy", "value function", "reward", "temporal difference", "Q-learning",
]

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)

# A second, declared normalisation reported beside the protocol's (amendment of 12 Oct):
# number words as digits, and the spoken contractions the engine writes out in full.
# It is applied to both texts and changes no pass mark; its effect is printed so the two
# figures can be compared.
NUMBER_WORDS = {"zero": "0", "nought": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
                "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
                "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"}
CONTRACTIONS = {"gonna": ["going", "to"], "wanna": ["want", "to"], "gotta": ["got", "to"],
                "kinda": ["kind", "of"], "sorta": ["sort", "of"], "theyre": ["they", "are"],
                "youre": ["you", "are"], "im": ["i", "am"], "ive": ["i", "have"], "weve": ["we", "have"],
                "youve": ["you", "have"], "theyve": ["they", "have"], "dont": ["do", "not"],
                "doesnt": ["does", "not"], "didnt": ["did", "not"], "cant": ["can", "not"],
                "cannot": ["can", "not"], "wont": ["will", "not"], "isnt": ["is", "not"],
                "arent": ["are", "not"], "wasnt": ["was", "not"], "werent": ["were", "not"],
                "havent": ["have", "not"], "hasnt": ["has", "not"], "thats": ["that", "is"],
                "whats": ["what", "is"], "theres": ["there", "is"], "heres": ["here", "is"],
                "ill": ["i", "will"], "youll": ["you", "will"], "well": ["well"], "id": ["id"]}


def normalise(text: str) -> list[str]:
    text = text.lower().replace("-", " ").replace("’", "'")
    text = _PUNCT.sub(" ", text).replace("'", "")
    return text.split()


def normalise_extended(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        if t in CONTRACTIONS:
            out.extend(CONTRACTIONS[t])
        else:
            out.append(NUMBER_WORDS.get(t, t))
    return out


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Substitutions, deletions, insertions of the cheapest alignment of hyp to ref."""
    n, m = len(ref), len(hyp)
    # cost[i][j] = (total, S, D, I) for ref[:i] vs hyp[:j]
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = prev[j - 1]
            else:
                sub = prev[j - 1]
                best = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
            dele = prev[j]
            cand = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
            if cand[0] < best[0]:
                best = cand
            ins = cur[j - 1]
            cand = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)
            if cand[0] < best[0]:
                best = cand
            cur.append(best)
        prev = cur
    _, s, d, i = prev[m]
    return s, d, i


def count_phrase(words: list[str], phrase: list[str]) -> int:
    if not phrase:
        return 0
    k = len(phrase)
    return sum(1 for p in range(len(words) - k + 1) if words[p:p + k] == phrase)


def find_phrase_time(words_with_time: list[tuple[str, float]], phrase: list[str], near: float) -> float | None:
    """The transcript time of the phrase's first word, at the occurrence nearest `near`."""
    if not phrase:
        return None
    texts = [w for w, _ in words_with_time]
    for length in (min(2, len(phrase)), 1):
        key = phrase[:length]
        starts = [t for p, (w, t) in enumerate(words_with_time) if texts[p:p + length] == key]
        if starts:
            return min(starts, key=lambda t: abs(t - near))
    return None


def machine_words(transcript: dict, start_s: float, end_s: float, margin: float = 0.0) -> list[tuple[str, float]]:
    out = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []) or []:
            mid = (w["start"] + w["end"]) / 2
            if start_s - margin <= mid < end_s + margin:
                for token in normalise(w["word"]):
                    out.append((token, w["start"]))
    return out


def speed_from_manifest() -> dict:
    rows = {}
    path = TRANSCRIPTS / "manifest.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["recording"]] = r
    audio = sum(r["duration_s"] for r in rows.values())
    wall = sum(r["wall_s"] for r in rows.values())
    return {"recordings": len(rows), "audio_h": round(audio / 3600, 2), "wall_h": round(wall / 3600, 2),
            "realtime_factor": round(audio / wall, 1) if wall else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true",
                        help="use the machine text as the reference, to prove the scorer gives zero error")
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()

    data = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    lead = float(data.get("clips", {}).get("lead_s", 0.0))
    term_tokens = [normalise(t) for t in TERMS]
    rows = []
    for seg in data["segments"]:
        stem = Path(seg["recording"]).stem
        tpath = TRANSCRIPTS / f"{stem}.json"
        if not tpath.exists():
            rows.append({"id": seg["id"], "recording": seg["recording"], "status": "no transcript"})
            continue
        transcript = json.loads(tpath.read_text(encoding="utf-8"))
        # The reference is typed from the clip, which runs `lead` seconds either side of
        # the segment, so the machine's words are taken over the same span.
        clip_lead = float(seg.get("clip_lead_s", lead))
        hyp_words = machine_words(transcript, seg["start_s"] - clip_lead, seg["end_s"] + clip_lead)
        hyp = [w for w, _ in hyp_words]
        if args.self_check:
            ref = list(hyp)
        else:
            ref = normalise(seg.get("reference_text") or "")
        if not ref:
            rows.append({"id": seg["id"], "recording": seg["recording"], "status": "no reference yet",
                         "machine_words": len(hyp)})
            continue
        s, d, i = edit_counts(ref, hyp)
        s_x, d_x, i_x = edit_counts(normalise_extended(ref), normalise_extended(hyp))
        term_total = term_intact = 0
        terms_missed = []
        for term, toks in zip(TERMS, term_tokens):
            n_ref = count_phrase(ref, toks)
            if n_ref:
                n_hyp = count_phrase(hyp, toks)
                term_total += n_ref
                term_intact += min(n_ref, n_hyp)
                if n_hyp < n_ref:
                    terms_missed.append(term)
        phrase = normalise(seg.get("phrase") or "")
        ref_time = seg.get("phrase_start_s")
        if ref_time is None and seg.get("phrase_start_in_clip_s") is not None:
            ref_time = seg["start_s"] - lead + float(seg["phrase_start_in_clip_s"])
        drift = None
        found_time = None
        if phrase and ref_time is not None:
            wide = machine_words(transcript, seg["start_s"] - clip_lead, seg["end_s"] + clip_lead, margin=10.0)
            found_time = find_phrase_time(wide, phrase, float(ref_time))
            drift = abs(found_time - float(ref_time)) if found_time is not None else None
        rows.append({
            "id": seg["id"], "recording": seg["recording"], "series": seg["series"],
            "start_s": seg["start_s"], "status": "scored",
            "reference_words": len(ref), "machine_words": len(hyp),
            "substitutions": s, "deletions": d, "insertions": i,
            "wer": round((s + d + i) / len(ref), 4),
            "wer_extended_normalisation": round((s_x + d_x + i_x) / len(ref), 4),
            "errors_extended": [s_x, d_x, i_x],
            "terms_in_reference": term_total, "terms_intact": term_intact, "terms_missed": terms_missed,
            "phrase": seg.get("phrase") or "", "phrase_reference_time_s": ref_time,
            "phrase_transcript_time_s": found_time, "phrase_drift_s": round(drift, 2) if drift is not None else None,
        })

    scored = [r for r in rows if r["status"] == "scored"]
    ref_words = sum(r["reference_words"] for r in scored)
    errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in scored)
    errors_x = sum(sum(r["errors_extended"]) for r in scored)
    insertions_x = sum(r["errors_extended"][2] for r in scored)
    terms_total = sum(r["terms_in_reference"] for r in scored)
    terms_intact = sum(r["terms_intact"] for r in scored)
    drifts = [r["phrase_drift_s"] for r in scored if r["phrase_drift_s"] is not None]
    phrases_unfound = [r["id"] for r in scored if r["phrase"] and r["phrase_drift_s"] is None]
    speed = speed_from_manifest()
    summary = {
        "segments_scored": len(scored), "segments_total": len(rows),
        "self_check": args.self_check,
        "reference_words": ref_words,
        "wer_overall": round(errors / ref_words, 4) if ref_words else None,
        "wer_overall_extended_normalisation": round(errors_x / ref_words, 4) if ref_words else None,
        "wer_extended_without_insertions": round((errors_x - insertions_x) / ref_words, 4) if ref_words else None,
        "wer_median_segment": round(statistics.median(r["wer"] for r in scored), 4) if scored else None,
        "terms_in_references": terms_total, "terms_intact": terms_intact,
        "terms_intact_share": round(terms_intact / terms_total, 4) if terms_total else None,
        "drift_measured": len(drifts), "phrases_not_found": phrases_unfound,
        "drift_median_s": round(statistics.median(drifts), 2) if drifts else None,
        "drift_max_s": round(max(drifts), 2) if drifts else None,
        "speed": speed,
        "T1_pass": (errors / ref_words <= 0.15) if ref_words else None,
        "T2_pass": (terms_intact / terms_total >= 0.80) if terms_total else None,
        "T3_pass": (statistics.median(drifts) <= 2.0 and max(drifts) <= 10.0 and not phrases_unfound) if drifts else None,
        "T4_pass": (speed["realtime_factor"] >= 5.0) if speed["realtime_factor"] else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment": "E22", "summary": summary, "segments": rows}, indent=1,
                              ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    print(f"E22: {len(scored)} of {len(rows)} segments scored" + (" (self-check)" if args.self_check else ""))
    for r in rows:
        if r["status"] != "scored":
            print(f"  {r['id']}  {r['recording']:<40s} {r['status']}")
        else:
            drift = f"{r['phrase_drift_s']:.1f} s" if r["phrase_drift_s"] is not None else ("not found" if r["phrase"] else "-")
            print(f"  {r['id']}  {r['recording']:<40s} WER {r['wer']*100:5.1f}%  "
                  f"terms {r['terms_intact']}/{r['terms_in_reference']}  drift {drift}")
    if scored:
        pm = lambda v: "-" if v is None else ("pass" if v else "FAIL")  # noqa: E731
        print(f"\n  with number words as digits and contractions expanded: WER {summary['wer_overall_extended_normalisation']*100:.1f}%, "
              f"of which substitutions and deletions alone {summary['wer_extended_without_insertions']*100:.1f}%")
        print(f"  overall WER {summary['wer_overall']*100:.1f}% (T1 {pm(summary['T1_pass'])}); "
              f"terms intact {terms_intact}/{terms_total} (T2 {pm(summary['T2_pass'])}); "
              f"drift median {summary['drift_median_s']} s, max {summary['drift_max_s']} s (T3 {pm(summary['T3_pass'])}); "
              f"speed {speed['realtime_factor']}x over {speed['audio_h']} h of audio (T4 {pm(summary['T4_pass'])})")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
