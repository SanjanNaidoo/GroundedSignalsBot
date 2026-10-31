"""E22 second pass on five clips: list the real disagreements, then score the decisions.

    python scripts/e22_spotcheck.py make     # writes lectures/e22-spotcheck.json (refuses to overwrite)
    python scripts/e22_spotcheck.py score    # reads the decisions, writes stress/results/lectures/e22-spotcheck.json

The first-pass references were typed blind and the machine disagreed with them at about
five hundred places over twenty clips. Listening to all of them again costs an hour; a
seeded five-clip sample costs ten minutes and gives an estimate with its size stated.
`make` picks five clips with seed 20261011, aligns each reference with the machine's
words after both are normalised the same way the scorer's second normalisation does
(number words as digits, contractions written out), so that "gonna" against "going to"
never appears, and writes one item per remaining disagreement: the clip time, what the
reference has, what the machine has, and an empty `heard` to fill with `M` (the machine
was right), `R` (the reference was right) or the words actually said.

`score` applies the decisions to the five references and reports the word error rate on
those five clips, under the same normalisation, beside the first-pass figure on the same
five, so the two can be compared like for like. Decisions can only move a reference
towards what was heard; an item left empty keeps the reference as typed.
"""

from __future__ import annotations

import difflib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from e22_score import (CONTRACTIONS, NUMBER_WORDS, TRANSCRIPTS, edit_counts,  # noqa: E402
                       machine_words, normalise, normalise_extended)

SEGMENTS = ROOT / "lectures" / "e22-segments.json"
SHEET = ROOT / "lectures" / "e22-spotcheck.json"
RESULT = ROOT / "stress" / "results" / "lectures" / "e22-spotcheck.json"
SEED = 20261011
CLIPS = 5


def timed_tokens(words: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Extended normalisation of the machine's words, each token keeping its time."""
    out = []
    for w, t in words:
        if w in CONTRACTIONS:
            out.extend((x, t) for x in CONTRACTIONS[w])
        else:
            out.append((NUMBER_WORDS.get(w, w), t))
    return out


def load_clip(seg: dict, lead: float):
    transcript = json.loads((TRANSCRIPTS / f"{Path(seg['recording']).stem}.json").read_text(encoding="utf-8"))
    hyp = timed_tokens(machine_words(transcript, seg["start_s"] - lead, seg["end_s"] + lead))
    ref = normalise_extended(normalise(seg.get("reference_text") or ""))
    return ref, hyp


def make() -> int:
    if SHEET.exists():
        print(f"{SHEET} exists; not overwriting a sheet that may be filled in")
        return 1
    data = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    lead = float(data["clips"]["lead_s"])
    ids = [s["id"] for s in data["segments"]]
    chosen = sorted(random.Random(SEED).sample(ids, CLIPS))
    clips = []
    for seg in data["segments"]:
        if seg["id"] not in chosen:
            continue
        ref, hyp = load_clip(seg, lead)
        hyp_tokens = [w for w, _ in hyp]
        clip_start = seg["start_s"] - lead
        items = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp_tokens, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            t = hyp[j1][1] - clip_start if j1 < len(hyp) else (hyp[-1][1] - clip_start if hyp else 0.0)
            items.append({"n": len(items) + 1, "clip_time_s": round(t, 1),
                          "kind": {"insert": "machine only", "delete": "reference only", "replace": "differs"}[tag],
                          "reference": " ".join(ref[i1:i2]), "machine": " ".join(hyp_tokens[j1:j2]),
                          "span": [i1, i2, j1, j2], "heard": ""})
        clips.append({"id": seg["id"], "clip": seg["clip"], "recording": seg["recording"],
                      "first_pass_reference_tokens": ref, "machine_tokens": hyp_tokens, "items": items})
        print(f"  {seg['id']}  {seg['clip']:<48s} {len(items)} disagreements")
    SHEET.write_text(json.dumps({
        "seed": SEED, "clips_chosen": chosen,
        "how_to_fill": "For each item, listen to the clip at clip_time_s and set heard to M if the machine's words "
                       "were said, R if the reference's words were said, or type the words actually said. "
                       "Leave heard empty to keep the reference as typed. 'machine only' means the machine has "
                       "words the reference does not; 'reference only' the reverse.",
        "clips": clips}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{sum(len(c['items']) for c in clips)} items over {len(clips)} clips written to {SHEET}")
    return 0


def apply_decisions(ref: list[str], items: list[dict]) -> tuple[list[str], int, int]:
    """The adjudicated reference: replacements applied from the end so spans stay valid."""
    out = list(ref)
    decided = 0
    for item in sorted(items, key=lambda it: it["span"][0], reverse=True):
        heard = (item.get("heard") or "").strip()
        if not heard:
            continue
        decided += 1
        i1, i2, _, _ = item["span"]
        if heard.upper() == "R":
            continue
        if heard.upper() == "M":
            new = item["machine"].split()
        else:
            new = normalise_extended(normalise(heard))
        out[i1:i2] = new
    return out, decided, len(items)


def score() -> int:
    sheet = json.loads(SHEET.read_text(encoding="utf-8"))
    rows = []
    ref_words_first = ref_words_adj = err_first = err_adj = sd_adj = 0
    decided_total = items_total = 0
    for clip in sheet["clips"]:
        ref = clip["first_pass_reference_tokens"]
        hyp = clip["machine_tokens"]
        adjudicated, decided, n_items = apply_decisions(ref, clip["items"])
        s1, d1, i1 = edit_counts(ref, hyp)
        s2, d2, i2 = edit_counts(adjudicated, hyp)
        rows.append({"id": clip["id"], "items": n_items, "decided": decided,
                     "reference_words_first_pass": len(ref), "reference_words_adjudicated": len(adjudicated),
                     "wer_first_pass": round((s1 + d1 + i1) / len(ref), 4) if ref else None,
                     "wer_adjudicated": round((s2 + d2 + i2) / len(adjudicated), 4) if adjudicated else None,
                     "errors_adjudicated": [s2, d2, i2]})
        ref_words_first += len(ref); ref_words_adj += len(adjudicated)
        err_first += s1 + d1 + i1; err_adj += s2 + d2 + i2; sd_adj += s2 + d2
        decided_total += decided; items_total += n_items
    summary = {
        "clips": len(rows), "seed": sheet["seed"], "items": items_total, "decided": decided_total,
        "wer_first_pass_same_clips": round(err_first / ref_words_first, 4) if ref_words_first else None,
        "wer_adjudicated": round(err_adj / ref_words_adj, 4) if ref_words_adj else None,
        "wer_adjudicated_substitutions_deletions": round(sd_adj / ref_words_adj, 4) if ref_words_adj else None,
        "normalisation": "protocol normalisation plus number words as digits and contractions written out, both sides",
        "T1_on_adjudicated_sample": (err_adj / ref_words_adj <= 0.15) if ref_words_adj else None,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps({"experiment": "E22 second pass", "summary": summary, "clips": rows},
                                 indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for r in rows:
        print(f"  {r['id']}  decided {r['decided']}/{r['items']}  WER first pass {r['wer_first_pass']*100:5.1f}%  "
              f"adjudicated {r['wer_adjudicated']*100:5.1f}%")
    print(f"\n  five clips, {ref_words_adj} reference words after adjudication: WER {summary['wer_adjudicated']*100:.1f}% "
          f"(first pass on the same clips {summary['wer_first_pass_same_clips']*100:.1f}%); "
          f"substitutions and deletions alone {summary['wer_adjudicated_substitutions_deletions']*100:.1f}%; "
          f"T1 on this sample {'met' if summary['T1_on_adjudicated_sample'] else 'not met'}; "
          f"{decided_total} of {items_total} items decided")
    print(f"written {RESULT}")
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("make", "score"):
        print(__doc__)
        return 2
    return make() if sys.argv[1] == "make" else score()


if __name__ == "__main__":
    raise SystemExit(main())
