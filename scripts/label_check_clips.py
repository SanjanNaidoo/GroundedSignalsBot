"""Cut ten labelled lecture windows into clips, so the labels can be checked by ear.

    python scripts/label_check_clips.py

protocols/lectures.md says the relevant-window labels are "checked against the audio for
ten of the forty" questions. I labelled from the transcript text; the check asks whether
the recording at that minute really says what the transcript says, so a label built on
a mishearing would be caught. This picks ten questions with seed 20261011 (five from
each half, only questions that have a relevant window), takes each one's
highest-scored window (earliest if tied), cuts that window's audio from the recording
as a mono 16-bit WAV, and writes `lectures/label-check.json` with the question, the
window, its transcript text and a `confirmed` field to fill in (true or false, with a
note). The clips come from the convener's recordings and are not committed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.lectures import load_windows  # noqa: E402
from e22_clips import cut, find_recording  # noqa: E402

BANK = ROOT / "stress" / "bank" / "lectures"
OUT = ROOT / "lectures" / "label-check-clips"
LIST = ROOT / "lectures" / "label-check.json"
SEED = 20261011
PER_HALF = 5


def main() -> int:
    if LIST.exists():
        print(f"{LIST} exists; not overwriting a check that may be in progress")
        return 1
    windows = {w.window_id: w for w in load_windows(ROOT / "lectures" / "transcripts")}
    rng = random.Random(SEED)
    picked = []
    for split in ("dev", "final"):
        labels = json.loads((BANK / f"labels-{split}.json").read_text(encoding="utf-8"))
        candidates = sorted(q for q, s in labels.items() if q != "_method" and any(v >= 2 for v in s.values()))
        for q in rng.sample(candidates, PER_HALF):
            scores = labels[q]
            best = max(scores.values())
            wid = sorted(w for w, s in scores.items() if s == best)[0]
            picked.append((split, q, wid, best))
    questions = {}
    for split in ("dev", "final"):
        for line in (BANK / f"lectures-{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                questions[row["id"]] = row["question"]
    out_rows = []
    for split, q, wid, score in picked:
        w = windows[wid]
        recording = find_recording(w.recording)
        mm, ss = divmod(int(w.start), 60)
        clip = OUT / f"{q}_{Path(w.recording).stem.replace(' ', '_')}_{mm:02d}m{ss:02d}s.wav"
        rate, seconds = cut(recording, w.start, w.end, clip)
        out_rows.append({"split": split, "id": q, "question": questions[q], "window": wid,
                         "reference": w.reference(), "label_score": score, "clip": clip.name,
                         "transcript_text": w.text, "confirmed": None, "note": ""})
        print(f"  {q}  {w.reference():<48s} score {score}  -> {clip.name} ({seconds:.0f} s)")
    LIST.write_text(json.dumps({"seed": SEED, "note": "listen to each clip with its transcript text; confirmed=true "
                                "if the recording says what the text says and the label stands, false otherwise, "
                                "with a note", "checks": out_rows}, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"{len(out_rows)} clips in {OUT}; fill in {LIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
