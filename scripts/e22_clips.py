"""Cut the twenty E22 segments out of the recordings as short audio clips.

    python scripts/e22_clips.py

protocols/lectures.md asks me to transcribe twenty thirty-second segments by hand, from
the audio, and to read off the time at which one distinctive phrase in each begins. Doing
that by scrubbing twenty videos to the right second is slow and error-prone; a folder of
twenty clips, one per segment, named by segment and recording, is not. Each clip is the
segment's thirty seconds plus two seconds either side so a word cut in half at the
boundary can still be heard, decoded with PyAV from the recording's own audio and written
as 16-bit mono WAV at the recording's sample rate.

The clips are derived from the convener's recordings and are not committed; the
segment list `lectures/e22-segments.json` gains, for each segment, the clip's file name
and a `phrase_start_in_clip_s` field, so the phrase time can be read off the clip player
and converted to recording time by scripts/e22_score.py (start_s - lead + clip time).
The reference text itself is typed into `reference_text`, nothing pre-filled: a machine
transcript shown beside the audio would steer the ear, and the test is of the machine.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "lectures" / "e22-segments.json"
CLIPS = ROOT / "lectures" / "e22-clips"
VIDEOS = ROOT / "Lecture Videos"
LEAD_S = 2.0

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_recording(name: str) -> Path:
    matches = [p for p in VIDEOS.rglob(name) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"{name} not found under {VIDEOS}")
    return matches[0]


def cut(recording: Path, start_s: float, end_s: float, out: Path) -> tuple[int, float]:
    """Decode [start_s, end_s] of the recording's audio to a mono 16-bit WAV."""
    import av
    import numpy as np

    with av.open(str(recording)) as container:
        stream = container.streams.audio[0]
        rate = stream.rate or 48000
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        # seek a little before the start; decoded frames carry their own times
        container.seek(int(max(0.0, start_s - 1.0) / stream.time_base), stream=stream, backward=True)
        pieces = []
        for frame in container.decode(stream):
            t0 = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if t0 is None:
                continue
            if t0 >= end_s:
                break
            for out_frame in resampler.resample(frame):
                samples = out_frame.to_ndarray().reshape(-1)
                n = samples.shape[0]
                f0 = float(out_frame.pts * out_frame.time_base) if out_frame.pts is not None else t0
                times = f0 + np.arange(n) / rate
                keep = (times >= start_s) & (times < end_s)
                if keep.any():
                    pieces.append(samples[keep])
        for out_frame in resampler.resample(None):  # flush
            pieces.append(out_frame.to_ndarray().reshape(-1))
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(audio.astype(np.int16).tobytes())
    return rate, audio.shape[0] / rate


def main() -> int:
    data = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    for seg in data["segments"]:
        recording = find_recording(seg["recording"])
        stem = Path(seg["recording"]).stem.replace(" ", "_")
        mm, ss = divmod(int(seg["start_s"]), 60)
        out = CLIPS / f"{seg['id']}_{stem}_{mm:02d}m{ss:02d}s.wav"
        rate, seconds = cut(recording, seg["start_s"] - LEAD_S, seg["end_s"] + LEAD_S, out)
        seg["clip"] = out.name
        seg["clip_lead_s"] = LEAD_S
        seg.setdefault("phrase_start_in_clip_s", None)
        print(f"  {seg['id']}  {seg['recording']:<45s} {seg['start']}  -> {out.name}  ({seconds:.1f} s at {rate} Hz)")
    data["clips"] = {"folder": "lectures/e22-clips", "lead_s": LEAD_S,
                     "note": "each clip is the segment plus lead_s either side; phrase_start_in_clip_s is read "
                             "off the clip player and converted to recording time as start_s - lead_s + that"}
    SEGMENTS.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(data['segments'])} clips in {CLIPS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
