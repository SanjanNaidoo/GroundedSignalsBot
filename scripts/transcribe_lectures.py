"""Transcribe the lecture recordings locally, with timestamps, one JSON per recording.

    python scripts/transcribe_lectures.py                       # everything not yet done
    python scripts/transcribe_lectures.py --only optimal_policy  # recordings whose name contains this
    python scripts/transcribe_lectures.py --limit 3             # the first three not yet done

Why local, and why this engine: the recordings are the convener's and have not been
cleared to leave the machine, the card is otherwise idle, and faster-whisper reads the
audio straight out of the video container, so nothing is extracted to disk. The CUDA
libraries it needs are the ones PyTorch already ships, which is why torch is imported
first and its library directory put on the search path.

What is written, under lectures/transcripts/:

  <recording>.json   every segment with start, end and text, every word with its own
                     timestamp and probability, and the settings that produced it
  <recording>.txt    the same transcript as "[mm:ss] text" lines, for reading
  manifest.jsonl     one line per recording: duration, wall time, speed, model, settings

A recording whose JSON says it is complete is skipped, so the run resumes after an
interruption. The recordings themselves are never written to, and the transcripts carry
the recording's file name and series so that a lecture reference can always be traced
back to the exact file and second it came from.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ctranslate2 looks for cuDNN and cuBLAS on the library path; torch carries both.
import torch  # noqa: E402

_TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(_TORCH_LIB):
    os.add_dll_directory(_TORCH_LIB)
    os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")

import av  # noqa: E402
import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".m4v"}
DSP_NAME = re.compile(r"^L\d{2}[a-z]$")
SAMPLE_RATE = 16000


def decode_audio(path: Path) -> np.ndarray:
    """The recording's audio as 16 kHz mono float32, decoded straight from the container.

    faster-whisper has its own decoder, but it calls the audio library with an argument
    the library's current release has removed, so the decoding is done here against the
    release that is installed. Nothing is written to disk; the samples go to the model
    as an array.
    """
    pieces = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                pieces.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # whatever the resampler still holds
            pieces.append(out.to_ndarray().reshape(-1))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32) / 32768.0


def series_of(stem: str) -> str:
    return "dsp" if DSP_NAME.match(stem) else "ml"


def find_recordings(folder: Path) -> list[Path]:
    """Every recording under the folder, one per file name, the top-level copy preferred."""
    seen: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
            seen.setdefault(path.name, path)
    return [seen[name] for name in sorted(seen)]


def stamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcribe(model: WhisperModel, path: Path, out_dir: Path, args) -> dict:
    started = time.monotonic()
    audio = decode_audio(path)
    decode_s = time.monotonic() - started
    segments_iter, info = model.transcribe(
        audio, language=args.language, beam_size=args.beam_size,
        word_timestamps=True, vad_filter=not args.no_vad,
        condition_on_previous_text=False,
    )
    segments = []
    for seg in segments_iter:
        segments.append({
            "id": seg.id, "start": round(seg.start, 2), "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 3),
            "no_speech_prob": round(seg.no_speech_prob, 3),
            "words": [{"start": round(w.start, 2), "end": round(w.end, 2),
                       "word": w.word, "probability": round(w.probability, 3)}
                      for w in (seg.words or [])],
        })
    wall = time.monotonic() - started
    record = {
        "recording": path.name,
        "stem": path.stem,
        "series": series_of(path.stem),
        "source_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "duration_s": round(info.duration, 2),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "model": args.model,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "vad_filter": not args.no_vad,
        "word_timestamps": True,
        "transcribed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_s": round(wall, 1),
        "decode_s": round(decode_s, 1),
        "realtime_factor": round(info.duration / wall, 1) if wall else None,
        "segments": segments,
        "complete": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{path.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                               encoding="utf-8", newline="\n")
    (out_dir / f"{path.stem}.txt").write_text(
        "\n".join(f"[{stamp(s['start'])}] {s['text']}" for s in segments) + "\n",
        encoding="utf-8", newline="\n")
    summary = {k: record[k] for k in ("recording", "series", "duration_s", "wall_s", "decode_s",
                                      "realtime_factor", "model", "compute_type", "beam_size",
                                      "vad_filter", "language", "language_probability",
                                      "transcribed_at")}
    summary["segments"] = len(segments)
    summary["words"] = sum(len(s["words"]) for s in segments)
    with (out_dir / "manifest.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", default="Lecture Videos")
    parser.add_argument("--out", default="lectures/transcripts")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true",
                        help="do not drop silences first; slower, and the timestamps "
                             "then include the pauses")
    parser.add_argument("--only", default="", help="only recordings whose name contains this")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many")
    parser.add_argument("--redo", action="store_true", help="transcribe again even if done")
    args = parser.parse_args()

    videos = (ROOT / args.videos) if not Path(args.videos).is_absolute() else Path(args.videos)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    recordings = find_recordings(videos)
    if args.only:
        recordings = [p for p in recordings if args.only.lower() in p.name.lower()]
    todo = []
    for p in recordings:
        done = out_dir / f"{p.stem}.json"
        if not args.redo and done.exists():
            try:
                if json.loads(done.read_text(encoding="utf-8")).get("complete"):
                    continue
            except json.JSONDecodeError:
                pass
        todo.append(p)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(recordings)} recordings found, {len(todo)} to transcribe, "
          f"model {args.model} ({args.compute_type}) on {args.device}")
    if not todo:
        return 0

    loaded = time.monotonic()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print(f"model loaded in {time.monotonic() - loaded:.0f} s\n")

    total_audio = total_wall = 0.0
    for n, path in enumerate(todo, start=1):
        try:
            s = transcribe(model, path, out_dir, args)
        except Exception as exc:  # one bad file must not stop the other 76
            print(f"[{n}/{len(todo)}] FAILED {path.name}: {exc}", flush=True)
            continue
        total_audio += s["duration_s"]
        total_wall += s["wall_s"]
        print(f"[{n}/{len(todo)}] {path.name}: {s['duration_s'] / 60:.1f} min in "
              f"{s['wall_s']:.0f} s ({s['realtime_factor']}x), {s['segments']} segments, "
              f"{s['words']} words, lang {s['language']} p={s['language_probability']}",
              flush=True)
    if total_wall:
        print(f"\ndone: {total_audio / 3600:.1f} h of audio in {total_wall / 60:.0f} min "
              f"({total_audio / total_wall:.1f}x real time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
