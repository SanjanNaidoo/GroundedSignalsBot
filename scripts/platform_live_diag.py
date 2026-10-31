"""Why the local answer in E30 took 557 s: the local model timed cold, warm, and after the reader.

In E30 (16 October) L02 went to the local model after three screenshots had been read,
and took 153 s to its first word and 557 s in all, 1,029 tokens: about 2.5 tokens a
second, where interface check 2 on 14 October saw about 32. The reader stays on the card
after a screenshot by design (ocr.free_vlm). This puts the same question to the same
served platform in a known order and records the card's memory beside each answer, so
the cause is measured rather than guessed. Nothing hosted is asked; nothing is spent.

    .venv\\Scripts\\python.exe scripts\\platform_live_diag.py                      (before the fix)
    .venv\\Scripts\\python.exe scripts\\platform_live_diag.py --out diag-after-fix  (after it)

After the fix (the server puts the reader down before a local answer and releases what
each answer reserved, under the batch runner's cap), steps 6 and 7 read a second
screenshot after a local answer and ask the local model once more.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import platform_live as pl  # noqa: E402

OUT = pl.OUT / "run-2026-10-16" / "diag-local-latency"
QUESTION = json.loads(pl.TASKS.read_text(encoding="utf-8").splitlines()[-1])["question"]   # L02
PORT = 8110


def card() -> dict:
    """Dedicated and shared (system RAM) GPU memory, as Windows counts them, in MiB."""
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    ps = ("$c = Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage','\\GPU Adapter Memory(*)\\Shared Usage';"
          "$c.CounterSamples | ForEach-Object { $_.Path + '=' + [math]::Round($_.CookedValue/1MB) }")
    counters = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    # One entry per adapter (the integrated one too); the card is the adapter with the
    # most dedicated memory in use, and its shared figure is the one that matters.
    by: dict[str, dict] = {}
    for line in counters.splitlines():
        if "=" not in line or "(" not in line:
            continue
        path, value = line.rsplit("=", 1)
        adapter = path.split("(", 1)[1].split(")", 1)[0]
        kind = "shared" if "shared usage" in path.lower() else "dedicated"
        by.setdefault(adapter, {})[kind] = int(value)
    gpu = max(by.values(), key=lambda v: v.get("dedicated", 0), default={})
    return {"t": round(time.monotonic(), 1), "nvidia_used": int(used or 0),
            "dedicated": gpu.get("dedicated", 0), "shared": gpu.get("shared", 0)}


def main() -> int:
    global OUT
    if "--out" in sys.argv:
        OUT = OUT.parent / sys.argv[sys.argv.index("--out") + 1]
    after_fix = OUT.name != "diag-local-latency"
    OUT.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            try:
                samples.append(card())
            except Exception:
                pass
            stop.wait(4)

    steps = []

    def local(label: str) -> None:
        before = card()
        rec = pl.ask(PORT, QUESTION, model=pl.LOCAL)
        after = card()
        f = rec["final"]
        gen_s = (rec.get("total_s") or 0) - (rec.get("first_s") or 0)
        steps.append({"step": label, "first_s": rec.get("first_s"), "total_s": rec.get("total_s"),
                      "pieces": rec.get("pieces"), "output_tokens": f.get("output_tokens"),
                      "pieces_per_s": round((rec.get("pieces") or 0) / gen_s, 1) if gen_s > 0 else None,
                      "right": pl.rule_holds({"label": "300 Hz", "kind": "value", "value": "300", "unit": "hz"},
                                             f.get("text") or "", f),
                      "error": f.get("error"), "card_before": before, "card_after": after})
        print(json.dumps(steps[-1]), flush=True)

    with pl.Server("diag", PORT, OUT, extra=("--max-usd", "0.01")):
        threading.Thread(target=sample, daemon=True).start()
        steps.append({"step": "server up", "card": card()})
        local("1 local, cold (reader not loaded)")
        local("2 local, warm")
        image = pl.OUT / "run-2026-10-16" / "S01.png"
        t0 = time.monotonic()
        status, read = pl.post(PORT, "/read", {"image": "data:image/png;base64,"
                                               + base64.b64encode(image.read_bytes()).decode("ascii")})
        steps.append({"step": "3 screenshot read (reader now on the card)", "status": status,
                      "seconds": round(time.monotonic() - t0, 1), "card": card()})
        print(json.dumps(steps[-1]), flush=True)
        local("4 local, after the reader")
        local("5 local, after the reader, again")
        if after_fix:
            t0 = time.monotonic()
            status, read = pl.post(PORT, "/read", {"image": "data:image/png;base64,"
                                                   + base64.b64encode(image.read_bytes()).decode("ascii")})
            steps.append({"step": "6 screenshot read after a local answer (the reader loads again)",
                          "status": status, "seconds": round(time.monotonic() - t0, 1),
                          "text_kept": "u[n + 2]" in (read.get("text") or ""), "card": card()})
            print(json.dumps(steps[-1]), flush=True)
            local("7 local, after the second screenshot")
        stop.set()
    (OUT / "steps.json").write_text(json.dumps({"question": QUESTION, "steps": steps, "samples": samples},
                                               indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"written {OUT / 'steps.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
