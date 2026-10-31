"""E30: the platform exactly as served, with Opus 5 live (protocols/platform-live.md).

Starts `ym07 serve --platform` itself, unchanged but for the port and the session ceiling,
and speaks to it over HTTP the way the page does: typed questions to /stream, a follow-up
with its history, screenshots to /read and their transcription on to /stream, the model
switch, every link a reply carries, and three failures of the hosted model (the spend
ceiling, a rejected key, no internet), each on its own server. The interface checks of
12-14 October ran Opus 5 as a stub; this is the run that does not.

    .venv\\Scripts\\python.exe scripts\\platform_live.py --dry-run   (stub models, nothing spent)
    .venv\\Scripts\\python.exe scripts\\platform_live.py             (the platform as served)

Every reply, the images, each server's log, summary.json and report.md go to
stress/results/platform-live/<run>/.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07 import remark                              # noqa: E402
from ym07.checks import normalise, value_pattern      # noqa: E402
from ym07.hybrid import held_back                     # noqa: E402
from ym07.web import MAX_QUESTION_CHARS               # noqa: E402

TASKS = ROOT / "stress" / "bank" / "platform-live" / "tasks.jsonl"
OUT = ROOT / "stress" / "results" / "platform-live"
YM07 = ROOT / ".venv" / "Scripts" / "ym07.exe"
DEFAULT, LOCAL = "A3-opus-pipeline", "A-r1-facts-marks"
MAX_USD = 2.00
SPEND_LIMIT = 2.10
REQUEST_TIMEOUT = 900
FAIL_QUESTION = "What is the Nyquist rate of a signal band-limited to 4 kHz?"


# --- the server, as a student's machine runs it ---------------------------------------------

class Server:
    """`ym07 serve --platform` in a child process, with its console kept in a log."""

    def __init__(self, name: str, port: int, out: Path, *, extra=(), env=None, dry_run=False):
        self.name, self.port, self.out = name, port, out
        self.cmd = [str(YM07), "serve", "--platform", "--port", str(port), *extra]
        if dry_run:
            self.cmd.append("--dry-run")
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
        self.log_path = out / f"server-{name}.log"

    def __enter__(self) -> "Server":
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.log.write("$ " + " ".join(self.cmd[1:]) + "\n")
        self.log.flush()
        started = time.monotonic()
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=self.env)
        while time.monotonic() - started < 1200:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {self.name} exited during start; see {self.log_path}")
            try:
                status, _ = get(self.port, "/state", timeout=5)
                if status == 200:
                    self.start_s = time.monotonic() - started
                    print(f"server {self.name} up on {self.port} after {self.start_s:.0f}s")
                    return self
            except OSError:
                pass
            time.sleep(5)
        raise RuntimeError(f"server {self.name} did not answer /state within 20 minutes")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def __exit__(self, *exc) -> None:
        # The launcher starts the interpreter as a child, so the whole tree goes, or the
        # model would stay on the card after the launcher had gone.
        subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                       capture_output=True)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        self.log.close()
        time.sleep(5)


def get(port: int, path: str, *, timeout: float = 30, headers=None) -> tuple[int, bytes]:
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def post(port: int, path: str, payload: dict, *, timeout: float = REQUEST_TIMEOUT) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    conn.close()
    try:
        return resp.status, json.loads(body)
    except json.JSONDecodeError:
        return resp.status, {"raw": body}


def ask(port: int, question: str, *, model=None, history=None, from_image="") -> dict:
    """One question as the page sends it, with when each piece of the reply arrived."""
    payload = {"question": question, "history": history or [], "grounded": True}
    if model:
        payload["model"] = model
    if from_image:
        payload["from_image"] = from_image
    started = time.monotonic()
    record = {"question": question, "model_asked": model or DEFAULT, "history_turns": len(history or [])}
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT)
        conn.request("POST", "/stream", json.dumps(payload), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        record["status"] = resp.status
        if "event-stream" not in (resp.getheader("Content-Type") or ""):
            body = resp.read().decode("utf-8", "replace")
            conn.close()
            record["final"] = json.loads(body) if body.startswith("{") else {"raw": body}
            record["total_s"] = round(time.monotonic() - started, 2)
            return record
        arrivals, restarts, final, buffer = [], [], {}, ""
        while True:
            chunk = resp.read(256)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", "replace")
            while "\n\n" in buffer:
                line, buffer = buffer.split("\n\n", 1)
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                if "delta" in event:
                    arrivals.append(time.monotonic() - started)
                elif "restart" in event:
                    restarts.append(event["restart"])
                else:
                    final = event
        conn.close()
    except (OSError, TimeoutError) as exc:
        record.update(status=None, final={"error": f"client: {exc}"},
                      total_s=round(time.monotonic() - started, 2), client_failure=True)
        return record
    record.update(final=final, restarts=restarts, pieces=len(arrivals),
                  first_s=round(arrivals[0], 2) if arrivals else None,
                  total_s=round(time.monotonic() - started, 2))
    return record


# --- marking -------------------------------------------------------------------------------

def rule_holds(rule: dict, text: str, final: dict) -> bool:
    kind = rule["kind"]
    if kind == "any":
        return any(rule_holds(r, text, final) for r in rule["rules"])
    if kind == "held_back":
        return held_back(text)
    if kind == "lookup":
        return bool(final.get("handout_lookup"))
    if kind == "cites":
        return bool(re.search(r"\[S\d+\]", text)) and bool(final.get("sources"))
    if kind == "value":
        unit = rule.get("unit") or ""
        tail = r"\s*\\?%" if unit == "%" else (r"\s*" + re.escape(unit) if unit else "")
        rule = {"label": rule["label"], "kind": "pattern", "pattern": value_pattern(rule["value"]) + tail}
    keep = {k: rule[k] for k in ("label", "kind", "value", "tolerance", "pattern", "unless", "options")
            if k in rule}
    return remark.holds(keep, text)


def mark(rules: list[dict], final: dict) -> dict[str, bool]:
    text = final.get("text") or ""
    return {r["label"]: (not final.get("error")) and rule_holds(r, text, final) for r in rules}


def read_screen(groups: list[list[str]], transcription: str) -> dict[str, bool]:
    squeezed = re.sub(r"\s+", "", normalise(remark.typeset_to_text(transcription or "")))
    return {" or ".join(alts): any(re.sub(r"\s+", "", a.lower()) in squeezed for a in alts)
            for alts in groups}


def resources_ok(expect: str | None, final: dict) -> bool | None:
    if expect is None:
        return None
    res = final.get("resources") or {}
    lectures, problems = res.get("lectures") or [], res.get("problems") or []
    if expect == "covered":
        return bool(lectures) and bool(problems)
    if expect == "uncovered":
        return not lectures and not problems
    if expect == "local":
        return not lectures and "Opus 5" in (res.get("lecture_note") or "")
    raise ValueError(expect)


def links_in(final: dict) -> list[str]:
    res = final.get("resources") or {}
    urls = [s.get("doc") for s in final.get("sources") or []]
    urls += [w.get("video") for w in res.get("lectures") or []]
    urls += [p.get("doc") for p in res.get("problems") or []]
    return [u for u in urls if u]


def readable_error(final: dict) -> bool:
    message = final.get("error") or ""
    return bool(message.strip()) and "Traceback" not in message and not final.get("text")


# --- the screenshots -----------------------------------------------------------------------

def render(lines: list[str], path: Path) -> None:
    """As scripts/case_screenshot.py renders its problem: typed notes, a plain font."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("consola.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    line_h = 40
    img = Image.new("RGB", (1180, line_h * (len(lines) + 2)), "white")
    draw = ImageDraw.Draw(img)
    y = line_h // 2
    for line in lines:
        draw.text((36, y), line, fill="black", font=font)
        y += line_h
    img.save(path)


# --- the run -------------------------------------------------------------------------------

def main_server(args, out: Path, tasks: list[dict]) -> dict:
    rows: list[dict] = []
    interface: dict[str, dict] = {}
    crashed = False
    port = 8099
    with Server("main", port, out, extra=("--max-usd", f"{MAX_USD:.2f}"), dry_run=args.dry_run) as srv:
        _, body = get(port, "/state")
        state = json.loads(body)
        offered = [m.get("key") for m in state.get("models") or []]
        interface["I1"] = {"ok": offered == [DEFAULT, LOCAL] and state.get("default") == DEFAULT,
                           "detail": f"offered {offered}, default {state.get('default')}"}
        _, page = get(port, "/")
        page = page.decode("utf-8", "replace")
        interface["I2"] = {"ok": "katex.min.js" in page and "katex.renderToString" in page,
                           "detail": "KaTeX loaded and the renderer wired to it"}
        empty, _ = post(port, "/ask", {"question": "   "})
        long_, _ = post(port, "/stream", {"question": "x" * (MAX_QUESTION_CHARS + 1)})
        interface["I3"] = {"empty": empty, "over_long": long_}

        for task in tasks:
            print(f"{task['id']} ({task['model'] or DEFAULT}) ...", flush=True)
            if task["kind"] == "followup":
                history: list = []
                for n, (turn, rules) in enumerate(zip(task["turns"], task["turn_rules"]), start=1):
                    rec = ask(port, turn, model=task["model"], history=history)
                    rec.update(task=task["id"], turn=n, rules=mark(rules, rec["final"]))
                    rows.append(rec)
                    history += [["Student", turn], ["Tutor", rec["final"].get("text", "")]]
            elif task["kind"] == "screenshot":
                image = out / f"{task['id']}.png"
                render(task["image_lines"], image)
                encoded = base64.b64encode(image.read_bytes()).decode("ascii")
                t0 = time.monotonic()
                status, read = post(port, "/read", {"image": "data:image/png;base64," + encoded})
                read_s = round(time.monotonic() - t0, 2)
                text = read.get("text") or ""
                rec = ask(port, text, model=task["model"], from_image=read.get("engine_name") or "") \
                    if text else {"final": {"error": read.get("error") or "nothing read"}, "status": status}
                rec.update(task=task["id"], turn=1, image=image.name,
                           read={"status": status, "latency_s": read_s, "text": text,
                                 "engine": read.get("engine_name"), "notes": read.get("notes"),
                                 "error": read.get("error")},
                           read_screen=read_screen(task["read_screen"], text),
                           rules=mark(task["rules"], rec["final"]))
                rows.append(rec)
            else:
                rec = ask(port, task["question"], model=task["model"])
                rec.update(task=task["id"], turn=1, rules=mark(task["rules"], rec["final"]))
                rows.append(rec)
            last = rows[-1]
            last["resources_ok"] = resources_ok(task.get("resources"), last["final"])
            if not srv.alive():
                crashed = True
                print(f"server died after {task['id']}")
                break
            verdicts = ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in last["rules"].items())
            print(f"  {last.get('total_s')}s, cost {last['final'].get('cost_usd')}, "
                  f"{'ERROR ' + str(last['final'].get('error')) if last['final'].get('error') else verdicts}")

        # I3's last part: after the two bad requests, the next question was answered.
        first = rows[0]["final"] if rows else {}
        interface["I3"]["ok"] = (interface["I3"]["empty"] == 400 and interface["I3"]["over_long"] == 413
                                 and bool(first.get("text")) and not first.get("error"))
        interface["I3"]["detail"] = (f"empty {interface['I3']['empty']}, over-long "
                                     f"{interface['I3']['over_long']}, next question "
                                     f"{'answered' if interface['I3']['ok'] else 'NOT answered'}")

        opened = []
        if srv.alive():
            for rec in rows:
                for url in links_in(rec["final"]):
                    try:
                        # Encoded as the browser encodes a link before it follows it: the
                        # notes are "EEE4114F DSP Notes.pdf", with spaces.
                        status, _ = get(port, quote(url.split("#")[0], safe="/"),
                                        headers={"Range": "bytes=0-63"})
                    except OSError as exc:
                        status = f"failed: {exc}"
                    opened.append({"task": rec["task"], "url": url, "status": status})
        interface["I4"] = {"ok": bool(opened) and all(o["status"] in (200, 206) for o in opened),
                           "detail": f"{sum(o['status'] in (200, 206) for o in opened)} of "
                                     f"{len(opened)} links opened", "links": opened}
        named = [(r["task"], r["model_asked"], r["final"].get("model")) for r in rows
                 if not r["final"].get("error")]
        interface["I5"] = {"ok": bool(named) and all(a == b for _, a, b in named),
                           "detail": f"{sum(a == b for _, a, b in named)} of {len(named)} replies "
                                     "named the model asked"}
        alive = srv.alive()
    return {"rows": rows, "interface": interface, "crashed": crashed or not alive,
            "start_s": round(srv.start_s, 1)}


def failure(args, out: Path, name: str, port: int, *, extra=(), env=None, opus_first=None,
            limit_s: float) -> dict:
    """One hosted failure on its own server, then the local model on the switch."""
    with Server(name, port, out, extra=extra, env=env) as srv:
        warm = ask(port, opus_first) if opus_first else None
        failed = ask(port, FAIL_QUESTION)
        local = ask(port, FAIL_QUESTION, model=LOCAL)
        alive = srv.alive()
    ok = (readable_error(failed["final"]) and (failed.get("total_s") or 1e9) <= limit_s
          and bool(local["final"].get("text")) and not local["final"].get("error") and alive)
    return {"name": name, "ok": ok, "limit_s": limit_s, "warm": warm, "failed": failed,
            "local": local, "alive": alive}


def summarise(main: dict, fails: list[dict], tasks: list[dict]) -> dict:
    rows = main["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)

    def met(tid: str) -> bool:
        recs = by.get(tid) or []
        return bool(recs) and all(r["rules"] and all(r["rules"].values()) for r in recs)

    spend = sum(r["final"].get("cost_usd") or 0 for r in rows)
    for f in fails:
        for rec in (f.get("warm"), f.get("failed"), f.get("local")):
            if rec:
                spend += rec["final"].get("cost_usd") or 0
    answered = [r for r in rows if r["final"].get("text") and not r["final"].get("error")]
    marks = {
        "PL1": {"ok": not main["crashed"] and len(answered) == len(rows)
                and all(f["alive"] for f in fails),
                "detail": f"{len(answered)} of {len(rows)} requests ended in a finished reply; "
                          f"server {'crashed' if main['crashed'] else 'stayed up'}"},
        "PL2": {"ok": sum(met(t) for t in ("P01", "P02", "P04", "P05", "P06", "P07")) >= 5,
                "detail": f"{sum(met(t) for t in ('P01', 'P02', 'P04', 'P05', 'P06', 'P07'))} of 6"},
        "PL3": {"ok": all(all((by.get(t) or [{}])[0].get("read_screen", {}).values() or [False])
                          for t in ("S01", "S02", "S03")),
                "detail": ", ".join(f"{t} {'kept' if all((by.get(t) or [{}])[0].get('read_screen', {}).values() or [False]) else 'NOT kept'}"
                                    for t in ("S01", "S02", "S03"))},
        "PL4": {"ok": sum(met(t) for t in ("S01", "S02", "S03")) >= 2,
                "detail": f"{sum(met(t) for t in ('S01', 'S02', 'S03'))} of 3"},
        "PL5": {"ok": met("P03") and bool((by.get("P03") or [{}])[0].get("resources_ok"))
                and bool((by.get("P08") or [{}])[0].get("resources_ok")),
                "detail": f"P03 cites {met('P03')}, cards {(by.get('P03') or [{}])[0].get('resources_ok')}; "
                          f"P08 withheld {(by.get('P08') or [{}])[0].get('resources_ok')}"},
        "PL6": {"ok": all(main["interface"][k]["ok"] for k in ("I1", "I2", "I3", "I4", "I5")),
                "detail": "; ".join(f"{k} {'met' if v['ok'] else 'MISSED'}" for k, v in main["interface"].items())},
        "PL7": {"ok": met("L01") and met("L02") and bool((by.get("L02") or [{}])[0].get("resources_ok")),
                "detail": f"L01 {met('L01')}, L02 {met('L02')} (no lecture minutes: "
                          f"{(by.get('L02') or [{}])[0].get('resources_ok')})"},
        "PL8": {"ok": len(fails) == 3 and all(f["ok"] for f in fails),
                "detail": ", ".join(f"{f['name']} {'met' if f['ok'] else 'MISSED'}" for f in fails) or "not run"},
        "PL9": {"ok": spend <= SPEND_LIMIT, "detail": f"${spend:.2f}"},
    }
    return {"marks": marks, "spend_usd": round(spend, 4),
            "tasks_met": {t["id"]: met(t["id"]) for t in tasks}}


def write_report(out: Path, summary: dict, main: dict, fails: list[dict], meta: dict) -> None:
    lines = [f"# E30: the platform as served, live ({meta['run']})", "",
             f"Commit {meta['commit']}{' (tree changed)' if meta['dirty'] else ''}; "
             f"`ym07 serve --platform --max-usd {MAX_USD:.2f}`; main server up after "
             f"{main['start_s']}s; spend ${summary['spend_usd']:.2f}.", "",
             "## Pass marks (protocols/platform-live.md)", "", "| mark | result | |", "|---|---|---|"]
    lines += [f"| {k} | {'met' if v['ok'] else '**missed**'} | {v['detail']} |"
              for k, v in summary["marks"].items()]
    lines += ["", "## Tasks", "", "| task | turn | model | seconds | cost | rules | cards |", "|---|---|---|---|---|---|---|"]
    for r in main["rows"]:
        f = r["final"]
        res = f.get("resources") or {}
        rules = "ERROR: " + str(f.get("error")) if f.get("error") else \
            ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in r["rules"].items()) or "-"
        lines.append(f"| {r['task']} | {r['turn']} | {f.get('model') or r['model_asked']} | {r.get('total_s')} | "
                     f"{f.get('cost_usd')} | {rules} | {len(res.get('lectures') or [])} lecture, "
                     f"{len(res.get('problems') or [])} past |")
    lines += ["", "## Failures", ""]
    for fl in fails:
        lines.append(f"- **{fl['name']}** {'met' if fl['ok'] else 'MISSED'}: hosted reply after "
                     f"{fl['failed'].get('total_s')}s (limit {fl['limit_s']}s): "
                     f"{(fl['failed']['final'].get('error') or fl['failed']['final'].get('text') or '')[:200]!r}; "
                     f"local then answered: {bool(fl['local']['final'].get('text'))}")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="stub models; failures are skipped")
    parser.add_argument("--skip-failures", action="store_true")
    args = parser.parse_args()

    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    run = "dry-run" if args.dry_run else time.strftime("run-%Y-%m-%d")
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs",
                                 "stress/bank/platform-live"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    meta = {"run": run, "commit": commit, "dirty": dirty, "date": time.strftime("%Y-%m-%d %H:%M"),
            "dry_run": args.dry_run, "max_usd": MAX_USD}

    main_result = main_server(args, out, tasks)
    fails: list[dict] = []
    if not (args.dry_run or args.skip_failures):
        fails.append(failure(args, out, "F1-ceiling", 8101, extra=("--max-usd", "0.01"),
                             opus_first="Who teaches this course?", limit_s=10))
        fails.append(failure(args, out, "F2-rejected-key", 8102, extra=("--max-usd", f"{MAX_USD:.2f}"),
                             env={"ANTHROPIC_API_KEY": "sk-ant-invalid-key-for-the-failure-test"},
                             limit_s=120))
        fails.append(failure(args, out, "F3-no-internet", 8103, extra=("--max-usd", f"{MAX_USD:.2f}"),
                             env={"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                                  "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                             limit_s=180))

    summary = summarise(main_result, fails, tasks)
    with open(out / "replies.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for rec in main_result["rows"]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {**meta, **summary, "interface": main_result["interface"], "failures": fails,
         "main_start_s": main_result["start_s"]}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    write_report(out, summary, main_result, fails, meta)
    print()
    for k, v in summary["marks"].items():
        print(f"{k} {'met' if v['ok'] else 'MISSED'}: {v['detail']}")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
