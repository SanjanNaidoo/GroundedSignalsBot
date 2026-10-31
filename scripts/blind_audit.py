"""E31: a blind reading of saved answers, the marking and the feedback (protocols/blind-audit.md).

    .venv\\Scripts\\python.exe scripts\\blind_audit.py draw     (once; commit the plan before reading)
    .venv\\Scripts\\python.exe scripts\\blind_audit.py serve    (the page; resumable)
    .venv\\Scripts\\python.exe scripts\\blind_audit.py score    (after the last reply)

Part A puts calculation answers in front of the reader with the values the answer must
state, including answers both systems passed that nobody has read. Part B puts the
error-finding replies in front of the reader with the mistake the student was given, and
asks what the rules cannot: whether the reply finds the step, says why it is wrong, and
confirms working that is right. Every item shows both systems' replies as A and B; which
is which stays in the plan and in this process, never on the page or in the readings file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

SEED = 20261016
SYSTEMS = ("A-r1-tuned", "A3-opus-pipeline")
PLAN = ROOT / "grades" / "audit-16oct-plan.json"
READINGS = ROOT / "grades" / "audit-16oct.jsonl"
OUT = ROOT / "stress" / "results" / "audit-16oct"
CALC_BANK = ROOT / "stress" / "bank" / "fresh" / "calc-bank.jsonl"
# The held-out error-finding replies answer the `final` bank, not `fresh`: the two share
# item ids, kinds and families but not numbers, and on 16 October the page showed `fresh`
# until Sanjan saw replies working a different question (protocols/blind-audit.md,
# amendment). `check_display` now refuses to serve a bank the replies were not marked on.
ERRORS_BANK = ROOT / "stress" / "bank" / "final" / "errors.jsonl"
HAND = ROOT / "stress" / "results" / "heldout" / "calc-hand-reading.json"
FORWARD_NOTE = ("Either reading of W1 is acceptable - rows as neurons, or its transpose as in "
                "the course notes - provided z1, a1 and the output all agree with the one used.")

CRITERIA = {
    "calc": [("verdict", "The answer", ["right", "wrong", "unclear"])],
    "error": [("locates", "Locates the step where the student went wrong", ["yes", "no"]),
              ("explains", "Explains why that step is wrong", ["yes", "partly", "no"]),
              ("value", "Gives the right final value", ["yes", "no"]),
              ("false_alarm", "Calls something wrong that was right", ["yes", "no"])],
    "control": [("confirms", "Confirms the working is correct", ["yes", "no"]),
                ("invents", "Invents an error that is not there", ["yes", "no"]),
                ("value", "Gives the right final value", ["yes", "no"])],
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _results(run: str, suite: str) -> dict[str, dict[str, dict]]:
    return {s: {r["id"]: r for r in _jsonl(ROOT / "stress" / "results" / run / suite / f"{s}.jsonl")}
            for s in SYSTEMS}


# --- draw ----------------------------------------------------------------------------------

def draw() -> int:
    if PLAN.exists():
        print(f"{PLAN} exists; the sample is drawn once.")
        return 1
    rng = random.Random(SEED)
    bank = {r["id"]: r for r in _jsonl(CALC_BANK)}
    reg, amd = _results("heldout", "calc"), _results("heldout-amended", "calc")
    read_13 = {r["id"] for r in json.loads(HAND.read_text(encoding="utf-8"))["items"]}
    ids = sorted(bank)
    passed = lambda runs, s, i: runs[s][i]["score"] == 1  # noqa: E731

    items: list[dict] = []
    chosen: set[str] = set()

    unread = [i for i in ids if all(passed(reg, s, i) for s in SYSTEMS) and i not in read_13]
    families = sorted({bank[i]["family"] for i in unread})
    for fam in sorted(rng.sample(families, 10)):
        pick = rng.choice([i for i in unread if bank[i]["family"] == fam])
        items.append({"part": "calc", "stratum": "A-unread", "item_id": pick})
        chosen.add(pick)
    for stratum, wanted, test in (("A-flip", 3, lambda s, i: not passed(reg, s, i) and passed(amd, s, i)),
                                  ("A-fail", 2, lambda s, i: not passed(amd, s, i))):
        for s in SYSTEMS:
            pool = [i for i in ids if test(s, i) and i not in chosen]
            for pick in rng.sample(pool, wanted):
                items.append({"part": "calc", "stratum": f"{stratum} ({s})", "item_id": pick})
                chosen.add(pick)

    errors = _jsonl(ERRORS_BANK)
    fams = sorted({e["family"] for e in errors})
    planted = {f: sorted(e["id"] for e in errors if e["family"] == f and e["kind"] == "error") for f in fams}
    controls = {f: sorted(e["id"] for e in errors if e["family"] == f and e["kind"] == "control") for f in fams}
    picks = [rng.choice(planted[f]) for f in fams]
    for f in sorted(rng.sample(fams, 3)):
        picks.append(rng.choice([i for i in planted[f] if i not in picks]))
    items += [{"part": "error", "stratum": "B-planted", "item_id": i} for i in picks]
    items += [{"part": "control", "stratum": "B-control", "item_id": rng.choice(controls[f])}
              for f in sorted(rng.sample(fams, 6))]

    part_a = [it for it in items if it["part"] == "calc"]
    part_b = [it for it in items if it["part"] != "calc"]
    rng.shuffle(part_a)
    rng.shuffle(part_b)
    token = 0
    for n, it in enumerate(part_a + part_b, start=1):
        order = list(SYSTEMS)
        rng.shuffle(order)
        it["n"] = n
        it["replies"] = []
        for label, system in zip("AB", order):
            token += 1
            it["replies"].append({"label": label, "token": f"r{token:03d}", "system": system})
    plan = {"seed": SEED, "drawn": time.strftime("%Y-%m-%d %H:%M"),
            "protocol": "protocols/blind-audit.md", "items": part_a + part_b}
    PLAN.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(part_a)} calculation items and {len(part_b)} error-finding items, "
          f"{token} replies, written to {PLAN}")
    return 0


# --- the display matches the replies -------------------------------------------------------

def check_display() -> list[str]:
    """Every question shown is the one its replies were marked on, or the reason it is not.

    A calculation's rules are named in the saved row (`per_check`), so they must be the
    shown bank's rules for that item. An error-finding row carries the run's verdict on
    the corrected value, so the shown bank's key must give the same verdict on every
    saved reply with a planted mistake.
    """
    from ym07.checks import Check

    problems = []
    calc_bank = {r["id"]: r for r in _jsonl(CALC_BANK)}
    for s, rows in _results("heldout", "calc").items():
        for i, row in rows.items():
            if set(row["per_check"]) != {c["label"] for c in calc_bank[i]["checks"]}:
                problems.append(f"calc {i} ({s}): rules differ from {CALC_BANK.name}")
    errors_bank = {r["id"]: r for r in _jsonl(ERRORS_BANK)}
    for s, rows in _results("heldout", "errors").items():
        for i, row in rows.items():
            item = errors_bank[i]
            if item["kind"] != row["kind"]:
                problems.append(f"errors {i} ({s}): kind differs")
            elif row["kind"] == "error":
                c = item["check"]
                rule = Check(item_id=i, label=c["label"], kind=c["kind"], value=c.get("value"),
                             pattern=c.get("pattern"))
                if rule.holds(row.get("text") or "") != bool(row["corrected"]):
                    problems.append(f"errors {i} ({s}): the shown key does not give the run's verdict")
    return problems


# --- serve ---------------------------------------------------------------------------------

class Session:
    """The plan, what is already read, and the blinded view of the next item."""

    def __init__(self) -> None:
        self.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        self.calc = {r["id"]: r for r in _jsonl(CALC_BANK)}
        self.errors = {r["id"]: r for r in _jsonl(ERRORS_BANK)}
        self.texts = {"calc": _results("heldout", "calc"), "error": _results("heldout", "errors")}
        self.done = {r["token"] for r in _jsonl(READINGS)} if READINGS.exists() else set()
        self.lock = threading.Lock()

    def progress(self) -> dict:
        total = sum(len(it["replies"]) for it in self.plan["items"])
        return {"read": len(self.done), "total": total}

    def next(self) -> dict:
        for it in self.plan["items"]:
            if all(r["token"] in self.done for r in it["replies"]):
                continue
            suite = "calc" if it["part"] == "calc" else "error"
            if suite == "calc":
                bank = self.calc[it["item_id"]]
                shown = {"question": bank["question"],
                         "must_state": [c["label"] for c in bank["checks"]],
                         "note": FORWARD_NOTE if bank["family"] == "forward_pass" else ""}
            else:
                bank = self.errors[it["item_id"]]
                shown = {"question": bank["message"],
                         "mistake": bank["mistake"] if it["part"] == "error" else "",
                         "must_state": [bank["check"]["label"]],
                         "note": ""}
            replies = [{"label": r["label"], "token": r["token"],
                        "text": self.texts[suite][r["system"]][it["item_id"]].get("text") or "(empty reply)"}
                       for r in it["replies"]]
            return {"done": False, "progress": self.progress(), "n": it["n"],
                    "of": len(self.plan["items"]), "part": it["part"], "shown": shown,
                    "criteria": CRITERIA[it["part"]], "replies": replies}
        return {"done": True, "progress": self.progress()}

    def record(self, payload: dict) -> tuple[int, dict]:
        item = next((it for it in self.plan["items"] if it["n"] == payload.get("n")), None)
        if item is None:
            return 400, {"error": "Unknown item; reload the page."}
        wanted = CRITERIA[item["part"]]
        marks = payload.get("marks") or {}
        lines = []
        for r in item["replies"]:
            if r["token"] in self.done:
                return 409, {"error": "This item was already saved; reload the page."}
            given = marks.get(r["label"]) or {}
            for key, _, options in wanted:
                if given.get(key) not in options:
                    return 400, {"error": f"Reply {r['label']}: answer every question."}
            lines.append({"token": r["token"], "n": item["n"], "item_id": item["item_id"],
                          "part": item["part"], "label": r["label"],
                          **{key: given[key] for key, _, _ in wanted},
                          "note": str((payload.get("notes") or {}).get(r["label"]) or "").strip(),
                          "saved": time.strftime("%Y-%m-%d %H:%M:%S")})
        READINGS.parent.mkdir(parents=True, exist_ok=True)
        with open(READINGS, "a", encoding="utf-8", newline="\n") as fh:
            for line in lines:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.done.update(r["token"] for r in item["replies"])
        return 200, {"saved": True, "next": self.next()}


def serve(port: int, open_browser: bool) -> int:
    problems = check_display()
    if problems:
        print(f"Not serving: {len(problems)} shown question(s) differ from what the replies answered, "
              f"e.g. {problems[0]}")
        return 1
    session = Session()
    page = PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A002
            pass

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/next":
                with session.lock:
                    self._json(200, session.next())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/save":
                self._json(404, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "The reading could not be read."})
                return
            with session.lock:
                status, body = session.record(payload)
            self._json(status, body)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    p = session.progress()
    print(f"Blind reading at {url}: {p['read']} of {p['total']} replies read, saving to {READINGS}")
    print("Every item is saved as you go. Ctrl-C here to stop; run again to resume.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {session.progress()['read']} of {p['total']} read.")
    finally:
        httpd.server_close()
    return 0


# --- score ---------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def score() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    readings = {r["token"]: r for r in _jsonl(READINGS)}
    replies = [(it, r) for it in plan["items"] for r in it["replies"]]
    missing = [r["token"] for _, r in replies if r["token"] not in readings]
    if missing:
        print(f"{len(missing)} of {len(replies)} replies not yet read; scoring needs all of them.")
        return 1
    reg, amd = _results("heldout", "calc"), _results("heldout-amended", "calc")
    errs = _results("heldout-amended", "errors")
    hand = {r["id"]: r for r in json.loads(HAND.read_text(encoding="utf-8"))["items"]}

    rows = []
    for it, r in replies:
        got = readings[r["token"]]
        row = {"n": it["n"], "item_id": it["item_id"], "part": it["part"], "stratum": it["stratum"],
               "system": r["system"], "reading": {k: v for k, v in got.items()
                                                  if k not in ("token", "n", "item_id", "part", "label", "saved")}}
        if it["part"] == "calc":
            row["registered_right"] = reg[r["system"]][it["item_id"]]["score"] == 1
            row["amended_right"] = amd[r["system"]][it["item_id"]]["score"] == 1
            row["read_right"] = got["verdict"] == "right"
            if it["item_id"] in hand:
                row["read_13oct_right"] = bool(hand[it["item_id"]][r["system"]]["right"])
        else:
            rec = errs[r["system"]][it["item_id"]]
            row["rule"] = {k: bool(rec.get(k)) for k in ("corrected", "claims_error", "confirms")}
        rows.append(row)

    calc = [x for x in rows if x["part"] == "calc"]
    unread = [x for x in calc if x["stratum"] == "A-unread"]
    earlier = [x for x in calc if "read_13oct_right" in x]
    ba1 = sum(x["amended_right"] and not x["read_right"] for x in unread)
    ba2 = sum(x["amended_right"] == x["read_right"] for x in calc)
    ba3 = sum(x["read_13oct_right"] == x["read_right"] for x in earlier)
    false_pass = [x for x in calc if x["amended_right"] and not x["read_right"]]
    rule_passes = sum(x["amended_right"] for x in calc)

    per_system = {}
    for s in SYSTEMS:
        planted = [x for x in rows if x["part"] == "error" and x["system"] == s]
        ctrl = [x for x in rows if x["part"] == "control" and x["system"] == s]
        rd = lambda x, k: x["reading"][k]  # noqa: E731
        full = sum(rd(x, "locates") == "yes" and rd(x, "explains") == "yes" and rd(x, "value") == "yes"
                   for x in planted)
        good_ctrl = sum(rd(x, "confirms") == "yes" and rd(x, "invents") == "no" for x in ctrl)
        per_system[s] = {
            "planted": len(planted), "controls": len(ctrl),
            "locates": sum(rd(x, "locates") == "yes" for x in planted),
            "explains_yes": sum(rd(x, "explains") == "yes" for x in planted),
            "explains_partly": sum(rd(x, "explains") == "partly" for x in planted),
            "value": sum(rd(x, "value") == "yes" for x in planted),
            "false_alarm": sum(rd(x, "false_alarm") == "yes" for x in planted),
            "full_diagnosis": full, "full_diagnosis_wilson": wilson(full, len(planted)),
            "confirms": sum(rd(x, "confirms") == "yes" for x in ctrl),
            "invents": sum(rd(x, "invents") == "yes" for x in ctrl),
            "controls_value": sum(rd(x, "value") == "yes" for x in ctrl),
            "controls_good": good_ctrl,
            "rule_vs_reading": {
                "corrected_vs_value": sum(x["rule"]["corrected"] == (rd(x, "value") == "yes") for x in planted),
                "claims_error_vs_locates": sum(x["rule"]["claims_error"] == (rd(x, "locates") == "yes")
                                               for x in planted),
                "confirms_vs_confirms": sum(x["rule"]["confirms"] == (rd(x, "confirms") == "yes") for x in ctrl),
            },
        }
    marks = {
        "BA1": {"ok": ba1 <= 1, "detail": f"{ba1} of {len(unread)} unread passes read as wrong or unclear"},
        "BA2": {"ok": ba2 >= 38, "detail": f"{ba2} of {len(calc)} agree"},
        "BA3": {"ok": ba3 >= 18, "detail": f"{ba3} of {len(earlier)} agree with 13 October"},
        "BF1": {"ok": all(v["full_diagnosis"] >= 9 for v in per_system.values()),
                "detail": "; ".join(f"{s} {v['full_diagnosis']} of {v['planted']}" for s, v in per_system.items())},
        "BF2": {"ok": all(v["controls_good"] >= 5 for v in per_system.values()),
                "detail": "; ".join(f"{s} {v['controls_good']} of {v['controls']}" for s, v in per_system.items())},
    }
    summary = {"plan": str(PLAN.relative_to(ROOT)), "readings": str(READINGS.relative_to(ROOT)),
               "scored": time.strftime("%Y-%m-%d %H:%M"), "marks": marks,
               "calc": {"replies": len(calc), "rule_passes": rule_passes,
                        "false_passes": len(false_pass),
                        "false_pass_rate_wilson": wilson(len(false_pass), rule_passes),
                        "false_pass_items": [(x["item_id"], x["system"], x["reading"]["verdict"],
                                              x["reading"].get("note", "")) for x in false_pass],
                        "unread_false_passes": ba1},
               "feedback": per_system, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    for k, v in marks.items():
        print(f"{k} {'met' if v['ok'] else 'MISSED'}: {v['detail']}")
    for s, v in per_system.items():
        print(f"{s}: locates {v['locates']}/{v['planted']}, explains {v['explains_yes']} (+{v['explains_partly']} partly), "
              f"value {v['value']}, false alarms {v['false_alarm']}; controls confirmed {v['confirms']}/{v['controls']}, "
              f"invented {v['invents']}")
    print(f"written {OUT / 'summary.json'}")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind reading</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<style>
  :root { --paper:#f3f5f3; --panel:#fff; --sunk:#eaeeeb; --ink:#16201e; --muted:#5c6763; --rule:#d4dbd6;
          --accent:#1c6355; --accent-soft:#dcebe6; --warn-soft:#f7eede; --no-soft:#f6e3dc; }
  @media (prefers-color-scheme: dark) {
    :root { --paper:#101513; --panel:#181e1c; --sunk:#131917; --ink:#e3e8e5; --muted:#95a09b; --rule:#29312d;
            --accent:#5dbfa6; --accent-soft:#1b2a25; --warn-soft:#2a2117; --no-soft:#2e1c16; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--paper); color:var(--ink); font:15px/1.55 system-ui, sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--rule);
           padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header b { font-weight:600; }
  .meter { flex:1; min-width:120px; height:6px; background:var(--sunk); border-radius:3px; overflow:hidden; }
  .meter i { display:block; height:100%; background:var(--accent); }
  main { max-width:1500px; margin:0 auto; padding:16px; }
  .card { background:var(--panel); border:1px solid var(--rule); border-radius:8px; padding:14px 16px; margin-bottom:14px; }
  .label { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:4px; }
  .key { background:var(--accent-soft); }
  .mistake { background:var(--warn-soft); }
  .replies { display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:14px; }
  @media (max-width: 900px) { .replies { grid-template-columns:1fr; } }
  .reply .body { max-height:70vh; overflow:auto; border:1px solid var(--rule); border-radius:6px; padding:8px 12px; background:var(--sunk); }
  .crit { margin-top:10px; }
  .crit .q { margin:8px 0 4px; }
  .crit button { font:inherit; padding:5px 12px; margin:0 6px 4px 0; border:1px solid var(--rule); border-radius:5px;
                 background:var(--panel); color:var(--ink); cursor:pointer; }
  .crit button.on { background:var(--accent); border-color:var(--accent); color:#fff; }
  textarea { width:100%; min-height:40px; font:inherit; margin-top:8px; padding:6px; border:1px solid var(--rule);
             border-radius:5px; background:var(--panel); color:var(--ink); }
  #save { font:inherit; font-weight:600; padding:10px 22px; border:0; border-radius:6px; background:var(--accent); color:#fff; cursor:pointer; }
  #save:disabled { opacity:.45; cursor:default; }
  #msg { margin-left:12px; color:var(--muted); }
  .katex-display { overflow-x:auto; overflow-y:hidden; }
  pre, code { white-space:pre-wrap; }
</style>
</head>
<body>
<header><b>Blind reading</b><span id="where"></span><div class="meter"><i id="bar"></i></div><span id="count"></span></header>
<main id="main"><div class="card">Loading...</div></main>
<script>
let current = null, marks = {};
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function renderRich(src) {
  const math = [];
  let text = String(src).replace(/\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|\\\(([\s\S]+?)\\\)|\$([^\n$]+?)\$/g,
    (m, a, b, c, d) => { const display = a !== undefined || b !== undefined;
      math.push({ tex: a ?? b ?? c ?? d, display }); return "@@MATH" + (math.length - 1) + "@@"; });
  text = text.replace(/</g, "&lt;");
  const html = window.marked ? marked.parse(text) : "<p>" + text.replace(/\n/g, "<br>") + "</p>";
  return html.replace(/@@MATH(\d+)@@/g, (m, i) => { const { tex, display } = math[Number(i)];
    if (!window.katex) return "<code>" + escapeHtml(tex) + "</code>";
    try { return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" }); }
    catch (e) { return "<code>" + escapeHtml(tex) + "</code>"; } });
}
function complete() {
  return current.replies.every(r => current.criteria.every(([k]) => (marks[r.label] || {})[k]));
}
function paint() {
  document.querySelectorAll(".crit button").forEach(b => {
    b.classList.toggle("on", (marks[b.dataset.r] || {})[b.dataset.k] === b.dataset.v); });
  document.getElementById("save").disabled = !complete();
}
function show(view) {
  current = view; marks = {};
  const p = view.progress;
  document.getElementById("count").textContent = p.read + " of " + p.total + " replies read";
  document.getElementById("bar").style.width = (p.total ? 100 * p.read / p.total : 0) + "%";
  const main = document.getElementById("main");
  if (view.done) { document.getElementById("where").textContent = "";
    main.innerHTML = '<div class="card"><b>All read.</b> Every reading is saved. You can close this page.</div>'; return; }
  const part = view.part === "calc" ? "Part A - a calculation" :
               view.part === "error" ? "Part B - the student made a mistake" : "Part B - the student's working is correct";
  document.getElementById("where").textContent = "Item " + view.n + " of " + view.of + " - " + part;
  const s = view.shown;
  let html = '<div class="card"><div class="label">' + (view.part === "calc" ? "Question" : "The student's message") +
             '</div>' + renderRich(s.question) + '</div>';
  if (view.part === "error") html += '<div class="card mistake"><div class="label">The mistake the student was given</div>' +
             escapeHtml(s.mistake) + '</div>';
  if (view.part === "control") html += '<div class="card mistake"><div class="label">The student\'s working</div>Correct throughout.</div>';
  html += '<div class="card key"><div class="label">' + (view.part === "calc" ? "What the answer must state" : "The right result") +
          '</div>' + s.must_state.map(x => renderRich(x)).join("") + (s.note ? '<p><i>' + escapeHtml(s.note) + '</i></p>' : '') + '</div>';
  html += '<div class="replies">';
  view.replies.forEach(r => {
    html += '<div class="card reply"><div class="label">Reply ' + r.label + '</div><div class="body">' + renderRich(r.text) +
            '</div><div class="crit">';
    view.criteria.forEach(([k, q, opts]) => {
      html += '<div class="q">' + escapeHtml(q) + '</div>' + opts.map(v =>
        '<button data-r="' + r.label + '" data-k="' + k + '" data-v="' + v + '">' + v + '</button>').join("");
    });
    html += '<textarea data-note="' + r.label + '" placeholder="Note (optional)"></textarea></div></div>';
  });
  html += '</div><div class="card"><button id="save" disabled>Save both and go on</button><span id="msg"></span></div>';
  main.innerHTML = html;
  main.querySelectorAll(".crit button").forEach(b => b.onclick = () => {
    marks[b.dataset.r] = marks[b.dataset.r] || {}; marks[b.dataset.r][b.dataset.k] = b.dataset.v; paint(); });
  document.getElementById("save").onclick = save;
  window.scrollTo(0, 0); paint();
}
async function save() {
  const notes = {}; document.querySelectorAll("textarea[data-note]").forEach(t => notes[t.dataset.note] = t.value);
  document.getElementById("save").disabled = true;
  const res = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify({ n: current.n, marks, notes }) });
  const body = await res.json();
  if (!res.ok) { document.getElementById("msg").textContent = body.error || "Not saved."; paint(); return; }
  show(body.next);
}
fetch("/api/next").then(r => r.json()).then(show);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("draw")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--no-browser", action="store_true")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.cmd == "draw":
        return draw()
    if args.cmd == "serve":
        return serve(args.port, not args.no_browser)
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
