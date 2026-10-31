"""E32: similar past questions on fifty fresh queries (protocols/problems-fresh.md).

    .venv\\Scripts\\python.exe scripts\\problem_fresh.py register   (Sanjan's queries -> queries.jsonl, once)
    .venv\\Scripts\\python.exe scripts\\problem_fresh.py pool       (search, term candidates, the shuffled pool)
    .venv\\Scripts\\python.exe scripts\\problem_fresh.py judge      (the page Sanjan scores the pool on)
    .venv\\Scripts\\python.exe scripts\\problem_fresh.py score      (R1-R3, after the last judgement)

The queries are Sanjan's, written into queries.txt without opening the papers searched.
`register` fixes them with a digest before anything searches them; `pool` runs the served
path (ResourceFinder.find on the browser's search text, threshold unchanged) and adds the
top units by the hand-written terms in terms.json, committed before the search; `judge`
shows each query's pool shuffled, without source, rank or score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
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

BANK = ROOT / "stress" / "bank" / "problems-fresh"
TEXT = BANK / "queries.txt"
QUERIES = BANK / "queries.jsonl"
MANIFEST = BANK / "manifest.json"
TERMS = BANK / "terms.json"
POOL = BANK / "pool.json"
JUDGEMENTS = ROOT / "grades" / "problems-fresh.jsonl"
OUT = ROOT / "stress" / "results" / "problems-fresh"
UNITS = ROOT / "corpus" / "processed" / "problems.jsonl"
SEED = 20261018
TERM_CANDIDATES = 3
UNEXAMINED = "| not examined"


def digest(path: Path) -> str:
    """As scripts/problem_eval.py digests the twelve-October queries."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- register ------------------------------------------------------------------------------

def register() -> int:
    if QUERIES.exists():
        print(f"{QUERIES} exists; the queries are registered once.")
        return 1
    queries = []
    for line in TEXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, _, text = line.partition(":")
        if kind not in ("S", "P"):
            print(f"not an S: or P: line: {line!r}")
            return 1
        text = text.strip()
        unexamined = text.endswith(UNEXAMINED)
        if unexamined:
            text = text[: -len(UNEXAMINED)].strip()
        queries.append({"id": f"fresh-{len(queries) + 1:02d}",
                        "kind": "student" if kind == "S" else "statement",
                        "question": text, "predicted_unexamined": unexamined})
    counts = {"queries": len(queries),
              "student": sum(q["kind"] == "student" for q in queries),
              "statement": sum(q["kind"] == "statement" for q in queries),
              "predicted_unexamined": sum(q["predicted_unexamined"] for q in queries)}
    if counts != {"queries": 50, "student": 25, "statement": 25, "predicted_unexamined": 12}:
        print(f"the file does not match the registered design: {counts}")
        return 1
    QUERIES.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries),
                       encoding="utf-8", newline="\n")
    MANIFEST.write_text(json.dumps({
        "written": "2026-10-18", "by": "Sanjan, without opening the papers searched",
        "file": QUERIES.name, "source": TEXT.name, **counts,
        "digest": digest(QUERIES), "source_digest": digest(TEXT),
        "note": "Registered before any search over the question units; relevance is judged afterwards "
                "on the pool built by `pool`."}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"registered {counts} with digest {digest(QUERIES)}")
    return 0


# --- the pool ------------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("−", "-").replace("–", "-")).strip()


def _term_hits(text: str, terms: list[str]) -> int:
    low = _norm(text)
    hits = 0
    for t in terms:
        t_n = _norm(t)
        if " " in t_n or not t_n.isalnum():
            hits += t_n in low
        else:
            hits += re.search(r"(?<![a-z0-9])" + re.escape(t_n) + r"(?![a-z0-9])", low) is not None
    return hits


def pool() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if digest(QUERIES) != manifest["digest"]:
        print("the queries no longer match their registration")
        return 1
    terms = json.loads(TERMS.read_text(encoding="utf-8"))
    queries = _jsonl(QUERIES)
    missing = [q["id"] for q in queries if not terms.get(q["id"])]
    if missing:
        print(f"no terms for {missing}")
        return 1

    from ym07.config import load_config
    from ym07.ocr import plain as searchable
    from ym07.problems import load
    from ym07.resources import ResourceFinder

    config = load_config(str(ROOT / "configs" / "eee4114f.yaml"))
    finder = ResourceFinder(method=config.retrieval.method)
    units = {u.unit_id: u for u in load(UNITS)}
    rng = random.Random(SEED)
    plan, shown_log = [], []
    for q in queries:
        found = finder.find(searchable(q["question"]), lectures=False)
        shown = [p["unit"] for p in found.problems]
        # What the threshold held back, for a query shown nothing (amendment of 18 October,
        # before any judging): the units the platform would have shown, so the judgement
        # can say whether withholding them was right.
        withheld = [] if shown else [u.unit_id for u, _ in finder.problems.search(searchable(q["question"]), 3)]
        shown_log.append({"id": q["id"], "shown": shown, "withheld_top": withheld, "note": found.problem_note,
                          "scores": [p.get("score") for p in found.problems]})
        by_terms = sorted(((_term_hits(u.text, terms[q["id"]]), uid) for uid, u in units.items()),
                          key=lambda t: (-t[0], t[1]))
        candidates = [uid for hits, uid in by_terms if hits > 0][:TERM_CANDIDATES]
        members = list(dict.fromkeys(shown + withheld + candidates))
        rng.shuffle(members)
        plan.append({"id": q["id"], "units": members})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shown.json").write_text(json.dumps({"queries_digest": manifest["digest"], "threshold": 0.74,
                                                "shown": shown_log}, indent=1) + "\n",
                                    encoding="utf-8", newline="\n")
    POOL.write_text(json.dumps({"seed": SEED, "term_candidates": TERM_CANDIDATES,
                                "queries_digest": manifest["digest"], "pool": plan}, indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    total = sum(len(p["units"]) for p in plan)
    print(f"pool: {total} units over {len(plan)} queries; what was shown is in {OUT / 'shown.json'}, "
          f"which the judge does not see")
    return 0


# --- judge ---------------------------------------------------------------------------------

RUBRIC = [("3", "the same problem (same setup; the numbers may differ)"),
          ("2", "the same kind of problem: the same method would solve it"),
          ("1", "the same topic only"),
          ("0", "unrelated")]


class Session:
    def __init__(self) -> None:
        from ym07.problems import load

        self.queries = {q["id"]: q for q in _jsonl(QUERIES)}
        self.pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"]
        self.units = {u.unit_id: u for u in load(UNITS)}
        self.done = {r["query"] for r in _jsonl(JUDGEMENTS)} if JUDGEMENTS.exists() else set()
        self.lock = threading.Lock()

    def progress(self) -> dict:
        return {"done": len(self.done), "total": len(self.pool),
                "units_left": sum(len(p["units"]) for p in self.pool if p["id"] not in self.done)}

    def next(self) -> dict:
        for n, entry in enumerate(self.pool, start=1):
            if entry["id"] in self.done:
                continue
            q = self.queries[entry["id"]]
            return {"done": False, "progress": self.progress(), "n": n, "of": len(self.pool),
                    "query": entry["id"], "question": q["question"],
                    "units": [{"slot": k, "reference": self.units[u].reference(), "text": self.units[u].text}
                              for k, u in enumerate(entry["units"])],
                    "rubric": RUBRIC}
        return {"done": True, "progress": self.progress()}

    def record(self, payload: dict) -> tuple[int, dict]:
        entry = next((p for p in self.pool if p["id"] == payload.get("query")), None)
        if entry is None or entry["id"] in self.done:
            return 409, {"error": "This query is not current; reload the page."}
        scores = payload.get("scores") or {}
        if any(str(scores.get(str(k))) not in ("0", "1", "2", "3") for k in range(len(entry["units"]))):
            return 400, {"error": "Score every past question."}
        JUDGEMENTS.parent.mkdir(parents=True, exist_ok=True)
        with open(JUDGEMENTS, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"query": entry["id"],
                                 "scores": {u: int(scores[str(k)]) for k, u in enumerate(entry["units"])},
                                 "note": str(payload.get("note") or "").strip(),
                                 "saved": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        self.done.add(entry["id"])
        return 200, {"saved": True, "next": self.next()}


def judge(port: int, open_browser: bool) -> int:
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
                self._json(400, {"error": "The judgement could not be read."})
                return
            with session.lock:
                status, body = session.record(payload)
            self._json(status, body)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    p = session.progress()
    print(f"Judging at {url}: {p['done']} of {p['total']} queries done, {p['units_left']} past questions "
          f"left, saving to {JUDGEMENTS}")
    print("Each query is saved as you go. Ctrl-C here to stop; run again to resume.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {session.progress()['done']} of {p['total']} queries done.")
    finally:
        httpd.server_close()
    return 0


# --- score ---------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def _marks(queries: list[dict], judged: dict, shown: dict) -> dict:
    """R1-R3 on one group of queries. Covered means a unit in the pool scored 2 or 3."""
    def rel(q, u):
        return judged[q["id"]].get(u, 0) >= 2

    covered = [q for q in queries if any(s >= 2 for s in judged[q["id"]].values())]
    uncovered = [q for q in queries if q not in covered]
    hit = [q for q in covered if any(rel(q, u) for u in shown[q["id"]])]
    top = [q for q in covered if shown[q["id"]] and rel(q, shown[q["id"]][0])]
    quiet = [q for q in uncovered if not shown[q["id"]]]
    rr = [next((1 / (k + 1) for k, u in enumerate(shown[q["id"]]) if rel(q, u)), 0.0) for q in covered]
    return {"queries": len(queries), "covered": len(covered), "uncovered": len(uncovered),
            "R1": [len(hit), len(covered), wilson(len(hit), len(covered))],
            "R2": [len(top), len(covered), wilson(len(top), len(covered))],
            "R3": [len(quiet), len(uncovered), wilson(len(quiet), len(uncovered))],
            "MRR": round(sum(rr) / len(rr), 3) if rr else None,
            "shown_on_covered": sum(bool(shown[q["id"]]) for q in covered)}


def score() -> int:
    queries = _jsonl(QUERIES)
    judged = {r["query"]: r["scores"] for r in _jsonl(JUDGEMENTS)} if JUDGEMENTS.exists() else {}
    if len(judged) < len(queries):
        print(f"{len(queries) - len(judged)} of {len(queries)} queries not yet judged")
        return 1
    shown = {s["id"]: s["shown"] for s in json.loads((OUT / "shown.json").read_text(encoding="utf-8"))["shown"]}
    every = _marks(queries, judged, shown)

    def rate(mark: str) -> float:
        k, n, _ = every[mark]
        return k / n if n else 0.0

    passed = {"R1": rate("R1") >= 0.70, "R2": rate("R2") >= 0.60, "R3": rate("R3") >= 2 / 3}
    by_kind = {k: _marks([q for q in queries if q["kind"] == k], judged, shown) for k in ("student", "statement")}
    predicted = [q for q in queries if q["predicted_unexamined"]]
    borne_out = sum(not any(s >= 2 for s in judged[q["id"]].values()) for q in predicted)
    summary = {"marks": passed, "all": every, "by_kind": by_kind,
               "unexamined_predictions_borne_out": [borne_out, len(predicted)],
               "judged_units": sum(len(v) for v in judged.values()), "scored": time.strftime("%Y-%m-%d %H:%M")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k in ("R1", "R2", "R3"):
        a, n, (lo, hi) = every[k]
        print(f"{k} {'met' if passed[k] else 'MISSED'}: {a} of {n} ({100 * lo:.0f}-{100 * hi:.0f}%)")
    print(f"MRR {every['MRR']}; covered {every['covered']}, uncovered {every['uncovered']}; "
          f"not-examined predictions borne out {borne_out} of {len(predicted)}")
    print(f"written {OUT / 'summary.json'}")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Similar past questions</title>
<style>
  :root { --paper:#f3f5f3; --panel:#fff; --sunk:#eaeeeb; --ink:#16201e; --muted:#5c6763; --rule:#d4dbd6;
          --accent:#1c6355; --accent-soft:#dcebe6; }
  @media (prefers-color-scheme: dark) {
    :root { --paper:#101513; --panel:#181e1c; --sunk:#131917; --ink:#e3e8e5; --muted:#95a09b; --rule:#29312d;
            --accent:#5dbfa6; --accent-soft:#1b2a25; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--paper); color:var(--ink); font:15px/1.55 system-ui, sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--rule);
           padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  .meter { flex:1; min-width:120px; height:6px; background:var(--sunk); border-radius:3px; overflow:hidden; }
  .meter i { display:block; height:100%; background:var(--accent); }
  main { max-width:1100px; margin:0 auto; padding:16px; }
  .card { background:var(--panel); border:1px solid var(--rule); border-radius:8px; padding:14px 16px; margin-bottom:14px; }
  .query { background:var(--accent-soft); font-size:16px; }
  .label { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:4px; }
  .rubric { font-size:13px; color:var(--muted); }
  pre { white-space:pre-wrap; font:14px/1.5 ui-monospace, Menlo, Consolas, monospace; max-height:45vh; overflow:auto;
        background:var(--sunk); border-radius:6px; padding:8px 10px; margin:6px 0 10px; }
  button.s { font:inherit; min-width:44px; padding:6px 12px; margin-right:6px; border:1px solid var(--rule);
             border-radius:5px; background:var(--panel); color:var(--ink); cursor:pointer; }
  button.s.on { background:var(--accent); border-color:var(--accent); color:#fff; }
  #save { font:inherit; font-weight:600; padding:10px 22px; border:0; border-radius:6px; background:var(--accent); color:#fff; cursor:pointer; }
  #save:disabled { opacity:.45; cursor:default; }
  textarea { width:100%; min-height:40px; font:inherit; padding:6px; border:1px solid var(--rule); border-radius:5px;
             background:var(--panel); color:var(--ink); }
</style>
</head>
<body>
<header><b>Similar past questions</b><span id="where"></span><div class="meter"><i id="bar"></i></div><span id="count"></span></header>
<main id="main"><div class="card">Loading...</div></main>
<script>
let current = null, scores = {};
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function complete() { return current.units.every(u => scores[u.slot] !== undefined); }
function paint() {
  document.querySelectorAll("button.s").forEach(b => b.classList.toggle("on", scores[b.dataset.slot] === b.dataset.v));
  document.getElementById("save").disabled = !complete();
}
function show(v) {
  current = v; scores = {};
  const p = v.progress;
  document.getElementById("count").textContent = p.done + " of " + p.total + " queries done";
  document.getElementById("bar").style.width = (p.total ? 100 * p.done / p.total : 0) + "%";
  const main = document.getElementById("main");
  if (v.done) { document.getElementById("where").textContent = "";
    main.innerHTML = '<div class="card"><b>All judged.</b> Every judgement is saved. You can close this page.</div>'; return; }
  document.getElementById("where").textContent = "Query " + v.n + " of " + v.of;
  let h = '<div class="card query"><div class="label">Your query</div>' + esc(v.question) + '</div>';
  h += '<div class="card rubric">' + v.rubric.map(r => "<b>" + r[0] + "</b> " + esc(r[1])).join(" &middot; ") + '</div>';
  v.units.forEach((u, i) => {
    h += '<div class="card"><div class="label">Past question ' + (i + 1) + ' of ' + v.units.length + ' &middot; ' + esc(u.reference) +
         '</div><pre>' + esc(u.text) + '</pre>' +
         ["3", "2", "1", "0"].map(s => '<button class="s" data-slot="' + u.slot + '" data-v="' + s + '">' + s + '</button>').join("") + '</div>';
  });
  h += '<div class="card"><textarea id="note" placeholder="Note (optional)"></textarea><p><button id="save" disabled>Save and go on</button> <span id="msg"></span></p></div>';
  main.innerHTML = h;
  main.querySelectorAll("button.s").forEach(b => b.onclick = () => { scores[b.dataset.slot] = b.dataset.v; paint(); });
  document.getElementById("save").onclick = save;
  window.scrollTo(0, 0); paint();
}
async function save() {
  document.getElementById("save").disabled = true;
  const res = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: current.query, scores, note: document.getElementById("note").value }) });
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
    sub.add_parser("register")
    sub.add_parser("pool")
    j = sub.add_parser("judge")
    j.add_argument("--port", type=int, default=8078)
    j.add_argument("--no-browser", action="store_true")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.cmd == "register":
        return register()
    if args.cmd == "pool":
        return pool()
    if args.cmd == "judge":
        return judge(args.port, not args.no_browser)
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
