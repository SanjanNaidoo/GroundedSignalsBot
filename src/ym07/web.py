"""The tutor in a browser — the demo, rather than a description of the demo.

Built from the same parts as `ym07 chat` and the benchmark: the same course
config, the same system prompt, the same retriever, the same model clients. A
web front end that reimplemented any of those would be a different system from
the one the report has numbers for, and showing it to anyone would be showing
them the wrong thing.

Served by the standard library. A web framework would be one more dependency on
a harness that has to stay installable on a laptop with no GPU, for a page that
serves one person at a time on one machine.

Three things are on screen that a terminal hides, and each is an argument the
report makes:

- **Provenance.** Every answer lists the passages it was given, with the file
  and page they came from, so grounding can be checked rather than believed.
- **Latency and cost.** Shown per turn. The case for a 7B model on one card is
  not that it is more accurate than a frontier model; it is what the answer
  costs and how long it takes.
- **Retrieval, on a switch.** Turning it off is the difference between C1 and
  C2 with the same weights, which is otherwise a table in a report.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import degrade
from .config import CourseConfig
from . import citations
from .hybrid import is_assessed
from .model import Answer, build_client, build_client_for
from .prompts import system_prompt, user_prompt
from dataclasses import dataclass, replace

from .handout import read_facts
from .resources import ResourceFinder
from .retrieval import about_the_course, conversation_query, handout_passages, passages_for


def said_so_far(question: str, history, screenshot: str = "") -> str:
    """Everything the student has said, for deciding whether the work is assessed.

    The student's own turns and their question, never the retrieved passages: course
    material is full of the words that mark assessed work - assignment, submit, due,
    marks - and reading the whole prompt instead cost the first final run 125 of 264
    calculations. A screenshot counts, because "this is my assignment, due Friday" is
    as likely to be photographed as typed.
    """
    mine = [text for who, text in (history or []) if str(who).lower().startswith("s")]
    return "\n".join([*mine, screenshot or "", question])


class SpendCap:
    """A hosted model's running cost for one browser session, with a ceiling.

    The batch runs stop at a stated spend, and a demonstration should too. Each answer's
    cost is already shown under it on the page; this keeps the session's total in the
    console and, once the total reaches the ceiling, answers with an error instead of
    calling the API again. Restarting the server starts a new session.
    """

    def __init__(self, client, max_usd: float, out=print):
        self._client = client          # named so degrade.cap_owner can see through it
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self.answers = 0
        self._out = out

    def answer(self, *, system, user, model, max_tokens, effort) -> Answer:
        if self.max_usd and self.spent_usd >= self.max_usd:
            return Answer(text="", model=model, effort=effort, stop_reason=None,
                          refusal_category=None, input_tokens=0, output_tokens=0,
                          cached_input_tokens=0, latency_s=0.0,
                          error=(f"This session has spent ${self.spent_usd:.2f}, its ceiling "
                                 f"of ${self.max_usd:.2f}. Restart the server with a higher "
                                 "--max-usd to go on."))
        reply = self._client.answer(system=system, user=user, model=model,
                                    max_tokens=max_tokens, effort=effort)
        self.answers += 1
        self.spent_usd += reply.cost_usd
        self._out(f"  ${reply.cost_usd:.3f} this answer; ${self.spent_usd:.2f} over "
                  f"{self.answers} answer(s) this session, ceiling ${self.max_usd:.2f}")
        return reply


@dataclass
class Pipeline:
    """One model the browser can answer with, and everything it is served with.

    The platform serves more than one (13 October): Opus 5 inside the delivered
    pipeline and the local R1 system, chosen per question by a switch on the page. Each
    keeps its own client, retrieval depth and prompt, exactly as its condition was
    measured, and its own lock: the local model and the screenshot reader share the
    card's lock, while a hosted model needs no card and does not wait behind it.
    """

    key: str                       # the condition's name, which the page sends back
    label: str                     # what the student sees on the switch
    client: object
    model_id: str
    top_k: int = 4
    use_course_prompt: bool = True
    lectures: bool = True          # lecture minutes beside its answers
    retrieval: bool = True         # whether its condition is grounded at all
    hosted: bool = False
    detail: str = ""               # a line for the switch's tooltip
    lock: object = None
    # The handout lookup (E27), for a condition that has one: course facts answered
    # from the handout, cited by page, with no model asked. See ym07.coursefacts.
    lookup: object = None

    def public(self) -> dict:
        return {"key": self.key, "label": self.label, "model": self.model_id,
                "lectures": self.lectures, "hosted": self.hosted, "detail": self.detail}


def offers_lectures(model_id: str) -> bool:
    """Whether lecture minutes are offered beside a model's answers.

    Sanjan, 13 October: lecture-segment retrieval is part of the Opus 5 tutor and not
    of the local R1 one. Kept here, as a property of what the browser offers, rather
    than on the conditions, which describe what was measured. R1 is recognised by its
    weights, so every R1 variant served is covered.
    """
    return "r1-distill" not in (model_id or "").lower()


# What a student sees in the lecture card when the chosen model does not offer them.
LECTURES_ELSEWHERE = ("Lecture minutes come with the Opus 5 tutor. Switch to Opus 5 to see "
                      "where a lecture explains this.")


def model_label(model_id: str) -> tuple[str, str]:
    """The name on the switch, and a line for its tooltip."""
    low = (model_id or "").lower()
    if low.startswith("claude-opus-5"):
        return "Opus 5", "Claude Opus 5 through the delivered pipeline; hosted, about $0.05 an answer"
    if "r1-distill" in low:
        return "R1", "DeepSeek-R1-Distill-7B with the course adapter, on this machine; free, offline"
    return (model_id or "model").rsplit("/", 1)[-1], ""


class _Sentinel:
    """Stands in for an Answer where there is none; the ladder only reads `error`."""

    def __init__(self, error: str | None = None):
        self.error = error


# The tab was closed mid-answer: not a failure to retry, and nothing more to send.
_Closed = _Sentinel(None)
# The streamer finished without handing back a reply.
_Nothing = _Sentinel("the model produced nothing")

# Matches the terminal tutor, for the same reason: enough turns to sustain a
# scaffolded exchange, few enough that the resent transcript does not dominate.
MAX_HISTORY_TURNS = 6

# A question, not a pasted chapter. Past this the prompt crowds out the retrieved
# passages and, with a conversation behind it, runs the card out of memory - so it is
# refused with a message rather than allowed to fail mid-answer.
MAX_QUESTION_CHARS = 4000

# A pasted screenshot arrives base64-encoded inside the same JSON post, which costs
# about a third in size, so the request cap has to clear the image cap in ocr.py with
# room for the question and the conversation behind it.
MAX_REQUEST_BYTES = 10 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__CODE__ tutor</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Jost:wght@400;500;600&display=swap">
<style>
  /* Lavender ground, white cards, periwinkle for what the student does and yellow for
     the tutor. Offline the font falls back to the system's and nothing else changes. */
  :root {
    --bg: #e4e3f7; --blob: #d7d5f3; --panel: #ffffff; --ink: #2f3a5a; --muted: #8b93ad;
    --line: #e7e9f3; --accent: #6c8ef5; --accent-dark: #5473e8; --accent-soft: #eef2fe;
    --student: #6c8ef5; --student-ink: #ffffff; --tutor: #f8f9fd;
    --warn: #b0602a; --warn-soft: #fdf1e6; --sun: #f8c53a; --sun-soft: #fff5d9;
    --shadow: 0 8px 28px rgba(84, 86, 160, 0.10);
    --lift: 0 6px 16px rgba(108, 142, 245, 0.35);
  }
  * { box-sizing: border-box; }
  /* An element marked hidden stays hidden whatever display its own rule gives it. */
  [hidden] { display: none !important; }
  body {
    margin: 0; color: var(--ink);
    background:
      radial-gradient(circle at 8% 4%, var(--blob) 0 230px, transparent 231px),
      radial-gradient(circle at 92% 26%, var(--blob) 0 280px, transparent 281px),
      radial-gradient(circle at 34% 108%, var(--blob) 0 320px, transparent 321px),
      var(--bg);
    background-attachment: fixed;
    font: 15px/1.6 "Jost", ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  }
  header {
    max-width: 1180px; margin: 14px auto 0; background: var(--panel);
    border-radius: 18px; box-shadow: var(--shadow);
    padding: 12px 22px; display: flex; flex-wrap: wrap; gap: 10px 22px;
    align-items: center; position: sticky; top: 12px; z-index: 5;
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand svg { width: 40px; height: 36px; flex: none; }
  header h1 { font-size: 24px; margin: 0; font-weight: 600; color: var(--accent); letter-spacing: 0.01em; }
  header .meta { color: var(--muted); font-size: 13px; }
  header .meta code {
    font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--accent-soft); color: var(--accent-dark);
    padding: 3px 10px; border-radius: 999px;
    display: inline-block; max-width: 230px; vertical-align: bottom;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .grow { flex: 1; }
  .toggle { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--muted); cursor: pointer; }
  .toggle input {
    appearance: none; -webkit-appearance: none; margin: 0; cursor: pointer;
    width: 40px; height: 22px; border-radius: 999px; background: var(--line);
    position: relative; transition: background 0.2s;
  }
  .toggle input::after {
    content: ""; position: absolute; top: 3px; left: 3px; width: 16px; height: 16px;
    border-radius: 50%; background: #fff; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s;
  }
  .toggle input:checked { background: var(--accent); }
  .toggle input:checked::after { transform: translateX(18px); }
  .toggle input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .models { display: inline-flex; gap: 4px; padding: 4px; border-radius: 999px; background: var(--accent-soft); }
  .models button {
    background: transparent; color: var(--accent-dark); border-radius: 999px;
    padding: 7px 16px; font-size: 14px; font-weight: 500;
    display: inline-flex; align-items: baseline; gap: 7px;
    transition: background 0.15s, color 0.15s, box-shadow 0.15s;
  }
  .models button .sub { font-size: 11.5px; font-weight: 400; opacity: 0.75; }
  .models button:hover:not([aria-checked="true"]) { background: rgba(108, 142, 245, 0.14); }
  .models button[aria-checked="true"] { background: var(--accent); color: #fff; box-shadow: var(--lift); }
  .models button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  main {
    max-width: 1180px; margin: 0 auto;
    /* The footer is fixed and its height changes when the composer grows to hold a
       transcription, so the gap underneath is measured rather than guessed. */
    padding: 20px 0 calc(var(--footer-h, 140px) + 24px);
    display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 22px; align-items: start;
  }
  @media (max-width: 1212px) { main, header { margin-left: 16px; margin-right: 16px; } }
  @media (max-width: 900px) {
    main { grid-template-columns: minmax(0, 1fr); }
    aside { position: static; max-height: none; }
  }
  #transcript {
    display: flex; flex-direction: column; gap: 20px; min-width: 0;
    background: var(--panel); border-radius: 18px; box-shadow: var(--shadow);
    padding: 24px; min-height: 55vh;
  }
  .turn { display: flex; flex-direction: column; gap: 7px; max-width: 88%; min-width: 0; }
  .turn.tutor, .turn.failed { align-self: flex-start; }
  .turn.student { align-self: flex-end; align-items: flex-end; }
  .who { display: flex; align-items: center; gap: 9px; font-size: 14px; font-weight: 500; color: var(--ink); }
  .who::before {
    content: ""; width: 30px; height: 30px; border-radius: 50%; flex: none;
    background-repeat: no-repeat; background-position: center;
  }
  .tutor .who::before, .failed .who::before {
    background-color: var(--sun-soft); background-size: 72%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 36'%3E%3Cpath d='M10 2h20a8 8 0 0 1 8 8v8a8 8 0 0 1-8 8H18l-8 7 2-7h-2a8 8 0 0 1-8-8v-8a8 8 0 0 1 8-8z' fill='%23f8c53a'/%3E%3Ccircle cx='13' cy='14' r='2.2' fill='white'/%3E%3Ccircle cx='20' cy='14' r='2.2' fill='white'/%3E%3Ccircle cx='27' cy='14' r='2.2' fill='white'/%3E%3C/svg%3E");
  }
  .student .who { flex-direction: row-reverse; }
  .student .who::before {
    background-color: var(--accent); background-size: 64%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='8' r='4.2' fill='white'/%3E%3Cpath d='M3.5 21c.8-4.4 4.2-7 8.5-7s7.7 2.6 8.5 7z' fill='white'/%3E%3C/svg%3E");
  }
  .bubble {
    background: var(--tutor); border: 1px solid var(--line); border-radius: 4px 18px 18px 18px;
    padding: 14px 18px; white-space: pre-wrap; overflow-wrap: anywhere; min-width: 0; max-width: 100%;
  }
  .student .bubble {
    background: var(--student); color: var(--student-ink); border-color: transparent;
    border-radius: 18px 4px 18px 18px; box-shadow: var(--lift);
  }
  .student .bubble a { color: inherit; }
  /* Typeset answers stop being preformatted text: the markdown supplies the spacing. */
  .bubble.rich { white-space: normal; }
  .bubble.rich p { margin: 0 0 10px; }
  .bubble.rich p:last-child, .bubble.rich ol:last-child, .bubble.rich ul:last-child { margin-bottom: 0; }
  .bubble.rich ol, .bubble.rich ul { margin: 0 0 10px; padding-left: 22px; }
  .bubble.rich h1, .bubble.rich h2, .bubble.rich h3 { font-size: 16px; font-weight: 600; margin: 14px 0 6px; color: var(--accent-dark); }
  .bubble.rich h1:first-child, .bubble.rich h2:first-child, .bubble.rich h3:first-child { margin-top: 0; }
  .bubble.rich code { font-size: 0.92em; background: var(--accent-soft); padding: 1px 5px; border-radius: 5px; }
  .bubble.rich table { border-collapse: collapse; margin: 6px 0 12px; font-size: 14px; }
  .bubble.rich th, .bubble.rich td { border: 1px solid var(--line); padding: 4px 10px; }
  .bubble.rich th { background: var(--accent-soft); font-weight: 500; }
  .katex-display { overflow-x: auto; overflow-y: hidden; padding: 2px 0; }
  .cursor { display: inline-block; width: 8px; border-radius: 2px; background: var(--accent); animation: blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .failed .bubble { background: var(--warn-soft); border-color: transparent; color: var(--warn); }
  .stats { font-size: 12px; color: var(--muted); display: flex; gap: 6px 14px; flex-wrap: wrap; padding-left: 4px; }
  /* The card ran short of memory. Said plainly, because an answer written without the
     course material behind it is a different kind of answer and the student is the one
     who has to decide how far to trust it. */
  .retrying, .degraded {
    font-size: 12.5px; padding: 6px 12px; border-radius: 10px; margin-top: 6px;
    background: var(--accent-soft); color: var(--muted); border: 1px dashed var(--line);
  }
  aside {
    display: flex; flex-direction: column; gap: 18px; font-size: 13px;
    position: sticky; top: 96px;
    max-height: calc(100vh - 110px - var(--footer-h, 140px)); overflow-y: auto;
    padding-right: 2px; scrollbar-width: thin;
  }
  .panel { background: var(--panel); border-radius: 18px; box-shadow: var(--shadow); padding: 18px 18px 12px; }
  .panel h2 {
    font-size: 17px; font-weight: 500; color: var(--ink); margin: 0 0 12px;
    display: flex; align-items: center; gap: 10px;
  }
  .panel h2 .ico {
    width: 28px; height: 28px; border-radius: 9px; display: grid; place-items: center; flex: none;
    background: var(--accent-soft); color: var(--accent);
  }
  .panel h2 .ico svg { width: 16px; height: 16px; }
  .source { padding: 10px 0; border-top: 1px solid var(--line); }
  .source:first-of-type { border-top: 0; padding-top: 2px; }
  .tag {
    font: 600 11px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
    color: #fff; background: var(--accent); padding: 1px 7px; border-radius: 6px;
  }
  .src-name { display: block; margin-top: 4px; overflow-wrap: anywhere; font-weight: 500; }
  .score { color: var(--muted); font-size: 12px; }
  .empty { color: var(--muted); display: block; padding-bottom: 6px; }
  .source a { color: var(--accent); text-decoration: none; margin-left: 6px; font-size: 12px; font-weight: 500; }
  .source a:hover { text-decoration: underline; }
  .card {
    border: 1px solid var(--line); border-radius: 14px; padding: 11px 13px; margin-bottom: 10px;
    background: var(--tutor); transition: box-shadow 0.15s, transform 0.15s;
  }
  .card:hover { box-shadow: 0 6px 16px rgba(84, 86, 160, 0.12); transform: translateY(-1px); }
  .card .ref { font-weight: 600; color: var(--ink); display: block; margin-bottom: 3px; }
  .card .snip { color: var(--muted); display: block; margin-bottom: 6px; }
  .card a {
    display: inline-block; margin: 2px 8px 2px 0; padding: 3px 12px; border-radius: 8px;
    background: var(--accent); color: #fff; text-decoration: none; font-size: 12px; font-weight: 500;
  }
  .card a:hover { background: var(--accent-dark); }
  .card a + a { background: var(--accent-soft); color: var(--accent-dark); }
  .card .flag { display: inline-block; font-size: 11.5px; color: var(--warn); margin-right: 8px; }
  footer {
    position: fixed; bottom: 0; left: 0; right: 0; padding: 12px 16px 14px;
    background: linear-gradient(to top, var(--bg) 72%, transparent);
  }
  .composer {
    max-width: 1180px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end;
    background: var(--panel); border-radius: 18px; box-shadow: var(--shadow);
    padding: 9px 9px 9px 18px; border: 1px solid transparent;
  }
  .composer:focus-within { border-color: var(--accent); }
  textarea {
    flex: 1; resize: none; font: inherit; color: inherit; background: transparent;
    border: 0; padding: 11px 2px; min-height: 46px;
    /* A transcribed question is a dozen lines long and has to be checked line by
       line, so the composer grows to it instead of showing one line at a time. */
    max-height: 32vh; overflow-y: auto; line-height: 1.5;
  }
  textarea:focus { outline: none; }
  textarea::placeholder { color: var(--muted); }
  button {
    font: inherit; font-weight: 500; background: var(--accent); color: #fff;
    border: 0; border-radius: 10px; padding: 12px 20px; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: default; }
  .icon-btn { width: 46px; height: 46px; padding: 0; border-radius: 50%; display: grid; place-items: center; flex: none; }
  .icon-btn svg { width: 20px; height: 20px; }
  #attach { background: var(--panel); color: var(--muted); border: 1px solid var(--line); }
  #attach:hover { color: var(--accent); border-color: var(--accent); }
  #send { box-shadow: var(--lift); }
  #send:hover:not(:disabled) { background: var(--accent-dark); }
  #attached {
    max-width: 1180px; margin: 0 auto 8px; display: flex; gap: 10px; align-items: center;
    background: var(--panel); box-shadow: var(--shadow); border-radius: 14px;
    padding: 7px 12px; font-size: 13px;
  }
  #attached.warn { background: var(--warn-soft); color: var(--warn); }
  #thumb { height: 38px; width: auto; border-radius: 6px; border: 1px solid var(--line); }
  #attached button { background: none; color: var(--muted); border: 0; padding: 2px 6px;
                     font-size: 13px; text-decoration: underline; }
  .shotread {
    margin-top: 8px; border-left: 3px solid var(--sun); background: var(--sun-soft); color: var(--ink);
    padding: 7px 11px; border-radius: 0 10px 10px 0; font-size: 13px; white-space: pre-wrap;
  }
  .shotread b { display: block; font-size: 12px; color: var(--muted); font-weight: 600;
                margin-bottom: 3px; }
  #shothint {
    max-width: 1180px; margin: -2px auto 8px; font-size: 12.5px; color: var(--muted);
    padding: 0 6px;
  }
  #shothint.warn { color: var(--warn); }
  #shothint b { color: var(--ink); font-weight: 600; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
  #attached.reading { animation: pulse 1.1s ease-in-out infinite; }
  body.dragging { outline: 3px dashed var(--accent); outline-offset: -8px; }
  .chips { max-width: 1180px; margin: 0 auto 10px; display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    font-size: 12.5px; background: var(--panel); color: var(--accent-dark);
    border: 1px solid var(--line); border-radius: 999px; padding: 6px 14px; cursor: pointer;
    box-shadow: 0 2px 8px rgba(84, 86, 160, 0.07); transition: background 0.15s, color 0.15s;
  }
  .chip:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
  .total { max-width: 1180px; margin: 8px auto 0; font-size: 12px; color: var(--muted); padding: 0 6px; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <svg viewBox="0 0 48 42" aria-hidden="true">
      <path d="M20 3h16a10 10 0 0 1 10 10v6a10 10 0 0 1-10 10h-1l2 7-9-7h-8a10 10 0 0 1-10-10v-6A10 10 0 0 1 20 3z" fill="#eaa21a"/>
      <path d="M12 9h16a10 10 0 0 1 10 10v6a10 10 0 0 1-10 10h-8l-9 7 2-7h-1A10 10 0 0 1 2 25v-6A10 10 0 0 1 12 9z" fill="#f8c53a"/>
      <circle cx="13" cy="22" r="2.3" fill="#fff"/><circle cx="20" cy="22" r="2.3" fill="#fff"/><circle cx="27" cy="22" r="2.3" fill="#fff"/>
    </svg>
    <h1>__CODE__ tutor</h1>
  </div>
  <!-- The models this tutor can answer with, filled in from /state; hidden when there
       is only one. -->
  <div class="models" id="models" role="radiogroup" aria-label="Which model answers" hidden></div>
  <span class="meta">answering with <code id="model">…</code></span>
  <span class="grow"></span>
  <label class="toggle">
    <input type="checkbox" id="grounded" checked>
    <span id="grounding">course material</span>
  </label>
</header>

<main>
  <div id="transcript">
    <div class="turn tutor">
      <span class="who">Tutor</span>
      <!-- One line: the bubble keeps line breaks, so the source must not add any. -->
      <div class="bubble">Ask me anything from __CODE__. I answer from the course material and cite the pages I used. Beside each answer you will find past test and exam questions of the same kind, with their memoranda, and with Opus 5 the lecture minutes that explain it. If you tell me something is work you are handing in for marks, I will help you get there rather than write it for you.</div>
    </div>
  </div>
  <aside>
    <section class="panel">
      <h2><span class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h7M9 17h5"/></svg></span>Passages used</h2>
      <div id="sources"><span class="empty">Nothing retrieved yet.</span></div>
    </section>
    <section class="panel">
      <h2><span class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="3"/><path d="M10 9.5v5l4.5-2.5z" fill="currentColor"/></svg></span>Lecture minutes</h2>
      <div id="lectures"><span class="empty">Ask a question to see where a lecture explains it.</span></div>
    </section>
    <section class="panel">
      <h2><span class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V3h6v1M9 11l2 2 4-4M9 17h6"/></svg></span>Past questions</h2>
      <div id="problems"><span class="empty">Past class tests and exams of the same kind appear here, with their memoranda.</span></div>
    </section>
  </aside>
</main>

<footer>
  <div class="chips" id="chips"></div>
  <div id="attached" hidden>
    <img id="thumb" alt="the screenshot you attached">
    <span id="attachname"></span>
    <button type="button" id="dropshot" title="remove">remove</button>
  </div>
  <!-- The transcription goes into the composer, not into a box of its own, because a
       box the student reads is not the same as text the student can fix. -->
  <div id="shothint" hidden></div>
  <form class="composer" id="composer">
    <input type="file" id="shotfile" accept="image/*" hidden>
    <textarea id="question" rows="1" placeholder="Ask a question, or paste a screenshot…" autofocus></textarea>
    <button type="button" id="attach" class="icon-btn" title="Attach a screenshot (or paste one)" aria-label="Attach a screenshot"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg></button>
    <button id="send" type="submit" class="icon-btn" title="Ask" aria-label="Ask"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M3.4 20.4 21 12 3.4 3.6l.1 6.5L15 12 3.5 13.9z"/></svg></button>
  </form>
  <div class="total" id="total"></div>
</footer>

<script>
const transcript = document.getElementById("transcript");
const sourcesBox = document.getElementById("sources");
const lecturesBox = document.getElementById("lectures");
const problemsBox = document.getElementById("problems");
const box = document.getElementById("question");
const send = document.getElementById("send");
const totalLine = document.getElementById("total");
const groundedBox = document.getElementById("grounded");
const history = [];
let turns = 0, spend = 0;

// The models on the switch, from /state, and the one in use. The choice is remembered in
// this browser for convenience only: if storage is unavailable the page starts on the
// server's default, which is Opus 5 on the platform.
const modelsBox = document.getElementById("models");
let models = [], chosen = null;
const LECTURES_HINT = "Ask a question to see where a lecture explains it.";
const LECTURES_ELSEWHERE = "Lecture minutes come with the Opus 5 tutor.";

function choose(key, remember) {
  chosen = models.find(m => m.key === key) || models[0] || null;
  if (!chosen) return;
  modelsBox.querySelectorAll("button").forEach(b =>
    b.setAttribute("aria-checked", String(b.dataset.key === chosen.key)));
  const shown = document.getElementById("model");
  shown.textContent = String(chosen.model).split("/").pop();
  shown.title = chosen.model;
  // Before any answer, the lecture card says whether this model offers lecture minutes;
  // cards already shown for an earlier answer are left alone.
  if (!lecturesBox.querySelector(".card")) {
    emptyNote(lecturesBox, chosen.lectures ? LECTURES_HINT : LECTURES_ELSEWHERE);
  }
  if (remember) {
    try { localStorage.setItem("ym07.model", chosen.key); } catch (e) { /* not kept */ }
  }
}

function tutorLabel(name) {
  return models.length > 1 && name ? "Tutor \u00b7 " + name : "Tutor";
}

// The composer sizes itself to its contents, capped by CSS. A transcription the
// student has to check against a picture cannot be checked through a one-line slot.
// Growing it moves the fixed footer up over the transcript, so the page is told how
// much room to leave underneath - otherwise the end of an answer sits behind the
// composer where no amount of scrolling reaches it.
function grow() {
  box.style.height = "auto";
  box.style.height = box.scrollHeight + "px";
  fitFooter();
}

function fitFooter() {
  const bar = document.querySelector("footer");
  if (bar) {
    document.documentElement.style.setProperty("--footer-h", bar.offsetHeight + "px");
  }
}
window.addEventListener("resize", fitFooter);

// Follow the answer as it is written, but stop the moment the student scrolls up.
// Calling scrollTo on every streamed token pinned the page to the bottom for the
// whole minute an answer takes, so scrolling back to re-read the question was
// impossible - it snapped down again before the next frame.
let following = true;
function atBottom() {
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - 80;
}
function stick() {
  if (following) window.scrollTo(0, document.body.scrollHeight);
}
window.addEventListener("scroll", () => { following = atBottom(); }, {passive: true});

box.addEventListener("input", grow);

const EXAMPLES = [
  "What does the region of convergence of a z-transform tell you?",
  "Explain the small-signal hybrid-pi model of a BJT.",
  "This is due tomorrow for marks - just give me the final answer."
];

const chips = document.getElementById("chips");
EXAMPLES.forEach(text => {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip";
  chip.textContent = text.length > 58 ? text.slice(0, 55) + "..." : text;
  chip.title = text;
  chip.onclick = () => { box.value = text; box.focus(); };
  chips.appendChild(chip);
});

function addTurn(who, text, cls) {
  const turn = document.createElement("div");
  turn.className = "turn " + cls;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  turn.append(label, bubble);
  transcript.appendChild(turn);
  // A new turn is the student's own doing, so it always follows.
  following = true;
  stick();
  return turn;
}

function showSources(sources) {
  sourcesBox.textContent = "";
  if (!sources || !sources.length) {
    const none = document.createElement("span");
    none.className = "empty";
    none.textContent = groundedBox.checked
      ? "Nothing was retrieved for that question."
      : "Retrieval is off - the model is answering from its own weights.";
    sourcesBox.appendChild(none);
    return;
  }
  sources.forEach(hit => {
    const row = document.createElement("div");
    row.className = "source";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = hit.tag;
    const name = document.createElement("span");
    name.className = "src-name";
    name.textContent = hit.source;
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = "score " + hit.score.toFixed(3);
    row.append(tag, name, score);
    if (hit.doc) {
      const open = document.createElement("a");
      open.href = hit.doc; open.target = "_blank"; open.rel = "noopener";
      open.textContent = "open page";
      row.appendChild(open);
    }
    sourcesBox.appendChild(row);
  });
}

function emptyNote(box, text) {
  box.textContent = "";
  const none = document.createElement("span");
  none.className = "empty";
  none.textContent = text;
  box.appendChild(none);
}

// The lecture minutes and past questions found for the question, each shown as a
// file name and a second, or a paper, question and page - things the student can
// open and check - and withheld entirely when nothing is close enough. The server
// decides that with the thresholds the two experiments fixed; the page only shows
// what it was sent and says why when it was sent nothing.
function showResources(res) {
  if (!res) {
    emptyNote(lecturesBox, "Lecture minutes are shown for grounded answers.");
    emptyNote(problemsBox, "Past questions are shown for grounded answers.");
    return;
  }
  lecturesBox.textContent = "";
  if (!res.lectures || !res.lectures.length) {
    emptyNote(lecturesBox, res.lecture_note || "No lecture minute to show.");
  } else {
    res.lectures.forEach(w => {
      const card = document.createElement("div");
      card.className = "card";
      const ref = document.createElement("span");
      ref.className = "ref";
      ref.textContent = w.recording + " at " + w.stamp;
      const snip = document.createElement("span");
      snip.className = "snip";
      snip.textContent = w.snippet;
      card.append(ref, snip);
      if (w.video) {
        const play = document.createElement("a");
        play.href = w.video; play.target = "_blank"; play.rel = "noopener";
        play.textContent = "play from " + w.stamp;
        card.appendChild(play);
      } else {
        const note = document.createElement("span");
        note.className = "flag";
        note.textContent = "recording not on this machine - open it in the course's recording store at " + w.stamp;
        card.appendChild(note);
      }
      lecturesBox.appendChild(card);
    });
  }
  problemsBox.textContent = "";
  if (!res.problems || !res.problems.length) {
    emptyNote(problemsBox, res.problem_note || "No past question to show.");
  } else {
    res.problems.forEach(u => {
      const card = document.createElement("div");
      card.className = "card";
      const ref = document.createElement("span");
      ref.className = "ref";
      ref.textContent = u.reference;
      const snip = document.createElement("span");
      snip.className = "snip";
      snip.textContent = u.snippet;
      card.append(ref, snip);
      if (u.answers_in_text) {
        const flag = document.createElement("span");
        flag.className = "flag";
        flag.textContent = "this paper carries its worked answers in the text";
        card.appendChild(flag);
      }
      if (u.doc) {
        const open = document.createElement("a");
        open.href = u.doc; open.target = "_blank"; open.rel = "noopener";
        open.textContent = "open the question (p." + u.page + ")";
        card.appendChild(open);
      }
      if (u.memo && u.memo.doc) {
        const memo = document.createElement("a");
        memo.href = u.memo.doc; memo.target = "_blank"; memo.rel = "noopener";
        memo.textContent = u.memo.kind === "memorandum" ? "memorandum (p." + u.memo.page + ")" : u.memo.kind;
        card.appendChild(memo);
      } else if (!u.memo) {
        const none = document.createElement("span");
        none.className = "flag";
        none.textContent = "no memorandum for this paper";
        card.appendChild(none);
      }
      problemsBox.appendChild(card);
    });
  }
}

async function ask(question, image, fromImage) {
  const mine = addTurn("You", question || "(screenshot)", "student");
  const pending = addTurn(tutorLabel(chosen && chosen.label), "", "tutor");
  const bubble = pending.querySelector(".bubble");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  cursor.textContent = "\\u00a0";
  bubble.appendChild(cursor);
  send.disabled = true;

  try {
    const res = await fetch("/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question, image, history, from_image: fromImage || "",
                            grounded: groundedBox.checked,
                            model: chosen ? chosen.key : undefined})
    });
    const data = await readStream(res, piece => {
      bubble.insertBefore(document.createTextNode(piece), cursor);
      stick();
    }, rung => {
      // The card ran out of memory and the answer is being tried again with less of
      // the prompt. Whatever is in the bubble belongs to the attempt that failed, so
      // it goes: two attempts left end to end would read as one answer that
      // contradicts itself halfway through.
      while (bubble.firstChild && bubble.firstChild !== cursor) {
        bubble.removeChild(bubble.firstChild);
      }
      let note = pending.querySelector(".retrying");
      if (!note) {
        note = document.createElement("div");
        note.className = "retrying";
        pending.insertBefore(note, bubble);
      }
      note.textContent = "out of memory \\u2014 trying again with " + rung;
      stick();
    });
    cursor.remove();
    const retried = pending.querySelector(".retrying");
    if (retried) retried.remove();

    if (data.error) {
      pending.className = "turn failed";
      bubble.textContent = data.error;
    } else {
      // What the screenshot was read as, shown under the student's own turn. A
      // misread question answered confidently is the failure to avoid here, and the
      // student is the only one who can see that the reading is wrong.
      if (data.screenshot && data.screenshot.text) {
        const read = document.createElement("div");
        read.className = "shotread";
        const label = document.createElement("b");
        label.textContent = "read from your screenshot"
          + (data.screenshot.notes && data.screenshot.notes.length
             ? " \\u00b7 " + data.screenshot.notes.join("; ") : "");
        read.appendChild(label);
        read.appendChild(document.createTextNode(data.screenshot.text));
        mine.appendChild(read);
      }
      bubble.classList.add("rich");
      bubble.innerHTML = renderRich(data.text);
      // The server names the model that wrote it, which is the one to believe.
      if (data.model_label) pending.querySelector(".who").textContent = tutorLabel(data.model_label);
      const stats = document.createElement("div");
      stats.className = "stats";
      // An answer the handout lookup gave was not written by a model, and says so.
      const bits = data.handout_lookup
        ? ["looked up in the course handout", "no model asked"]
        : [data.latency_s.toFixed(1) + " s", data.output_tokens + " tokens out"];
      bits.push(data.cost_usd ? "$" + data.cost_usd.toFixed(4) : "no API charge");
      if (data.sources.length) bits.push(data.sources.length + " passages retrieved");
      bits.forEach(text => {
        const span = document.createElement("span");
        span.textContent = text;
        stats.appendChild(span);
      });
      pending.appendChild(stats);
      // An answer that only fitted once some of its grounding was given up says so.
      // Silently serving a thinner answer is the failure this whole path exists to
      // avoid: the student cannot see that the passages went missing, and that is
      // exactly the answer they should be checking hardest.
      if (data.note) {
        const warn = document.createElement("div");
        warn.className = "degraded";
        warn.textContent = data.note;
        pending.appendChild(warn);
      }

      showSources(data.sources);
      showResources(data.resources);
      // The screenshot's text goes into the history, not the word "(screenshot)",
      // so a follow-up that changes one number still knows what the problem was -
      // and it keeps the label saying where those words came from. Without the
      // label a later turn reads a misread formula as something the student typed
      // and checked, which is how a garbled question gets confidently answered.
      let said = question;
      if (data.screenshot && data.screenshot.text) {
        said = (question
                + "\\n[the following was read from a screenshot by OCR and "
                + "may contain mistakes the student did not make]\\n"
                + data.screenshot.text).trim();
      } else if (fromImage) {
        // A question the student confirmed still says where its words came from. The
        // caution it earns on later turns is the lighter one, but a subscript lost in
        // transcription and not spotted is still lost on turn three.
        said = question
          + "\\n[transcribed from a screenshot and checked by the student]";
      }
      history.push(["Student", said], ["Tutor", data.text]);
      turns += 1;
      spend += data.cost_usd;
      totalLine.textContent = turns + " turn(s) this session"
        + (spend ? ", $" + spend.toFixed(4) + " spent" : ", no API charge");
    }
  } catch (err) {
    cursor.remove();
    pending.className = "turn failed";
    bubble.textContent = "Could not reach the tutor: " + err;
  } finally {
    send.disabled = false;
    box.focus();
  }
}

// The answer arrives as server-sent events: a piece of text at a time, then one final
// message carrying the whole reply, its passages and its cost.
async function readStream(res, onPiece, onRestart) {
  // A request rejected before generation starts - no question, too long, an image
  // that could not be read - answers with plain JSON rather than an event stream.
  // Parsing that as a stream finds no events and loses the message, so the reason
  // the student needs is read off here instead.
  const kind = res.headers.get("content-type") || "";
  if (!res.ok || !res.body || !kind.includes("event-stream")) {
    try { return await res.json(); }
    catch (err) { return { error: "the tutor stopped before finishing" }; }
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", result = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\\n\\n")) >= 0) {
      const line = buffer.slice(0, cut).trim();
      buffer = buffer.slice(cut + 2);
      if (!line.startsWith("data:")) continue;
      const event = JSON.parse(line.slice(5));
      if (event.delta !== undefined) onPiece(event.delta);
      // A restart is not the final event: the attempt that failed is being replaced,
      // and the result still comes from whichever attempt finishes.
      else if (event.restart !== undefined) { if (onRestart) onRestart(event.restart); }
      else result = event;
    }
  }
  return result || { error: "the tutor stopped before finishing" };
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Mathematics is lifted out before markdown sees it, then typeset back in. Markdown
// otherwise reads the underscores in x_{k} as emphasis and the backslashes as escapes,
// and returns a formula with pieces missing. Same renderer as the grading page.
function renderRich(src) {
  const math = [];
  let text = String(src).replace(
    /\\\\\\[([\\s\\S]+?)\\\\\\]|\\$\\$([\\s\\S]+?)\\$\\$|\\\\\\(([\\s\\S]+?)\\\\\\)|\\$([^\\n$]+?)\\$/g,
    (m, a, b, c, d) => {
      const display = a !== undefined || b !== undefined;
      const tex = a !== undefined ? a : b !== undefined ? b : c !== undefined ? c : d;
      math.push({ tex, display });
      return "@@MATH" + (math.length - 1) + "@@";
    });
  // A box written outside the maths delimiters is a stated answer, so it reads as one.
  text = text.replace(/\\\\boxed\\{([^{}]*)\\}/g, "**$1**");
  text = text.replace(/</g, "&lt;");
  let html = window.marked ? marked.parse(text) : "<p>" + text.replace(/\\n/g, "<br>") + "</p>";
  return html.replace(/@@MATH(\\d+)@@/g, (m, i) => {
    const { tex, display } = math[Number(i)];
    if (!window.katex) return "<code>" + escapeHtml(tex) + "</code>";
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" });
    } catch (e) {
      return "<code>" + escapeHtml(tex) + "</code>";
    }
  });
}

// --- screenshots -----------------------------------------------------------
// An attached image is read straight away, by itself, and the transcription is put
// in the composer for the student to correct before anything is asked. The student
// is the only reader who can see both the image and the text, and a machine reading
// of mathematics is not good enough to use without them: the point of the two steps
// is that a lost subscript is a typing correction rather than a wrong answer.
let shot = null;            // the image, as a data URL, while it is being read
let shotEngine = "";        // the reader that produced the text now in the composer
let shotOriginal = "";      // that text as read, to tell a corrected one from an untouched one
let ocrReady = false;
let readerName = "a machine reader";
const attachRow = document.getElementById("attached");
const thumb = document.getElementById("thumb");
const attachName = document.getElementById("attachname");
const shotFile = document.getElementById("shotfile");
const shotHint = document.getElementById("shothint");

function hint(html, warn) {
  shotHint.innerHTML = html;
  shotHint.classList.toggle("warn", !!warn);
  shotHint.hidden = !html;
  fitFooter();
}

function holdShot(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = () => {
    shot = reader.result;
    thumb.src = shot;
    attachName.textContent = (file.name || "pasted image")
      + " \\u00b7 " + Math.round(file.size / 1024) + " KB";
    attachRow.classList.remove("warn");
    attachRow.hidden = false;
    fitFooter();
    readShot(file.name || "pasted image");
  };
  reader.readAsDataURL(file);
}

// Read the image on its own request. Nothing is asked of the tutor yet.
async function readShot(name) {
  if (!shot) return;
  if (!ocrReady) {
    attachRow.classList.add("warn");
    attachName.textContent = name + " \\u00b7 nothing on this machine can read it";
    hint("Fetch the vision reader with <b>ym07 fetch-reader</b>, or install Tesseract, "
         + "then attach it again. In the meantime, type the question out.", true);
    return;
  }
  attachRow.classList.add("reading");
  attachName.textContent = name + " \\u00b7 reading it with " + readerName + "\\u2026";
  hint("");
  send.disabled = true;
  try {
    const res = await fetch("/read", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({image: shot})
    });
    const data = await res.json();
    attachRow.classList.remove("reading");
    if (!res.ok || data.error) {
      attachRow.classList.add("warn");
      attachName.textContent = name + " \\u00b7 could not be read";
      hint(escapeHtml(data.error || "the reader failed") + " \\u2014 type the question instead.",
           true);
      return;
    }
    shotEngine = data.engine_name || data.engine;
    // Into the composer, where it can be edited, rather than into a panel where it
    // can only be looked at. Anything the student had already typed is kept above
    // it - "why is my working wrong?" is the question, and the screenshot is the
    // working, so overwriting one with the other throws away the half that says
    // what is being asked.
    const typed = box.value.trim() && box.value !== shotOriginal ? box.value.trim() : "";
    shotOriginal = typed ? typed + "\\n\\n" + data.text : data.text;
    box.value = shotOriginal;
    grow();
    box.focus();
    box.setSelectionRange(box.value.length, box.value.length);
    attachName.textContent = name + " \\u00b7 read by " + shotEngine
      + " in " + data.latency_s + "s";
    const notes = (data.notes || []).length
      ? " \\u00b7 " + data.notes.map(escapeHtml).join("; ") : "";
    hint("<b>Check the text above against the picture</b> before you ask \\u2014 readers "
         + "lose subscripts and misread Greek letters. Fix anything wrong, then press Ask."
         + notes);
  } catch (err) {
    attachRow.classList.remove("reading");
    attachRow.classList.add("warn");
    hint("Could not reach the reader: " + escapeHtml(String(err)), true);
  } finally {
    send.disabled = false;
  }
}

function clearShot() {
  shot = null;
  shotEngine = "";
  shotOriginal = "";
  attachRow.hidden = true;
  attachRow.classList.remove("warn", "reading");
  thumb.removeAttribute("src");
  shotFile.value = "";
  hint("");
  fitFooter();
}

document.getElementById("dropshot").addEventListener("click", clearShot);
document.getElementById("attach").addEventListener("click", () => shotFile.click());
shotFile.addEventListener("change", () => holdShot(shotFile.files[0]));

document.addEventListener("paste", event => {
  const item = [...(event.clipboardData?.items || [])]
    .find(i => i.type.startsWith("image/"));
  if (item) { event.preventDefault(); holdShot(item.getAsFile()); }
});

["dragenter", "dragover"].forEach(name =>
  document.addEventListener(name, event => {
    event.preventDefault();
    document.body.classList.add("dragging");
  }));
["dragleave", "drop"].forEach(name =>
  document.addEventListener(name, event => {
    event.preventDefault();
    document.body.classList.remove("dragging");
  }));
document.addEventListener("drop", event => holdShot(event.dataTransfer?.files[0]));

document.getElementById("composer").addEventListener("submit", event => {
  event.preventDefault();
  const question = box.value.trim();
  if ((!question && !shot) || send.disabled) return;
  // Once the image has been read, what gets sent is the text in the composer - the
  // student's version of it - and the name of the reader that drafted it. The image
  // itself only goes up if the reading failed, which is the one case where the
  // server still has to try.
  const readOk = !!shotEngine;
  const sending = readOk ? null : shot;
  const engine = readOk ? shotEngine : "";
  box.value = "";
  grow();
  clearShot();
  ask(question, sending, engine);
});

box.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("composer").requestSubmit();
  }
});

fitFooter();

fetch("/state").then(r => r.json()).then(state => {
  models = state.models || [];
  if (models.length > 1) {
    models.forEach(m => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.key = m.key;
      b.setAttribute("role", "radio");
      b.title = m.detail || m.model;
      const name = document.createElement("span");
      name.textContent = m.label;
      const sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = m.hosted ? "hosted" : "on this PC";
      b.append(name, sub);
      b.onclick = () => choose(m.key, true);
      modelsBox.appendChild(b);
    });
    modelsBox.hidden = false;
  }
  let saved = null;
  try { saved = localStorage.getItem("ym07.model"); } catch (e) { /* none kept */ }
  if (models.length) {
    choose(models.some(m => m.key === saved) ? saved : state.default, false);
  } else {
    document.getElementById("model").textContent = state.model;
  }
  // Attaching always works, whether or not this machine can read the image: a
  // dead button looks like a broken feature, and the student cannot tell the two
  // apart. If there is no engine, the attachment row says so, where it is read.
  ocrReady = !!state.ocr;
  readerName = state.ocr_engine_name || state.ocr_engine || "a machine reader";
  const attachBtn = document.getElementById("attach");
  attachBtn.title = ocrReady
    ? "Attach a screenshot, or paste one - read by " + readerName
    : "Nothing on this machine can read an image yet";
  groundedBox.disabled = !state.corpus_loaded;
  groundedBox.checked = state.corpus_loaded;
  document.getElementById("grounding").textContent = state.corpus_loaded
    ? state.passages + " passages of course material"
    : "no course material loaded";
});
</script>
</body>
</html>
"""


def _engine_label(engine: str) -> str:
    """The reader's name as the interface shows it to a student."""
    from .ocr import ENGINE_NAMES

    return ENGINE_NAMES.get(engine, engine)


def _json(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    """Serve a local file, honouring a single byte range so a video can seek.

    The recordings and PDFs never leave this machine through git; they are served here
    so that a reference opens at its second or its page in the student's own browser.
    """
    size = path.stat().st_size
    start, end = 0, size - 1
    rng = handler.headers.get("Range")
    partial = False
    if rng and rng.startswith("bytes="):
        spec = rng[6:].split(",")[0].strip()
        a, _, b = spec.partition("-")
        try:
            if a:
                start = int(a)
                end = int(b) if b else size - 1
            elif b:
                start = max(0, size - int(b))
            partial = True
        except ValueError:
            start, end, partial = 0, size - 1, False
        if start >= size:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.end_headers()
            return
        end = min(end, size - 1)
    length = end - start + 1
    handler.send_response(206 if partial else 200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    if partial:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    with path.open("rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            chunk = fh.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            try:
                handler.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            remaining -= len(chunk)


def build_handler(*, config: CourseConfig, retriever, client=None, model_id: str = "",
                  top_k: int = 4, use_course_prompt: bool = True,
                  resources: ResourceFinder | None = None,
                  pipelines: list[Pipeline] | None = None, lookup=None):
    page = PAGE.replace("__CODE__", config.code)

    # How the course runs, read out of the course handout once at startup. The tutor
    # answers a question about the course from these rather than from a search: the
    # values are the document's, and a changed handout changes them. See
    # handout.read_facts for why they are a labelled list and not the pages themselves.
    handout_facts = (read_facts(retriever.chunks, config.handout)
                     if retriever is not None else ())

    # One GPU serves one generation at a time. Without this a second browser tab
    # does not queue, it allocates its own KV cache alongside the first and the
    # card runs out of memory - the same reason the batch runner drops to one
    # worker for the local arm.
    generating = threading.Lock()

    # The models on the switch. Called with one client, as the interface checks and the
    # single-condition demo do, the tutor has one model and no switch. A hosted model
    # gets a lock of its own, so one question at a time still goes to the API, but it
    # never queues behind the card.
    if not pipelines:
        label, detail = model_label(model_id)
        pipelines = [Pipeline(key=model_id or "default", label=label, detail=detail,
                              client=client, model_id=model_id, top_k=top_k,
                              use_course_prompt=use_course_prompt,
                              lectures=offers_lectures(model_id), lookup=lookup)]
    for pipe in pipelines:
        pipe.lock = threading.Lock() if pipe.hosted else generating
    by_key = {pipe.key: pipe for pipe in pipelines}

    @contextmanager
    def on_the_card(pipe):
        """One answer at a time per model, and the card to itself for a local one.

        Found live in E30 (16 October): after a screenshot, the reader stayed on the card
        beside the local model, whose next answer did not fit; Windows paged the overflow
        into system RAM rather than failing, and every local answer for the rest of the
        session took about fourteen minutes instead of under one. The batch runner the
        results come from never had both on the card and released what each answer
        reserved; this does the same. Inside the card's lock, so a screenshot cannot be
        read between the two.
        """
        with pipe.lock:
            if pipe.hosted:
                yield
                return
            from .ocr import free_vlm, vlm_loaded

            if vlm_loaded():
                free_vlm()
            try:
                yield
            finally:
                degrade.free_card()

    class TutorHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ym07"

        def log_message(self, fmt, *args):  # noqa: A002 - signature is the base class's
            pass  # the console belongs to the server's own progress, not to request logs

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/state":
                from .ocr import available as ocr_available
                from .ocr import engines as ocr_engines

                ocr_ok, ocr_engine = ocr_available()
                _json(self, {
                    "code": config.code,
                    "model": pipelines[0].model_id,
                    "models": [pipe.public() for pipe in pipelines],
                    "default": pipelines[0].key,
                    "ocr": ocr_ok,
                    "ocr_engine": ocr_engine,
                    "ocr_engine_name": _engine_label(ocr_engine),
                    "ocr_engines": [_engine_label(name) for name in ocr_engines()],
                    "corpus_loaded": retriever is not None,
                    "passages": len(retriever) if retriever is not None else 0,
                    "top_k": pipelines[0].top_k,
                    "resources": resources.describe() if resources is not None else None,
                })
                return
            # A recording or a PDF on this machine, opened at a second or a page. Only
            # files the finder knows about are served, by their bare name: nothing in
            # the path can walk anywhere else.
            from urllib.parse import unquote

            for prefix, lookup, kind in (("/video/", "video_path", "video/mp4"),
                                         ("/doc/", "doc_path", "application/pdf")):
                if self.path.startswith(prefix) and resources is not None:
                    name = unquote(self.path[len(prefix):].split("#")[0].split("?")[0])
                    found = getattr(resources, lookup)(Path(name).name) if name else None
                    if found is None:
                        _json(self, {"error": "not found"}, status=404)
                    else:
                        _send_file(self, found, kind)
                    return
            _json(self, {"error": "not found"}, status=404)

        def _read_image(self, payload: dict):
            """Transcribe an uploaded image, or send the student the reason it failed.

            Returns the OcrResult, or None once a response has already gone out.
            """
            import base64

            from .ocr import OcrFailed, OcrUnavailable, extract

            try:
                raw = payload["image"]
                if "," in raw[:64]:    # strip a data: URL prefix if the browser sent one
                    raw = raw.split(",", 1)[1]
                result = extract(base64.b64decode(raw, validate=False))
            except (OcrUnavailable, OcrFailed) as exc:
                _json(self, {"error": str(exc)}, status=400)
                return None
            except Exception as exc:
                _json(self, {"error": f"the screenshot could not be read: {exc}"}, status=400)
                return None
            if result.empty:
                _json(self, {"error": "no text could be read from that image. If it is a "
                                      "photograph or handwriting, type the question instead."},
                      status=400)
                return None
            return result

        def do_POST(self) -> None:
            # Reading an image is its own request, answered before any question is
            # asked. The transcription goes back to the browser, which puts it in the
            # composer for the student to correct; only then does a question get sent.
            # Splitting it this way is what makes the reading correctable at all - on
            # one round trip the student sees the transcription and the answer to it
            # at the same time, too late to fix either.
            if self.path == "/read":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_REQUEST_BYTES:
                        self.rfile.read(min(length, MAX_REQUEST_BYTES))
                        _json(self, {"error": "that image is larger than the tutor will "
                                              "read."}, status=413)
                        return
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError) as exc:
                    _json(self, {"error": f"bad request: {exc}"}, status=400)
                    return
                if not payload.get("image"):
                    _json(self, {"error": "no image"}, status=400)
                    return
                # The reader and the tutor share one card, so a transcription waits
                # for any generation in flight rather than allocating beside it.
                with generating:
                    result = self._read_image(payload)
                if result is None:
                    return
                _json(self, {"text": result.text, "engine": result.engine,
                             "engine_name": _engine_label(result.engine),
                             "notes": result.notes, "latency_s": result.latency_s,
                             "width": result.width, "height": result.height})
                return

            if self.path not in ("/ask", "/stream"):
                _json(self, {"error": "not found"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_REQUEST_BYTES:
                    self.rfile.read(min(length, MAX_REQUEST_BYTES))
                    _json(self, {"error": "that is longer than the tutor will read. "
                                          "Send a question rather than a document."}, status=413)
                    return
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as exc:
                _json(self, {"error": f"bad request: {exc}"}, status=400)
                return

            # Which model answers: the page sends the key of the one on the switch, and
            # anything else - a script, an older page - gets the first, the default.
            pipe = by_key.get(str(payload.get("model") or "")) or pipelines[0]

            question = (payload.get("question") or "").strip()
            if len(question) > MAX_QUESTION_CHARS:
                _json(self, {"error": f"that question is {len(question):,} characters; the tutor "
                                      f"takes up to {MAX_QUESTION_CHARS:,}."}, status=413)
                return

            # Where the question came from, when it came from an image. The browser
            # normally reads the image at /read first and sends the transcription here
            # as ordinary text, naming the engine that produced it - so the question
            # the tutor answers is the one the student confirmed.
            from_image = (payload.get("from_image") or "").strip()[:80]

            # An image posted straight to /ask or /stream is still read here. Nothing in
            # the browser does that any more; it is the path a script takes, and the
            # fallback if the two-step read fails. A reading nobody has checked carries
            # the stronger caution, which is why it stays a separate argument.
            screenshot = ""
            shot_meta = None
            if payload.get("image"):
                result = self._read_image(payload)
                if result is None:
                    return
                screenshot = result.text
                shot_meta = {"text": result.text, "engine": result.engine,
                             "engine_name": _engine_label(result.engine),
                             "notes": result.notes}

            if not question and not screenshot:
                _json(self, {"error": "no question"}, status=400)
                return
            if not question:
                question = "Help me with the question in this screenshot."

            # Retrieval searches what the screenshot said as well as what was typed: a
            # student who pastes an aliasing question and types nothing still needs the
            # aliasing passages.
            # The search text, not the text anybody reads. A transcription comes back
            # as LaTeX, which is right for the student and right for the model, and
            # wrong for the index: the corpus is text pulled out of PDFs and spells
            # mathematics in Unicode, with no LaTeX command anywhere in it, so
            # searching `\delta` against a passage that says `δ` matches on nothing.
            # Without this the most precisely transcribed question retrieves worst.
            from .ocr import plain as searchable

            # The same function the suites use, so the browser and the results cannot
            # disagree about what a question is answered from. A question about how the
            # course runs gets the handout, whole; anything else gets a ranked search,
            # made with the screenshot's words folded in and its mathematics rewritten
            # the way the corpus spells it.
            grounded = bool(payload.get("grounded")) and retriever is not None and pipe.retrieval
            # The history arrives from the browser rather than being held here, so
            # the server keeps no per-visitor state and a reload starts clean.
            history = [(who, text) for who, text in (payload.get("history") or [])]
            # A follow-up is searched with the student's previous message (E33,
            # retrieval.conversation_query); a first question is searched as it was.
            search_text = searchable(f"{conversation_query(question, history)}\n{screenshot}".strip())
            # The handout lookup, where this model has one (E27): a question plainly
            # asking for one of the handout's facts is answered from it, and its wider
            # test of what is a course question decides what the model is given when
            # it is not. Never for a screenshot, whose question is in the image.
            about = looked_up = None
            if pipe.lookup is not None and grounded and not screenshot:
                looked_up = pipe.lookup.reply(question)
                about = True if looked_up is not None else pipe.lookup.about_the_course(question)
            hits = passages_for(
                question, retriever if grounded else None, code=config.code,
                top_k=pipe.top_k, handout=config.handout,
                query=search_text, about=about)
            if looked_up is not None:
                hits = handout_passages(retriever, config.handout)
            about_course = about_the_course(question) if about is None else about

            # The lecture minutes and past questions for the same search text, found
            # with the indexes E24 and E25 measured and withheld below their thresholds.
            # They are looked up before generation so the cards arrive with the answer
            # and cost the student nothing to wait for.
            extras = None
            if grounded and resources is not None:
                try:
                    extras = resources.find(search_text, lectures=pipe.lectures).as_dict()
                    if not pipe.lectures:
                        extras["lecture_note"] = LECTURES_ELSEWHERE
                except Exception as exc:  # a broken index must not take the answer down
                    print(f"  resources failed: {exc}")

            sources = [{"tag": f"S{n}", "source": hit.chunk.citation(), "score": hit.score,
                        "doc": (f"/doc/{hit.chunk.source}#page={hit.chunk.page}"
                                if resources is not None and hit.chunk.page
                                and resources.doc_path(hit.chunk.source) else None)}
                       for n, hit in enumerate(hits, start=1)]

            if looked_up is not None:
                print(f"  answered from the handout by lookup: {', '.join(looked_up.fields)}")
                reply = {"text": looked_up.text, "model": pipe.key, "model_label": pipe.label,
                         "latency_s": 0.0, "cost_usd": 0.0, "output_tokens": 0,
                         "sources": sources, "note": "",
                         "handout_lookup": list(looked_up.fields),
                         "screenshot": None, "resources": extras}
                if self.path == "/stream":
                    self._send_whole(reply)
                else:
                    _json(self, reply)
                return

            if self.path == "/stream":
                self._stream(pipe, question, hits, history, grounded, sources, screenshot,
                             shot_meta, from_image, extras, about_course)
                return

            system = system_prompt(config, grounded=grounded) if pipe.use_course_prompt else ""

            def attempt(kept_hits, kept_history, budget):
                with degrade.budgeted(pipe.client, budget):
                    return pipe.client.answer(
                        system=system,
                        user=user_prompt(question, kept_hits,
                                         kept_history[-MAX_HISTORY_TURNS * 2:] or None,
                                         facts=handout_facts if about_course else (),
                                         handout_pages=bool(kept_hits) and all(
                                             h.chunk.source == config.handout
                                             for h in kept_hits),
                                         screenshot=screenshot,
                                         from_image=from_image,
                                         assessed=is_assessed(question, said_so_far(
                                             question, kept_history, screenshot))),
                        model=pipe.model_id,
                        max_tokens=config.model.max_tokens,
                        effort=config.model.effort,
                    )

            with on_the_card(pipe):
                served = degrade.with_fallback(attempt, hits=hits, history=history,
                                               free=degrade.free_card)
            answer: Answer = served.answer
            # A citation the passage does not support is removed before the student sees
            # it, against the passages actually sent rather than those retrieved. The
            # same function the suites use. See ym07.citations.
            if answer is not None and not answer.error:
                cleaned, dropped, _ = citations.checked(answer.text, served.hits)
                if dropped:
                    answer = replace(answer, text=cleaned)
                    print(f"  removed unsupported citation(s) {dropped}")

            if served.exhausted:
                # The student gets the explanation rather than the allocator's message.
                # Which is not the same as pretending it worked: it still arrives as a
                # failure, and the console still carries the real error.
                print(f"  out of memory on every rung: {answer.error}")
                _json(self, {"error": served.note})
                return
            if answer.error:
                _json(self, {"error": answer.error})
                return

            _json(self, {
                "text": answer.text,
                "model": pipe.key,
                "model_label": pipe.label,
                "latency_s": answer.latency_s,
                "cost_usd": answer.cost_usd,
                "output_tokens": answer.output_tokens,
                # Only the passages the answering attempt was actually given, so the
                # citations under the reply are the ones it could have used.
                "sources": sources[:len(served.hits)],
                "note": served.note,
                "screenshot": shot_meta,
                "resources": extras,
            })

        def _send_whole(self, reply: dict) -> None:
            """A reply that is already complete, sent as the stream's events, so the page
            renders it exactly as it renders a streamed one."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for event in ({"delta": reply["text"]}, {"done": True, **reply}):
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _stream(self, pipe, question, hits, history, grounded, sources,
                    screenshot="", shot_meta=None, from_image="", extras=None,
                    about_course=None) -> None:
            """The same answer, sent as it is written.

            A long derivation takes the better part of a minute on one card, and a
            student watching an empty box cannot tell that from a crash. The prompt,
            the decoding and the seed are the batch runner's, so what appears here is
            what the results tables were computed from.
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send(event: dict) -> bool:
                try:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError):
                    return False  # the tab was closed mid-answer

            system = system_prompt(config, grounded=grounded) if pipe.use_course_prompt else ""
            gone: list[bool] = []  # the tab was closed mid-answer

            def attempt(kept_hits, kept_history, budget):
                course = about_the_course(question) if about_course is None else about_course
                user = user_prompt(question, kept_hits,
                                   kept_history[-MAX_HISTORY_TURNS * 2:] or None,
                                   facts=handout_facts if course else (),
                                   handout_pages=bool(kept_hits) and all(
                                             h.chunk.source == config.handout
                                             for h in kept_hits),
                                   screenshot=screenshot, from_image=from_image,
                                   assessed=is_assessed(question, said_so_far(
                                       question, kept_history, screenshot)))
                reply = None
                with degrade.budgeted(pipe.client, budget):
                    if hasattr(pipe.client, "stream"):
                        for piece in pipe.client.stream(system=system, user=user, model=pipe.model_id,
                                                   max_tokens=config.model.max_tokens,
                                                   effort=config.model.effort):
                            if isinstance(piece, str):
                                if not send({"delta": piece}):
                                    gone.append(True)
                                    return _Closed
                            else:
                                reply = piece
                    else:
                        # Hosted models answer in one piece; the page handles either.
                        reply = pipe.client.answer(system=system, user=user, model=pipe.model_id,
                                              max_tokens=config.model.max_tokens,
                                              effort=config.model.effort)
                        if not reply.error and not send({"delta": reply.text or ""}):
                            gone.append(True)
                            return _Closed
                return reply if reply is not None else _Nothing

            # Running out of memory happens in the prefill, before a token is emitted,
            # so a retry almost always starts from an empty bubble. Almost: the cache
            # also grows as it decodes, so a long answer can fail halfway through one.
            # The page is told to clear what it has either way, because a second attempt
            # appended to the first would read as one answer that contradicts itself.
            with on_the_card(pipe):
                served = degrade.with_fallback(
                    attempt, hits=hits, history=history, free=degrade.free_card,
                    on_retry=lambda rung: send({"restart": rung.label}))

            if gone:
                return
            answer = served.answer
            if served.exhausted:
                print(f"  out of memory on every rung: {getattr(answer, 'error', '')}")
                send({"error": served.note})
                return
            if answer is None or answer.error:
                send({"error": answer.error if answer else "the model produced nothing"})
                return
            # The citation check needs the whole answer, so on the streamed path it runs
            # here rather than before the first token. The `done` event carries the full
            # text and the browser renders that over what it streamed, so a tag the
            # passage does not support is visible while the answer is being written and
            # gone once it is finished. Worth saying plainly: this is later than the
            # batch runner, which never shows it at all.
            final = answer.text
            dropped: list[int] = []
            if not answer.error:
                final, dropped, _ = citations.checked(answer.text, served.hits)
                if dropped:
                    print(f"  removed unsupported citation(s) {dropped}")
            send({"done": True, "text": final, "model": pipe.key, "model_label": pipe.label,
                  "latency_s": answer.latency_s,
                  "cost_usd": answer.cost_usd, "output_tokens": answer.output_tokens,
                  "sources": sources[:len(served.hits)], "note": served.note,
                  "screenshot": shot_meta, "resources": extras})

    return TutorHandler


def serve(
    *,
    config: CourseConfig,
    retriever,
    client=None,
    model_id: str = "",
    top_k: int = 4,
    use_course_prompt: bool = True,
    condition_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    out=print,
    resources: ResourceFinder | None = None,
    extra: tuple[str, ...] = (),
    pipelines: list[Pipeline] | None = None,
) -> int:
    handler = build_handler(config=config, retriever=retriever, client=client,
                            model_id=model_id, top_k=top_k,
                            use_course_prompt=use_course_prompt, resources=resources,
                            pipelines=pipelines)
    httpd = ThreadingHTTPServer((host, port), handler)

    first_k = pipelines[0].top_k if pipelines else top_k
    grounding = (
        f"{len(retriever)} passages, top {first_k} per question"
        if retriever is not None else
        "NO COURSE MATERIAL - run `ym07 ingest` for the grounded tutor"
    )
    from .ocr import ENGINE_NAMES
    from .ocr import engines as ocr_engines

    ready = ocr_engines()
    reader = (" then ".join(ENGINE_NAMES.get(name, name) for name in ready) if ready
              else "none - `ym07 fetch-reader` installs one")

    out(f"{config.code} tutor on http://{host}:{port}")
    if pipelines and len(pipelines) > 1:
        for n, pipe in enumerate(pipelines):
            notes = ["selected when the page opens" if n == 0 else "one click away on the switch"]
            if not pipe.lectures:
                notes.append("no lecture minutes")
            out(f"  {'models:' if n == 0 else '':10s} {pipe.label}: {pipe.key} ({pipe.model_id}); "
                + "; ".join(notes))
    else:
        if condition_name:
            out(f"  condition: {condition_name}")
        out(f"  model:     {pipelines[0].model_id if pipelines else model_id}")
    out(f"  grounding: {grounding}")
    out(f"  reads:     {reader}")
    out(f"  resources: {resources.describe() if resources is not None else 'none loaded'}")
    for line in extra:
        out(f"  {line}")
    out("  Ctrl-C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        out("\nstopped")
    finally:
        httpd.server_close()
    return 0


# The platform as the master brief describes it (13 October): Opus 5 inside the
# delivered pipeline, selected when the page opens because the rule of 11 October made it
# the default, and the local R1 system one click away, free and offline, without lecture
# minutes (see offers_lectures). Since 14 October the local system is the delivered R1
# system with the handout lookup in front of it (E27, by Sanjan's decision after the
# result) and, by E29's registered rule, the lookup's marks rule; see RESEARCH_DECISIONS,
# 14 October.
PLATFORM = ("A3-opus-pipeline", "A-r1-facts-marks")

# The share of the card one process may take, as the batch runner takes it (`ym07 stress
# run --cap-gb 15.5`). Without it an answer that does not fit is paged into system RAM by
# Windows instead of failing into the fallback ladder: E30's local answer took 557 s.
CARD_CAP_GB = 15.5


def run_server(args) -> int:
    import sys

    from .conditions import HOSTED_PIPELINE, OSS_MODEL, get_condition
    from .config import load_config
    from .retrieval import build_retriever

    config = load_config(args.config)

    # Serving a named condition is the whole point: it is the difference between
    # demonstrating the system the report measured and demonstrating something
    # that merely resembles it. The tuned adapter was trained on at most four
    # passages, so handing it the config's eight would be a different system. With
    # several conditions each keeps its own settings; the first is the default.
    if getattr(args, "platform", False):
        names = list(PLATFORM)
    else:
        names = ([args.condition] if args.condition else []) + list(getattr(args, "also", None) or [])
    names = list(dict.fromkeys(names))
    conditions = [get_condition(name) for name in names]

    if not args.dry_run and any(c.resolved_provider(config) == "local" for c in conditions):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    min(1.0, CARD_CAP_GB * 1e9 / torch.cuda.get_device_properties(0).total_memory), 0)
        except ImportError:
            pass

    retrieves = (any(c.use_retrieval for c in conditions) if conditions else True) \
        and not args.no_retrieval
    retriever = build_retriever(args.chunks, config.retrieval.method) if retrieves else None
    if retriever is None and retrieves and Path(args.chunks).exists():
        print(f"warning: {args.chunks} is empty - serving ungrounded.", file=sys.stderr)

    # The lecture and past-question indexes, when their material exists. Built with the
    # same retrieval method as the passages, so the three cannot differ in method.
    resources = None
    if retrieves and not getattr(args, "no_resources", False):
        try:
            resources = ResourceFinder(method=config.retrieval.method)
        except Exception as exc:
            print(f"warning: resource indexes not loaded: {exc}", file=sys.stderr)

    pipelines: list[Pipeline] = []
    extra: list[str] = []

    # One lookup per version in use, built over the shared retriever (E27, E28).
    lookups: dict = {}
    if retriever is not None and any(c.handout_lookup for c in conditions):
        from .coursefacts import build as build_lookup

        for key in sorted({(int(c.handout_lookup), bool(c.handout_marks))
                           for c in conditions if c.handout_lookup}):
            lookups[key] = build_lookup(retriever, config.handout, version=key[0], marks=key[1])
        extra.append("handout:   course facts looked up for "
                     + ", ".join(f"{c.name} (version {int(c.handout_lookup)}"
                                 f"{', marks rule' if c.handout_marks else ''})"
                                 for c in conditions if c.handout_lookup))

    def add(key, client, model_id, provider, top_k, use_course_prompt, retrieval,
            with_lookup=0, with_marks=False):
        hosted = provider != "local"
        label, detail = model_label(model_id)
        if hosted and not args.dry_run:
            client = SpendCap(client, max_usd=args.max_usd)
            extra.append(f"spending:  {label} at most ${args.max_usd:.2f} this session; "
                         "each answer's cost is shown under it")
        pipelines.append(Pipeline(key=key, label=label, detail=detail, client=client,
                                  model_id=model_id, top_k=top_k,
                                  use_course_prompt=use_course_prompt, retrieval=retrieval,
                                  lectures=offers_lectures(model_id), hosted=hosted,
                                  lookup=lookups.get((int(with_lookup or 0), bool(with_marks)))))

    for n, condition in enumerate(conditions):
        client = build_client_for(condition, config, dry_run=args.dry_run)
        if condition.name in HOSTED_PIPELINE:
            # These were measured through ym07.hosted, which puts the deterministic
            # assessed-work gate around the hosted model. The bare client under the same
            # name would demonstrate a system the results do not describe.
            from .hosted import GatedClient

            client = GatedClient(client, gate=condition.gate)
        model_id = (args.model if n == 0 and args.model else condition.resolved_model(config))
        add(condition.name, client, model_id, condition.resolved_provider(config),
            condition.resolved_top_k(config), condition.use_course_prompt,
            condition.use_retrieval, with_lookup=int(condition.handout_lookup),
            with_marks=bool(condition.handout_marks))

    if not pipelines:
        # No condition named: the config's model, as `ym07 serve` always did.
        provider = args.provider or config.model.provider
        model_id = args.model or (OSS_MODEL if provider == "local" else config.model.id)
        add(model_id, build_client(args.dry_run, provider, config), model_id, provider,
            config.retrieval.top_k, True, True)

    return serve(
        config=config,
        retriever=retriever,
        pipelines=pipelines,
        condition_name=", ".join(names) or None,
        host=args.host,
        port=args.port,
        resources=resources,
        extra=tuple(extra),
    )
