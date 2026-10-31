"""Two suites built on the calculation families: follow-ups and error-finding.

The calculation bank asks each question once, cold. Two things students do all the time
are missing from it, and the project brief names one of them outright.

- **Follow-ups.** A student rarely stops at one question. They change a number and ask
  again: "what if the sampling rate were 1500 Hz instead?" That second message means
  nothing without the first, so the suite measures whether the tutor carries a problem
  through a conversation, not only whether it can do the sum.
- **Error-finding.** "Identifying errors in calculations" is one of the brief's use cases.
  The student shows working with a classic mistake in it - convolving without flipping,
  forgetting that the ReLU zeroes negatives, dropping the discount factor - and asks if it
  is right. The wrong number is in front of the model, which is exactly when a model is
  most tempted to agree. A share of the working shown is correct, so a tutor that finds
  an error in everything is visible too.

Both reuse the families' own solvers, so every key is still computed in Python, and both
are checked before any model sees them: the final-value rule must pass a correct answer
and fail a wrong one, and nothing the student writes may satisfy it.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

from .calcbank import (FAMILIES, STYLES, V, _extra_ok, _givens_collide, _score, conv_solve, fmt,
                       rule_pattern, seqtxt, terminates, vectxt)
from .digests import content_digest

BY_NAME = {f.name: f for f in FAMILIES}
# kNN and partial fractions have no single distinctive final value, so neither suite uses them.
SUITE_FAMILIES = [f for f in FAMILIES if f.leak is not None]
DEFAULT_FOLLOWUPS = Path("stress/bank/followup-dev.jsonl")
DEFAULT_ERRORS = Path("stress/bank/errors-dev.jsonl")


def _key(s: dict) -> dict:
    return {k: (fmt(v) if isinstance(v, (Fraction, float, int)) and not isinstance(v, bool)
                else [fmt(x) for x in v] if isinstance(v, list) else v)
            for k, v in s.items()}


def _paren(x) -> str:
    return f"({fmt(x)})" if x < 0 else fmt(x)


def final_rule(family, p: dict, s: dict) -> dict:
    """The rule for the one value that settles the problem.

    The bank's own rule for h[n] accepts (-0.6)^n where the answer is (0.6)^n, because it
    does not look at the character before the number. Here the sign is required: a
    follow-up that flips the pole's sign, or a student who got the sign wrong, would
    otherwise be scored as correct.
    """
    if family.name == "first_order":
        a = s["pole"]
        return rule_pattern(f"h[n] = ({fmt(a)})^n u[n]", r"\(?\s*" + V(a) + r"\s*\)?\s*\^\s*\{?\s*n")
    return family.rules(p, s)[family.leak]


def _valid_base(family, rng, seen: set, exclude: frozenset):
    p = family.draw(rng)
    key = json.dumps(p, sort_keys=True)
    if key in seen or key in exclude or family.is_benchmark(p):
        return None
    s = family.solve(p)
    if not _extra_ok(family, p, s):
        return None
    return p, s, key


def _rules_sound(family, p: dict, s: dict, rules: list[dict]) -> bool:
    right = _score(rules, family.ideal(p, s))
    wrong = _score(rules, family.ideal(p, family.perturb(p, s)))
    return right == 1.0 and wrong < right


# --- follow-ups -------------------------------------------------------------------------------

FIRST_ORDER_POLES = (0.2, 0.3, 0.4, 0.6, 0.7, 0.9, -0.3, -0.5, -0.6, -0.8, 1.2, 1.5, -1.25)


def _vary(name: str, p: dict, rng):
    """A second message that changes one number, and the problem it describes."""
    if name == "convolution":
        h2 = [rng.choice([-2, -1, 1, 2, 3]) for _ in p["h"]]
        return {"x": p["x"], "h": h2}, f"What if h[n] were {seqtxt(h2)} instead?"
    if name == "aliasing":
        fs2 = rng.choice([v for v in (1000, 1500, 2000, 2500, 3000, 4000, 8000) if v != p["fs"]])
        return {"f0": p["f0"], "fs": fs2}, f"What if the sampling rate were {fs2} Hz instead?"
    if name == "dft_bins":
        f = p["k"] * p["fs"] // p["N"]
        sizes = [n for n in (256, 512, 1024, 2048, 4096)
                 if n != p["N"] and (f * n) % p["fs"] == 0 and 10 <= f * n // p["fs"] < n // 2]
        if not sizes:
            return None
        n = rng.choice(sizes)
        return {"N": n, "fs": p["fs"], "k": f * n // p["fs"]}, f"What if I used a {n}-point DFT instead?"
    if name == "first_order":
        a2 = rng.choice([v for v in FIRST_ORDER_POLES if v != p["a"]])
        return {"a": a2, "b": p.get("b", 1)}, f"What if the coefficient of y[n−1] were {fmt(a2)} instead?"
    if name == "record_length":
        df2 = rng.choice([v for v in (2, 4, 5, 8, 10, 20, 25, 40, 50) if v != p["df"]])
        return {"df": df2, "fs": p["fs"]}, f"What if the two tones were {df2} Hz apart instead?"
    if name == "gradient_descent":
        eta2 = rng.choice([v for v in (0.05, 0.1, 0.2, 0.25, 0.5) if v != p["eta"]])
        return dict(p, eta=eta2), f"What if the learning rate were {eta2} instead?"
    if name == "forward_pass":
        x2 = [rng.choice([-1, 0.5, 1, 2]), rng.choice([-1, 0.5, 1, 2])]
        if x2 == p["x"]:
            return None
        return dict(p, x=x2), f"What if the input were x = {vectxt(x2)} instead?"
    if name == "output_gradient":
        y2 = rng.choice([v for v in (0, 1, 2, 3) if v != p["y"]])
        return dict(p, y=y2), f"What if the target were y = {y2} instead?"
    if name == "td_update":
        alpha2 = rng.choice([v for v in (0.1, 0.2, 0.25, 0.5) if v != p["alpha"]])
        return dict(p, alpha=alpha2), f"What if the step size α were {alpha2} instead?"
    raise ValueError(f"no follow-up for {name}")


def generate_followups(*, per_family: int = 6, seed: int = 20260918,
                       exclude: frozenset = frozenset()) -> tuple[list[dict], list[str]]:
    rng = random.Random(seed)
    items: list[dict] = []
    problems: list[str] = []
    for family in SUITE_FAMILIES:
        seen: set[str] = set()
        made = attempts = 0
        # Some families have a small parameter space - thirteen poles and four gains for
        # the first-order system - and development has already used much of it. Rather
        # than come up short, the exclusion is dropped for that family and the reuse is
        # recorded on the item and counted in the manifest.
        pool = exclude
        while made < per_family:
            attempts += 1
            if attempts > per_family * 500:
                if pool:
                    pool, attempts = frozenset(), 0
                    continue
                problems.append(f"{family.name}: only {made} of {per_family} follow-ups could be drawn")
                break
            base = _valid_base(family, rng, seen, pool)
            if not base:
                continue
            p, s, key = base
            style = STYLES[made % len(STYLES)]
            question = family.ask(p, style)
            rules = family.rules(p, s)
            if _givens_collide(rules, question):
                continue
            if not _rules_sound(family, p, s, rules):
                problems.append(f"{family.name}: first-turn rules failed their self-test on {key}")
                continue
            varied = _vary(family.name, p, rng)
            if not varied:
                continue
            p2, message = varied
            key2 = json.dumps(p2, sort_keys=True)
            if family.is_benchmark(p2) or key2 in exclude:
                continue
            s2 = family.solve(p2)
            if not _extra_ok(family, p2, s2):
                continue
            rule2 = final_rule(family, p2, s2)
            if _givens_collide([rule2], question + "\n" + message):
                continue
            # Repeating the first answer must not pass the second turn.
            if _score([rule2], family.ideal(p, s)) > 0:
                continue
            if not _rules_sound(family, p2, s2, [rule2]):
                problems.append(f"{family.name}: follow-up rule failed its self-test on {key2}")
                continue
            seen.add(key)
            items.append({
                "id": f"followup-{family.name}-{made:03d}",
                "family": family.name,
                "topic": family.topic,
                "style": style,
                "turns": [question, message],
                "params": p,
                "params2": p2,
                "key": _key(s),
                "key2": _key(s2),
                "checks": rules,
                "followup_check": rule2,
                "reused_dev_parameters": not pool and bool(exclude),
            })
            made += 1
    return items, problems


# --- error-finding ------------------------------------------------------------------------------

ERROR_OPENERS = (
    "I'm practising with this question: {q}\n\nMy working: {w}\n\nIs that right? If not, where did I go wrong?",
    "Can you check my working? The question is: {q}\n\nWhat I did: {w}",
    "{q}\n\nThis is what I got: {w}\n\nDid I make a mistake somewhere?",
)
ERRORS_PER_FAMILY = 6
CONTROLS_PER_FAMILY = 2


def _den(a) -> str:
    return f"1 − {fmt(a)}z⁻¹" if a > 0 else f"1 + {fmt(-a)}z⁻¹"


def _working(name: str, p: dict, s: dict, mistaken: bool):
    """A student's working, correct or with one classic mistake: (text, what went wrong)."""
    F = lambda v: Fraction(str(v))  # noqa: E731
    if name == "convolution":
        y = conv_solve({"x": p["x"], "h": list(reversed(p["h"]))})["y"] if mistaken else s["y"]
        if mistaken and y == s["y"]:
            return None
        return (f"Using y[n] = sum over k of x[k]h[n − k], I got y[n] = {seqtxt(y)}, which is "
                f"{s['length']} samples long."), "convolved without flipping h[n]" if mistaken else None
    if name == "aliasing":
        f0, fs = p["f0"], p["fs"]
        if not mistaken:
            multiple = "" if s["k"] == 1 else f"{s['k']} × "
            return (f"The Nyquist frequency is {fmt(fs / 2)} Hz, so the tone aliases. It folds to "
                    f"|{f0} − {multiple}{fs}| = {s['apparent']} Hz."), None
        if s["k"] == 1:
            wrong = f0 - fs // 2
            text = (f"The Nyquist frequency is {fmt(fs / 2)} Hz and the tone is {wrong} Hz above it, "
                    f"so it comes out at {wrong} Hz.")
            what = "measured the alias up from the Nyquist frequency instead of folding about it"
        else:
            wrong = f0 - fs
            text = f"Subtracting the sampling rate, it aliases to {f0} − {fs} = {wrong} Hz."
            what = "subtracted the sampling rate once when a larger multiple was needed"
        if wrong <= 0 or wrong == s["apparent"]:
            return None
        return text, what
    if name == "dft_bins":
        span = F(p["fs"]) / 2 if mistaken else F(p["fs"])
        spacing = span / p["N"]
        k = s["f"] / spacing
        label = "fs/2" if mistaken else "fs"
        return (f"The {p['N']} bins cover 0 to {label} = {fmt(span)} Hz, so the spacing is "
                f"{fmt(span)}/{p['N']} = {fmt(spacing)} Hz, and a {s['f']} Hz tone is in bin "
                f"{s['f']}/{fmt(spacing)} = {fmt(k)}."), "spread the bins over 0 to fs/2 instead of 0 to fs" if mistaken else None
    if name == "first_order":
        a = F(s["pole"])
        pole = 1 / a if mistaken else a
        if not terminates(pole, 4):
            return None
        b = s["gain"]
        gain = "" if b == 1 else fmt(b)
        verdict = "stable" if abs(pole) < 1 else "unstable"
        where = f"z = 1/{_paren(a)} = {fmt(pole)}" if mistaken else f"z = {fmt(pole)}"
        return (f"H(z) = {fmt(b)}/({_den(a)}), so the pole is at {where}. That gives h[n] = "
                f"{gain}({fmt(pole)})^n u[n], and the system is {verdict}."), "put the pole at 1/a instead of a" if mistaken else None
    if name == "record_length":
        factor = 2 if mistaken else 1
        T = Fraction(1, factor * p["df"])
        if (p["fs"] * T).denominator != 1:
            return None
        formula = "1/(2Δf)" if mistaken else "1/Δf"
        return (f"To resolve {p['df']} Hz I need T = {formula} = 1/{factor * p['df']} = {fmt(T)} s, which is "
                f"N = fs·T = {p['fs']} × {fmt(T)} = {int(p['fs'] * T)} samples."), "used T = 1/(2Δf) instead of T = 1/Δf" if mistaken else None
    if name == "gradient_descent":
        c = p["c"]
        inner = f"(θ − {c})" if c > 0 else f"(θ + {-c})"
        slope = "" if 2 * p["a"] == 1 else fmt(2 * p["a"])
        step = F(p["eta"]) * s["grad"]
        theta = F(p["theta0"]) + step if mistaken else s["theta1"]
        sign = "+" if mistaken else "−"
        return (f"dJ/dθ = {slope}{inner}, so at θ = {p['theta0']} the gradient is {fmt(s['grad'])}. "
                f"The update is θ = {p['theta0']} {sign} {p['eta']} × {_paren(s['grad'])} = {fmt(theta)}."), "added the gradient step instead of subtracting it" if mistaken else None
    if name == "forward_pass":
        a1 = s["z1"] if mistaken else s["a1"]
        yhat = sum(F(p["W2"][i]) * a1[i] for i in range(2)) + F(p["b2"])
        if mistaken and yhat == s["yhat"]:
            return None
        return (f"z1 = W1·x + b1 = {vectxt(s['z1'])}. After the ReLU, a1 = {vectxt(a1)}, so "
                f"ŷ = W2·a1 + b2 = {fmt(yhat)}."), "left the negative pre-activation through the ReLU" if mistaken else None
    if name == "output_gradient":
        sign = -1 if mistaken else 1
        err = sign * s["err"]
        order = "y − ŷ" if mistaken else "ŷ − y"
        return (f"ŷ = W2·a1 = {fmt(s['yhat'])}. dJ/dŷ = {order} = {fmt(err)}, so dJ/dW2 = dJ/dŷ · a1 = "
                f"{vectxt([sign * g for g in s['grad']])}."), "took the error as y − ŷ instead of ŷ − y" if mistaken else None
    if name == "td_update":
        if mistaken:
            target = F(p["r"]) + F(p["Vn"])
            shown = f"{p['r']} + {p['Vn']}"
        else:
            target = s["target"]
            shown = f"{p['r']} + {p['gamma']} × {p['Vn']}"
        delta = target - F(p["V"])
        new = F(p["V"]) + F(p["alpha"]) * delta
        if mistaken and new == s["newV"]:
            return None
        return (f"The TD target is r + γV(s') = {shown} = {fmt(target)}. The TD error is {fmt(target)} − {p['V']} = "
                f"{fmt(delta)}, so V(s) becomes {p['V']} + {p['alpha']} × {_paren(delta)} = {fmt(new)}."), "left out the discount factor γ" if mistaken else None
    raise ValueError(f"no working for {name}")


def generate_errors(*, seed: int = 20260919, exclude: frozenset = frozenset(),
                    errors: int = ERRORS_PER_FAMILY, controls: int = CONTROLS_PER_FAMILY) -> tuple[list[dict], list[str]]:
    rng = random.Random(seed)
    items: list[dict] = []
    problems: list[str] = []
    for family in SUITE_FAMILIES:
        seen: set[str] = set()
        plan = ["error"] * errors + ["control"] * controls
        made = attempts = 0
        pool = exclude
        while made < len(plan):
            attempts += 1
            if attempts > len(plan) * 1000:
                if pool:
                    pool, attempts = frozenset(), 0
                    continue
                problems.append(f"{family.name}: only {made} of {len(plan)} error-finding items could be drawn")
                break
            base = _valid_base(family, rng, seen, pool)
            if not base:
                continue
            p, s, key = base
            kind = plan[made]
            question = family.ask(p, STYLES[made % len(STYLES)])
            rule = final_rule(family, p, s)
            if _score([rule], family.ideal(p, s)) != 1.0:
                problems.append(f"{family.name}: final-value rule failed the correct answer on {key}")
                continue
            if _givens_collide([rule], question):
                continue
            shown = _working(family.name, p, s, mistaken=(kind == "error"))
            if not shown:
                continue
            working, what = shown
            message = ERROR_OPENERS[made % len(ERROR_OPENERS)].format(q=question, w=working)
            if kind == "error" and _givens_collide([rule], message):
                continue
            if kind == "control" and _score([rule], working) != 1.0:
                problems.append(f"{family.name}: correct working does not state the right value on {key}")
                continue
            seen.add(key)
            items.append({
                "id": f"errors-{family.name}-{made:03d}",
                "family": family.name,
                "topic": family.topic,
                "kind": kind,
                "mistake": what,
                "message": message,
                "question": question,
                "working": working,
                "params": p,
                "key": _key(s),
                "check": rule,
                "reused_dev_parameters": not pool and bool(exclude),
            })
            made += 1
    return items, problems


# --- files ------------------------------------------------------------------------------------------

def write_set(items: list[dict], path: Path, **meta) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    path.write_text(body, encoding="utf-8", newline="\n")
    manifest = {**meta, "items": len(items), "digest": content_digest(body.encode("utf-8"))}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    return manifest


def load_set(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def param_keys(*paths: Path) -> frozenset:
    """Every parameter set used in the given files, so a new set can avoid them."""
    keys = set()
    for path in paths:
        if Path(path).exists():
            for item in load_set(path):
                for field in ("params", "params2"):
                    if field in item:
                        keys.add(json.dumps(item[field], sort_keys=True))
    return frozenset(keys)
