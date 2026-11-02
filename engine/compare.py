"""Across-the-boards comparison: which findings replicate, which
contrast, which are one athlete's alone.

Spark reads the four profiles' FACT PACKS -- card-level results
only (effect, likely range, tier, n per question), never raw rows --
and writes the comparison. The discipline matches the rest of the
app:

- Every number in a summary must trace to the comparison input for
  that question (display rounding allowed, as in the narrative
  validator); causal/medical/prescriptive/prediction language is
  linted out; a tier may be named for a question only when some
  profile in that entry actually holds it.
- Two semantic checks the numbers make possible: a "replicated"
  entry must have at least two live findings (solid or emerging)
  pointing the same way and none pointing the other way; a
  "contrasts" entry must have live findings in both directions.
  An "individual" entry must cite a profile that actually holds a
  live finding on that question.
- Profile labels are used verbatim and are the only names Spark
  ever sees: Matt Reed by name (his own board); the other three
  profiles carry descriptor labels only.
- One retry with the failures fed back, then a deterministic
  fallback builds the same three lists straight from the packs
  with template summaries. The stored file names its source.

Run: python -m engine.compare <factpack.json> [<factpack.json> ...]
     --out comparison.json [--spend-log PATH]
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path

from .narrative import (_LINTS, _TIER_LABELS, _TIER_WORDS, MODEL,
                        _fact_numbers, _number_traces, _parse_content,
                        _text_numbers, append_spend_log, call_spark)
from .skeptic import _EXTRA_LINTS

_ALL_LINTS = _LINTS + _EXTRA_LINTS
_LIVE = ("solid", "emerging")
_PROFILE_COUNTS = {2.0, 3.0, 4.0}


def _sign(v):
    if v is None:
        return 0
    return 1 if v > 0 else (-1 if v < 0 else 0)


# -------------------------------------------------------------- assembly

def assemble_comparison(packs):
    """Card-level comparison input from a list of fact packs.
    Returns {"labels": [...], "questions": {qid: {...}}}."""
    labels, questions = [], {}
    for pack in packs:
        label = pack.get("athlete", {}).get("display_label")
        labels.append(label)
        for card in pack.get("cards", []):
            qid = card["question"]
            entry = questions.setdefault(qid, {
                "question_text": card.get("question_text"),
                "profiles": {}})
            entry["profiles"][label] = {
                "effect": card.get("effect"),
                "range": card.get("likely_range"),
                "tier": card.get("tier"),
                "n": card.get("n"),
            }
    return {"labels": labels, "questions": questions}


def _live_profiles(qentry):
    """label -> entry for profiles with a live (solid/emerging)
    finding on this question."""
    return {label: e for label, e in qentry["profiles"].items()
            if e.get("tier") in _LIVE and e.get("effect") is not None}


# ---------------------------------------------------------------- prompt

_SYSTEM = """You are the comparison analyst for Stridewise, a personal
training-data validation app. You are given card-level results from
several athletes' boards: for each question, each profile's effect,
likely range, evidence tier, and paired count. You never see raw
data. Statistics have already decided what is true on each board;
your job is to compare the boards honestly.

Write a JSON object with exactly three keys:

- "replicated": findings that point the same way on at least two
  boards. A finding counts only when its tier is "solid" or
  "emerging" on that board.
- "contrasts": questions where live findings (solid or emerging)
  point in OPPOSITE directions on different boards.
- "individual": live findings that belong to one board alone --
  no other board has a live finding on that question, or one
  board's finding stands clearly apart. Each entry names the
  profile it belongs to.

Entry shapes:
- replicated / contrasts: {"question": "A1", "summary": "..."}
- individual: {"profile_label": "...", "question": "A9",
  "summary": "..."}

Hard rules:
- Refer to profiles ONLY by the labels given in the input, copied
  verbatim. Never invent a name, and never describe a profile by
  anything but its label.
- Every number in a summary must be copied from that question's
  input values (effects, likely ranges, counts) -- display rounding
  is fine. Never combine numbers across profiles into a new figure.
- You may name a tier for a question only when a profile in that
  entry actually holds that tier on that question. In an individual
  entry, name only that profile's own tier.
- No causal language: never "because", "since", "so that",
  "due to", "leads to", "causes", "caused by", or "results in",
  and never write that one thing improves, prevents, or guarantees
  another. Describe associations and their directions only.
- No medical language, no advice, no predictions.
- Do not manufacture entries: a question with no live finding on
  any board belongs in none of the three lists. Keep each summary
  to one or two plain sentences.

Respond with ONLY the JSON object. No markdown fences, no
commentary."""


def build_prompt(comp, failures=None):
    user = "COMPARISON INPUT:\n" + json.dumps(comp)
    if failures:
        user += (
            "\n\nYour previous draft was rejected by the validator "
            "for these reasons:\n"
            + "\n".join(f"- {f}" for f in failures)
            + "\nWrite a corrected comparison that fixes every one "
              "of them.")
    return _SYSTEM, user


# ------------------------------------------------------------- validation

def _mentioned_tiers(text):
    low = text.lower()
    mentioned = set()
    for t in _TIER_WORDS:
        labels = (t.replace("_", " "),) + _TIER_LABELS.get(t, ())
        if any(lbl in low for lbl in labels):
            mentioned.add(t)
    return mentioned


def _check_summary(text, allowed_numbers, allowed_tiers, where):
    failures = []
    if not isinstance(text, str) or not text.strip():
        return [f"{where}: summary has no text"]
    for tok in _text_numbers(text):
        if not _number_traces(tok, allowed_numbers):
            failures.append(
                f"{where}: number {tok:g} does not trace to the "
                f"question's input values")
    low = text.lower()
    for label, pat in _ALL_LINTS:
        if re.search(pat, low):
            failures.append(f"{where}: {label} ({pat})")
    extra = _mentioned_tiers(text) - set(allowed_tiers)
    if extra:
        failures.append(
            f"{where}: states tier(s) {sorted(extra)} not held on "
            f"this question")
    return failures


def validate_comparison(out, comp):
    """Return a list of failure strings. Empty means valid."""
    failures = []
    if not isinstance(out, dict):
        return ["output is not a JSON object"]
    questions = comp["questions"]
    labels = set(comp["labels"])
    for name in ("replicated", "contrasts", "individual"):
        entries = out.get(name)
        if entries is None:
            failures.append(f"missing section: {name}")
            continue
        if not isinstance(entries, list):
            failures.append(f"section {name} is not a list")
            continue
        for i, entry in enumerate(entries):
            where = f"{name}[{i}]"
            if not isinstance(entry, dict):
                failures.append(f"{where}: entry is not an object")
                continue
            qid = entry.get("question")
            if qid not in questions:
                failures.append(f"{where}: unknown question {qid!r}")
                continue
            qentry = questions[qid]
            allowed = _fact_numbers(qentry) | _PROFILE_COUNTS
            live = _live_profiles(qentry)
            signs = {_sign(e["effect"]) for e in live.values()}
            if name == "replicated":
                pos = sum(1 for e in live.values()
                          if _sign(e["effect"]) > 0)
                neg = sum(1 for e in live.values()
                          if _sign(e["effect"]) < 0)
                if not ((pos >= 2 and neg == 0)
                        or (neg >= 2 and pos == 0)):
                    failures.append(
                        f"{where}: {qid} has no same-direction pair "
                        f"of live findings to replicate")
                tiers = {e["tier"] for e in qentry["profiles"].values()}
                failures += _check_summary(
                    entry.get("summary"), allowed, tiers, where)
            elif name == "contrasts":
                if not (1 in signs and -1 in signs):
                    failures.append(
                        f"{where}: {qid} has no opposite-direction "
                        f"live findings to contrast")
                tiers = {e["tier"] for e in qentry["profiles"].values()}
                failures += _check_summary(
                    entry.get("summary"), allowed, tiers, where)
            else:  # individual
                label = entry.get("profile_label")
                if label not in labels:
                    failures.append(
                        f"{where}: unknown profile label {label!r}")
                    continue
                mine = qentry["profiles"].get(label) or {}
                if mine.get("tier") not in _LIVE:
                    failures.append(
                        f"{where}: {label} holds no live finding on "
                        f"{qid}")
                failures += _check_summary(
                    entry.get("summary"), allowed,
                    {mine.get("tier")}, where)
    return failures


# --------------------------------------------------------------- fallback

def _fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else "?"


def _frag(label, e):
    rng = e.get("range") or [None, None]
    lo = rng[0] if len(rng) > 0 else None
    hi = rng[1] if len(rng) > 1 else None
    return (f"{label}: effect {_fmt(e.get('effect'))} (likely range "
            f"{_fmt(lo)} to {_fmt(hi)}, n {_fmt(e.get('n'))}, "
            f"{e.get('tier')})")


def template_comparison(comp):
    """Deterministic comparison straight from the packs. Every
    number in every summary is copied from the input."""
    replicated, contrasts, individual = [], [], []
    covered = set()
    for qid, qentry in comp["questions"].items():
        live = _live_profiles(qentry)
        if not live:
            continue
        pos = [l for l, e in live.items() if _sign(e["effect"]) > 0]
        neg = [l for l, e in live.items() if _sign(e["effect"]) < 0]
        text = qentry.get("question_text") or qid
        if (len(pos) >= 2 and not neg) or (len(neg) >= 2 and not pos):
            frags = "; ".join(_frag(l, live[l]) for l in live)
            replicated.append({
                "question": qid,
                "summary": f"{text} — the same direction on "
                           f"{len(live)} boards: {frags}."})
            covered.add(qid)
        elif pos and neg:
            frags = "; ".join(_frag(l, live[l]) for l in live)
            contrasts.append({
                "question": qid,
                "summary": f"{text} — live findings point in "
                           f"opposite directions: {frags}."})
            covered.add(qid)
    for label in comp["labels"]:
        mine = []
        for qid, qentry in comp["questions"].items():
            if qid in covered:
                continue
            e = qentry["profiles"].get(label) or {}
            if e.get("tier") in _LIVE and e.get("effect") is not None:
                others_live = [l for l in _live_profiles(qentry)
                               if l != label]
                if not others_live:
                    mine.append((qid, qentry, e))
        mine.sort(key=lambda t: (0 if t[2]["tier"] == "solid" else 1,
                                 -(t[2]["n"] or 0)))
        for qid, qentry, e in mine[:3]:
            text = qentry.get("question_text") or qid
            individual.append({
                "profile_label": label, "question": qid,
                "summary": f"{text} — a live finding on this board "
                           f"alone: {_frag(label, e)}."})
    return {"replicated": replicated, "contrasts": contrasts,
            "individual": individual}


# ------------------------------------------------------------ orchestration

def generate_comparison(packs, call=None, max_attempts=2):
    """Returns (comparison_dict, usages)."""
    if call is None:
        call = functools.partial(call_spark, max_tokens=6000,
                                 reasoning_effort="low")
    comp = assemble_comparison(packs)
    usages = []
    failures = None
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        system, user = build_prompt(comp, failures)
        content, usage, finish = call(system, user)
        usages.append(dict(usage, model=MODEL, purpose=(
            f"across-boards comparison attempt {attempts}")))
        if not isinstance(content, str) or not content.strip():
            failures = [f"empty completion (finish_reason={finish})"]
            continue
        try:
            out = _parse_content(content)
        except (ValueError, TypeError) as exc:
            failures = [f"output was not usable JSON: {exc}"]
            continue
        failures = validate_comparison(out, comp)
        if not failures:
            return _wrap(packs, out, "spark", attempts, []), usages
    out = template_comparison(comp)
    return _wrap(packs, out, "template", attempts,
                 failures or []), usages


def _wrap(packs, out, source, attempts, failures):
    return {
        "generated_on": max((p.get("generated_on") or "")
                            for p in packs),
        "window_anchor": packs[0].get("window_anchor") if packs
        else None,
        "model": MODEL,
        "source": source,
        "attempts": attempts,
        "validation": {"passed": source == "spark",
                       "final_failures": failures},
        "labels": [p.get("athlete", {}).get("display_label")
                   for p in packs],
        "replicated": out.get("replicated", []),
        "contrasts": out.get("contrasts", []),
        "individual": out.get("individual", []),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stridewise across-the-boards comparison")
    ap.add_argument("factpacks", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spend-log")
    args = ap.parse_args(argv)
    packs = [json.loads(Path(p).read_text()) for p in args.factpacks]
    comp, usages = generate_comparison(packs)
    Path(args.out).write_text(json.dumps(comp, indent=2))
    if args.spend_log:
        append_spend_log(args.spend_log, usages)
    print(json.dumps({
        "source": comp["source"], "attempts": comp["attempts"],
        "replicated": len(comp["replicated"]),
        "contrasts": len(comp["contrasts"]),
        "individual": len(comp["individual"]),
        "validation_failures": comp["validation"]["final_failures"],
        "calls": len(usages), "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
