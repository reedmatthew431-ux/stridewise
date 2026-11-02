"""Skeptic layer: a pre-generated "challenge this" note for every solid
or emerging association card.

Design (same discipline as the narrative layer):
- Spark sees ONE card plus the profile aggregates — never raw rows.
- Spark writes 2–3 sentences: the strongest innocent explanation for
  the pattern, and what evidence would settle it.
- A deterministic validator checks the note before it is stored:
  every number must trace to the card or the aggregates, causal,
  medical, prescriptive and prediction language is linted out (the
  narrative lint list plus the banned causal connectors), and the
  note may not name a tier other than the card's own.
- One retry with the failures fed back, then the card is recorded as
  having NO skeptic note. Unvalidated text is never stored.
- App calls use the STANDARD tier only (model muse-spark-1.3).

Run: python -m engine.skeptic <factpack.json> <out.json>
     [--spend-log PATH] [--max-calls N]
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path

from .narrative import (_LINTS, _TIER_LABELS, _TIER_WORDS, MODEL,
                        _fact_numbers, _number_traces, _text_numbers,
                        append_spend_log, call_spark)

# The narrative system prompt bans these connectors by instruction;
# the skeptic validator enforces them as lint too.
_EXTRA_LINTS = [
    ("causal language", r"\bdue to\b"),
    ("causal language", r"\bso that\b"),
    ("causal language", r"\bcaused by\b"),
]
_ALL_LINTS = _LINTS + _EXTRA_LINTS

_SYSTEM = """You are the skeptic for Stridewise, a personal training-data
validation app. Statistics have already decided what is true. Your job
is to challenge ONE pattern card: give the strongest innocent
explanation for the pattern, and say what evidence would settle
whether the innocent explanation or the pattern is the better reading.

Write exactly 2 or 3 sentences of plain text. No JSON, no markdown,
no heading.

Hard rules:
- Every number you write must appear in the card or the aggregates
  given below, copied as displayed (display rounding is fine). Never
  invent, add, or combine numbers.
- No causal language. Never use causal connectors: "because",
  "since", "so that", "due to", "leads to", "causes", "caused by",
  and "results in" are banned. Describe associations only — the
  innocent explanation is a possibility to check, not a claim.
- No medical language, no diagnosis, no advice, and no instruction
  to the athlete. "What would settle it" is about EVIDENCE (more
  nights, a wider range of weeks, a missing field), never about
  what the athlete should do.
- No predictions about the athlete's future results or feelings.
- If you name a tier, it must be the tier of this card.
- Name a concrete alternative reading (for example a schedule
  artifact, a third variable moving with both sides, a measurement
  quirk, or a narrow slice of the calendar doing the work), not a
  generic "correlation is not causation"."""


def build_prompt(pack, card, failures=None):
    view = {k: card.get(k) for k in (
        "question", "question_text", "tier", "effect", "likely_range",
        "n", "binding_rule", "sentence", "methods_note")}
    user = (
        "CARD:\n" + json.dumps(view, indent=1)
        + "\n\nAGGREGATES:\n" + json.dumps(pack.get("aggregates", {}))
    )
    if failures:
        user += (
            "\n\nYour previous draft was rejected by the validator for "
            "these reasons:\n" + "\n".join(f"- {f}" for f in failures)
            + "\nWrite a corrected note that fixes every one of them."
        )
    return _SYSTEM, user


def _sentence_count(text):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip())
             if p.strip()]
    return len(parts)


def validate_note(text, pack, card):
    """Return a list of failure strings. Empty list means valid."""
    failures = []
    if not isinstance(text, str) or not text.strip():
        return ["empty note"]
    allowed = _fact_numbers(card) | _fact_numbers(pack.get("aggregates", {}))
    for tok in _text_numbers(text):
        if not _number_traces(tok, allowed):
            failures.append(
                f"number {tok:g} does not trace to the card or the "
                f"aggregates")
    low = text.lower()
    for label, pat in _ALL_LINTS:
        if re.search(pat, low):
            failures.append(f"{label} ({pat})")
    dem = card.get("demotion") or {}
    own_tiers = {card.get("tier"),
                 dem.get("was") if isinstance(dem, dict) else None,
                 dem.get("from") if isinstance(dem, dict) else None}
    own_tiers.discard(None)
    mentioned = set()
    for t in _TIER_WORDS:
        labels = (t.replace("_", " "),) + _TIER_LABELS.get(t, ())
        if any(lbl in low for lbl in labels):
            mentioned.add(t)
    upgraded = mentioned - own_tiers
    if upgraded:
        failures.append(
            f"states tier(s) {sorted(upgraded)} but the card's tier "
            f"is {card.get('tier')}")
    count = _sentence_count(text)
    if count < 2 or count > 3:
        failures.append(f"note has {count} sentence(s); needs 2 or 3")
    return failures


def attempt_note(pack, card, call, failures=None):
    """One API call. Returns (note_or_None, usage, failures)."""
    system, user = build_prompt(pack, card, failures)
    content, usage, finish = call(system, user)
    usage = dict(usage, model=MODEL,
                 purpose=f"skeptic note {card['question']}")
    if not isinstance(content, str) or not content.strip():
        return None, usage, [
            f"empty completion (finish_reason={finish}): the model "
            f"returned no visible text"]
    text = content.strip()
    fails = validate_note(text, pack, card)
    if fails:
        return None, usage, fails
    return text, usage, []


def generate_note(pack, card, call, max_attempts=2):
    """Retry-then-none orchestration. Returns (record, usages) where
    record is {"status": "ok", "note": ...} or
    {"status": "no_note", "note": None, "final_failures": [...]}."""
    usages = []
    failures = None
    for attempt in range(1, max_attempts + 1):
        note, usage, failures = attempt_note(pack, card, call, failures)
        usage = dict(usage, purpose=(
            f"skeptic note {card['question']} attempt {attempt}"))
        usages.append(usage)
        if note is not None:
            return {"status": "ok", "note": note,
                    "attempts": attempt}, usages
    return {"status": "no_note", "note": None, "attempts": max_attempts,
            "final_failures": failures or []}, usages


def eligible_cards(pack):
    return [c for c in pack.get("cards", [])
            if c.get("tier") in ("solid", "emerging")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stridewise skeptic notes")
    ap.add_argument("factpack")
    ap.add_argument("out")
    ap.add_argument("--spend-log")
    args = ap.parse_args(argv)
    pack = json.loads(Path(args.factpack).read_text())
    call = functools.partial(call_spark, max_tokens=2500,
                             reasoning_effort="low")
    notes = {}
    usages = []
    for card in eligible_cards(pack):
        rec, us = generate_note(pack, card, call)
        notes[card["question"]] = rec
        usages.extend(us)
    Path(args.out).write_text(json.dumps(notes, indent=2))
    if args.spend_log:
        append_spend_log(args.spend_log, usages)
    print(json.dumps({
        "notes": {q: r["status"] for q, r in notes.items()},
        "calls": len(usages), "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
