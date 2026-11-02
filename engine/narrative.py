"""Narrative layer: Muse Spark writes the Analyst Report, statistics
decide what is true.

Design (locked, see the Spark-layer brief):
- The fact pack is the ONLY thing Spark ever sees. Never raw rows.
- Spark returns STRUCTURED JSON where every claim carries the citation
  IDs of the facts it rests on (claims are born tagged).
- A deterministic validator checks every claim before anyone sees it:
  every number must trace to a cited fact's value within display
  rounding, causal and medical language is linted out, untagged claims
  are rejected, and tiers may not be upgraded or demotions softened.
- One retry with the failures fed back, then a template fallback built
  from the cards' own sentences. The user never sees unvalidated prose.
- App calls use the STANDARD tier only (model muse-spark-1.3). The
  contributor tier is banned here: it permits training on traffic and
  fact packs are personal data.

Calls go through the custom.meta-model-api connector. The credential
is never handled by this module: dynamic_credentials places an hsurr
surrogate on the request and the real key is swapped in on egress to
api.meta.ai only.

Run: python -m engine.narrative <factpack.json> <out.json>
     [--spend-log PATH] [--max-attempts 2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MODEL = "muse-spark-1.3"
CONNECTOR = "custom.meta-model-api"
API_BASE = "https://api.meta.ai/v1"
API_HOSTS = ["api.meta.ai"]
_CRED_BIN = "/opt/muse/skills/custom-connector-setup/bin"

SECTIONS = ("headline_findings", "ruled_out", "evidence_needs")

# ---------------------------------------------------------------------------
# Citation index: every fact gets an ID and the set of numbers a claim
# citing it is allowed to state.
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _text_numbers(text):
    out = []
    for tok in _NUM_RE.findall(str(text)):
        tok = tok.replace(",", "")
        if tok in ("", "+", "-", "."):
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _fact_numbers(obj):
    """All numbers a fact carries: numeric fields at any depth, numbers
    inside its own text fields, absolute values, and minute-to-hour
    style unit conversions for larger values."""
    vals = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            vals.add(float(o))
        elif isinstance(o, str):
            for v in _text_numbers(o):
                vals.add(v)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    extra = set()
    for v in vals:
        extra.add(abs(v))
        if abs(v) >= 20:
            extra.add(v / 60.0)
            extra.add(v * 60.0)
    vals |= extra
    if isinstance(obj, dict) and obj.get("field_size") \
            and obj.get("matched_finishers") is not None:
        vals.add(100.0 * obj["matched_finishers"] / obj["field_size"])
    return vals


def build_index(pack):
    """Map citation ID -> {kind, tier, numbers, demotion}."""
    index = {}
    for card in pack.get("cards", []):
        dem = card.get("demotion") or {}
        index[card["question"]] = {
            "kind": "card",
            "tier": card.get("tier"),
            "demotion": bool(card.get("demotion")),
            "demotion_from": dem.get("from") if isinstance(dem, dict)
            else None,
            "numbers": _fact_numbers(card),
        }
    for rc in pack.get("race_cards", []):
        cid = f"{rc['question']}@{rc['date']}"
        index[cid] = {
            "kind": "race",
            "tier": None,
            "demotion": False,
            "numbers": _fact_numbers(rc),
        }
    index["AGG"] = {
        "kind": "aggregate",
        "tier": None,
        "demotion": False,
        "numbers": _fact_numbers(pack.get("aggregates", {})),
    }
    return index


# ---------------------------------------------------------------------------
# Language lint
# ---------------------------------------------------------------------------

_CAUSAL = [
    r"\bcaus\w*\b", r"\bleads? to\b", r"\bled to\b", r"\bresults? in\b",
    r"\bbecause\b", r"\bmakes? you\b", r"\bwill make\b", r"\bimprov\w*\b",
    r"\bboost\w*\b", r"\bprevent\w*\b", r"\bguarantee\w*\b",
    r"\bensures?\b", r"\bdrives?\b", r"\bhurts?\b", r"\bruins?\b",
]
_MEDICAL = [
    r"\bdiagnos\w*\b", r"\btreat\w*\b", r"\bsymptom\w*\b",
    r"\bdisorder\b", r"\bapnea\b", r"\bclinical\w*\b", r"\bprescri\w*\b",
    r"\bmedical advice\b", r"\binjur\w*\b",
]
_PRESCRIPTIVE = [
    r"\byou should\b", r"\btry to\b", r"\bmake sure you\b",
    r"\baim to\b", r"\baim for\b", r"\bbe sure to\b",
]
_PREDICTION = [
    r"\bwill (run|sleep|feel|perform|race)\b",
    r"\bexpect (you|him|her|them) to\b",
    r"\bon track (for|to)\b", r"\blikely to (run|sleep|feel|perform)\b",
]
_LINTS = (
    [("causal language", p) for p in _CAUSAL]
    + [("medical language", p) for p in _MEDICAL]
    + [("prescriptive language", p) for p in _PRESCRIPTIVE]
    + [("prediction about the athlete", p) for p in _PREDICTION]
)
_TIER_WORDS = ("solid", "emerging", "no_pattern", "insufficient_data")
_TIER_LABELS = {"no_pattern": ("no pattern",), "insufficient_data":
                ("not enough data", "insufficient")}


def _number_traces(token, allowed):
    """True when token could be a display rounding of an allowed value."""
    for v in allowed:
        diff = abs(token - v)
        if diff <= 0.51:
            return True
        if v != 0 and diff <= 0.011 * abs(v):
            return True
    return False


def validate_narrative(narrative, index):
    """Return a list of failure strings. Empty list means valid."""
    failures = []
    if not isinstance(narrative, dict):
        return ["output is not a JSON object"]
    sections = narrative.get("sections", narrative)
    for name in SECTIONS:
        claims = sections.get(name)
        if claims is None:
            failures.append(f"missing section: {name}")
            continue
        if not isinstance(claims, list):
            failures.append(f"section {name} is not a list")
            continue
        for i, claim in enumerate(claims):
            where = f"{name}[{i}]"
            if not isinstance(claim, dict) \
                    or not isinstance(claim.get("text"), str) \
                    or not claim["text"].strip():
                failures.append(f"{where}: claim has no text")
                continue
            text = claim["text"]
            cites = claim.get("cites")
            if not isinstance(cites, list) or not cites:
                failures.append(f"{where}: untagged claim (no cites)")
                continue
            unknown = [c for c in cites if c not in index]
            if unknown:
                failures.append(f"{where}: unknown citations {unknown}")
                continue
            facts = [index[c] for c in cites]
            allowed = set()
            for f in facts:
                allowed |= f["numbers"]
            for tok in _text_numbers(text):
                if not _number_traces(tok, allowed):
                    failures.append(
                        f"{where}: number {tok:g} does not trace to "
                        f"cited facts {cites}")
            low = text.lower()
            for label, pat in _LINTS:
                if re.search(pat, low):
                    failures.append(f"{where}: {label} ({pat})")
            cited_tiers = {f["tier"] for f in facts if f["tier"]}
            # A demotion acknowledgment legitimately names the tier
            # the fact dropped FROM; that is not an upgrade.
            cited_tiers |= {f["demotion_from"] for f in facts
                            if f.get("demotion_from")}
            mentioned = set()
            for t in _TIER_WORDS:
                labels = (t.replace("_", " "),) + _TIER_LABELS.get(t, ())
                if any(lbl in low for lbl in labels):
                    mentioned.add(t)
            upgraded = mentioned - cited_tiers
            if upgraded:
                failures.append(
                    f"{where}: states tier(s) {sorted(upgraded)} not "
                    f"held by cited facts {sorted(cited_tiers)}")
            card_tiers = [f["tier"] for f in facts if f["kind"] == "card"]
            if name == "headline_findings" and any(
                    t in ("no_pattern", "insufficient_data")
                    for t in card_tiers):
                failures.append(
                    f"{where}: headline claim rests on a no-pattern or "
                    f"insufficient-data card")
            if name == "ruled_out" and any(
                    t in ("solid", "emerging") for t in card_tiers):
                failures.append(
                    f"{where}: ruled-out claim rests on a live card")
            if any(f["demotion"] for f in facts) and not any(
                    w in low for w in ("demot", "drop", "fell",
                                       "no longer")):
                failures.append(
                    f"{where}: cites a demoted fact without "
                    f"acknowledging the demotion")
    return failures


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You are the analyst for Stridewise, a personal training-data
validation app. Statistics have already decided what is true. Your job
is only to explain what the facts mean and connect them. You must never
invent a number, upgrade a tier, or soften a demotion.

You will receive one FACT PACK as JSON. It is the only evidence you may
use. Write the Analyst Report as a JSON object with exactly three keys:

- "headline_findings": claims about the patterns that cleared the bar
  (cards with tier "solid" or "emerging", and race cards). Lead with
  what matters most.
- "ruled_out": what was checked and did not hold up (cards with tier
  "no_pattern" or "insufficient_data"). State plainly that the data did
  not show a pattern. This section is a feature, not a failure.
- "evidence_needs": for questions the data cannot yet settle, state
  what the EVIDENCE needs (more nights, a missing field, a wider range
  of weeks). Never phrase this as a prediction about the athlete and
  never tell the athlete what to do.

Every claim is an object {"text": "...", "cites": ["A1", ...]}.
Citation IDs: pattern cards are cited by their question ID (for
example "A1"); race cards are cited as QUESTION@DATE (for example
"C1@2026-10-17"); the aggregates block is cited as "AGG".

Hard rules for every claim:
- Every number in the text must be a value carried by one of the cited
  facts (display rounding is fine).
- No causal language: nothing causes, leads to, improves, prevents, or
  will make anything else happen. Describe associations only.
- No medical language and no advice. Never tell the athlete what to do.
- No predictions about the athlete's future results or feelings.
- If you name a tier, it must be the tier of a cited fact. A demoted
  fact must be described as demoted or dropped.
- Keep each claim to one or two sentences, in plain language.

Respond with ONLY the JSON object. No markdown fences, no commentary."""


def build_prompt(pack, failures=None):
    directory = []
    for card in pack.get("cards", []):
        directory.append(f"{card['question']} (tier {card.get('tier')})")
    for rc in pack.get("race_cards", []):
        directory.append(
            f"{rc['question']}@{rc['date']} (race: {rc.get('race')})")
    directory.append("AGG (aggregates)")
    user = (
        "CITATION IDS:\n" + "\n".join(directory)
        + "\n\nFACT PACK:\n" + json.dumps(pack)
    )
    if failures:
        user += (
            "\n\nYour previous draft was rejected by the validator for "
            "these reasons:\n" + "\n".join(f"- {f}" for f in failures)
            + "\nWrite a corrected report that fixes every one of them."
        )
    return _SYSTEM, user


# ---------------------------------------------------------------------------
# API call (through the connector; no credential handling here)
# ---------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def call_spark(system, user, model=MODEL, max_tokens=6000):
    """One chat completion on the standard tier. Returns
    (content_text, usage_dict, finish_reason). The token budget is
    generous on purpose: this model's reasoning tokens count against
    max_tokens, and a tight cap returns empty content (measured on the
    first live attempt, 2026-11-02). Raises on transport or API
    errors."""
    if "contributor" in model:
        raise ValueError("contributor tier is banned for app calls")
    if model != MODEL:
        raise ValueError(f"narrative layer is pinned to {MODEL}")
    sys.path.insert(0, _CRED_BIN)
    from dynamic_credentials import add_surrogate_to_request, \
        read_json_response
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        API_BASE + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    add_surrogate_to_request(req, CONNECTOR, allowed_hosts=API_HOSTS)
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(req, timeout=180) as resp:
        body = read_json_response(resp)
    choice = body["choices"][0]
    content = choice["message"].get("content")
    usage = body.get("usage", {})
    return content, {
        "prompt_tokens": usage.get("prompt_tokens",
                                   usage.get("input_tokens")),
        "completion_tokens": usage.get("completion_tokens",
                                       usage.get("output_tokens")),
    }, choice.get("finish_reason")


def _parse_content(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty or non-text content")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Template fallback (deterministic; built only from fact-pack text)
# ---------------------------------------------------------------------------

def template_fallback(pack):
    by_tier = {"solid": [], "emerging": [], "no_pattern": [],
               "insufficient_data": []}
    for card in pack.get("cards", []):
        by_tier.setdefault(card.get("tier"), []).append(card)
    headlines = [{"text": c["sentence"], "cites": [c["question"]]}
                 for c in by_tier["solid"] + by_tier["emerging"]]
    if not headlines:
        headlines = [{
            "text": "No pattern cleared the solid or emerging bar in "
                    "this refresh.",
            "cites": ["AGG"],
        }]
    ruled = [{"text": c["sentence"], "cites": [c["question"]]}
             for c in by_tier["no_pattern"] + by_tier["insufficient_data"]]
    needs = []
    for c in by_tier["insufficient_data"]:
        needs.append({
            "text": f"Not yet testable: binding constraint was "
                    f"\"{c.get('binding_rule', 'insufficient data')}\".",
            "cites": [c["question"]],
        })
    for c in by_tier["solid"] + by_tier["emerging"] \
            + by_tier["no_pattern"]:
        rule = c.get("binding_rule") or ""
        if "cap" in rule or "below solid bar" in rule:
            needs.append({
                "text": f"More evidence would sharpen this: binding "
                        f"constraint was \"{rule}\".",
                "cites": [c["question"]],
            })
    if not needs:
        needs = [{
            "text": "Every testable question met its evidence bar in "
                    "this refresh.",
            "cites": ["AGG"],
        }]
    return {"headline_findings": headlines, "ruled_out": ruled,
            "evidence_needs": needs}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_report(pack, call=call_spark, max_attempts=2):
    """Returns (narrative_dict, usages). usages holds one entry per API
    call actually made, each {model, prompt_tokens, completion_tokens,
    purpose}."""
    index = build_index(pack)
    usages = []
    failures = None
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        system, user = build_prompt(pack, failures)
        content, usage, finish = call(system, user)
        usage = dict(usage, model=MODEL,
                     purpose=f"analyst report attempt {attempts}")
        usages.append(usage)
        try:
            sections = _parse_content(content)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failures = [
                f"output was not usable (finish_reason={finish}): "
                f"{exc}"]
            continue
        failures = validate_narrative(sections, index)
        if not failures:
            return _wrap(pack, sections, "spark", attempts, []), usages
    sections = template_fallback(pack)
    return _wrap(pack, sections, "template", attempts,
                 failures or []), usages


def _wrap(pack, sections, source, attempts, failures):
    if set(sections) != set(SECTIONS):
        sections = {k: sections.get(k, []) for k in SECTIONS}
    return {
        "athlete_id": pack.get("athlete", {}).get("id"),
        "display_label": pack.get("athlete", {}).get("display_label"),
        "generated_on": pack.get("generated_on"),
        "window_anchor": pack.get("window_anchor"),
        "model": MODEL,
        "source": source,
        "attempts": attempts,
        "validation": {"passed": source == "spark",
                       "final_failures": failures},
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Spend log (format matches ~/workspace/hackathon/spend-log.md)
# ---------------------------------------------------------------------------

_RATE_IN, _RATE_OUT = 1.25 / 1e6, 4.25 / 1e6


def _est_cost(usage):
    tin = usage.get("prompt_tokens") or 0
    tout = usage.get("completion_tokens") or 0
    return tin * _RATE_IN + tout * _RATE_OUT


def append_spend_log(path, entries, when=None):
    """Append one row per call plus a running-totals line. Returns the
    new (row_count, total_tokens, total_cost)."""
    path = Path(path)
    text = path.read_text() if path.exists() else ""
    rows = [ln for ln in text.splitlines()
            if re.match(r"\|\s*\d+\s*\|", ln)]
    n = len(rows)
    total_tokens, total_cost = 0, 0.0
    for ln in rows:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        try:
            total_tokens += int(cells[4]) + int(cells[5])
            total_cost += float(cells[6].replace("~$", "").replace("$", ""))
        except (IndexError, ValueError):
            pass
    when = when or datetime.now(ZoneInfo("America/New_York"))
    lines = []
    for e in entries:
        n += 1
        tin = e.get("prompt_tokens") or 0
        tout = e.get("completion_tokens") or 0
        cost = _est_cost(e)
        total_tokens += tin + tout
        total_cost += cost
        lines.append(
            f"| {n} | {when:%Y-%m-%d} | {e['purpose']} | {e['model']} "
            f"| {tin} | {tout} | ~${cost:.4f} |")
    lines.append(
        f"\nRunning totals after #{n}: {total_tokens} tokens, "
        f"~${total_cost:.2f} of $150 (estimates at third-party "
        f"standard-tier rates).")
    with path.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    return n, total_tokens, total_cost


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stridewise Analyst Report")
    ap.add_argument("factpack")
    ap.add_argument("out")
    ap.add_argument("--spend-log")
    ap.add_argument("--max-attempts", type=int, default=2)
    args = ap.parse_args(argv)
    pack = json.loads(Path(args.factpack).read_text())
    narrative, usages = generate_report(pack,
                                        max_attempts=args.max_attempts)
    Path(args.out).write_text(json.dumps(narrative, indent=2))
    if args.spend_log:
        append_spend_log(args.spend_log, usages)
    print(json.dumps({
        "source": narrative["source"],
        "attempts": narrative["attempts"],
        "validation_failures": narrative["validation"]["final_failures"],
        "usages": usages,
        "out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
