"""Hypothesis proposals: Spark proposes, the gate decides.

Spark's third designed job (Spark-layer brief): read one profile's
fact pack and propose NEW association questions for the board's
"being tested" section. The discipline is the same as everywhere
else in Stridewise:

- Spark proposes candidates as structured specs, never prose claims.
  A candidate is a SINGLE driver built from the profile's existing
  series with one transform from a fixed menu (lags, rolling means,
  threshold splits, weekend/weekday splits). No multi-driver
  interactions, no steps, no raw pace variables, no fields the
  profile does not carry.
- The module constructs the proposed series itself, from the same
  normalized data and with the same sensor-exclusion handling as the
  registry and ask builders, and runs every constructible candidate
  through the identical evidence gate (engine.stats.evaluate: same
  floors, density/recency caps, week-block bootstrap ranges).
- Candidates that duplicate a standing registry question or an
  already-evaluated ask pair -- including under a transform
  equivalence, e.g. "previous night's sleep length" IS sleep length
  at a one-night lag -- are recorded as rejected with the reason, as
  are algebraically coupled pairs (a driver built from the outcome
  night's own sleep length, tested against that length).
- Spark's one-line rationale per candidate is validated before it
  is stored: numbers must trace to the fact pack or the menu, and
  the shared language lints apply. A rationale that fails twice is
  withheld (rationale_status records why); the candidate's gated
  result stands on its own either way.
- If Spark's output is unusable twice, a small rule-chosen candidate
  set is evaluated instead and the file is labeled
  proposed_by="template". Unvalidated text is never stored.

Run: python -m engine.proposals <normalized.json> <factpack.json>
     [out.json] [--anchor YYYY-MM-DD] [--spend-log PATH]
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

from . import ask as ask_mod
from .factpack import _pinned_view
from .model import load_normalized, parse_date
from .narrative import (_LINTS, MODEL, _fact_numbers, _number_traces,
                       _parse_content, _text_numbers, append_spend_log,
                       call_spark)
from .registry import A14_CAVEAT, _sentence
from .skeptic import _EXTRA_LINTS
from .stats import evaluate
from .weekly import build_weeks

_ALL_LINTS = _LINTS + _EXTRA_LINTS

# ------------------------------------------------------------- the menu

NIGHTLY_TRANSFORMS = ("lag1", "lag2", "roll3", "thresh",
                      "weekend_only", "weekday_only")
WEEKLY_TRANSFORMS = ("lag1", "roll2")

TRANSFORM_DESC = {
    "lag1": "the driver's value one night earlier (weekly grain: the "
            "value from one week earlier than the board's pairing)",
    "lag2": "the driver's value two nights earlier (nightly only)",
    "roll3": "the mean of the driver's values over the three nights "
             "ending that night (nightly only)",
    "roll2": "the mean of the driver's values over that week and the "
             "week before (weekly only)",
    "thresh": "a yes/no split of nights at a fixed cut on the driver "
              "(cuts are defined per driver, listed in the menu)",
    "weekend_only": "the pair computed on weekend nights only",
    "weekday_only": "the pair computed on weekday nights only",
}

# driver -> (cut in raw units, side, dh phrase, dl phrase, cut label)
THRESHOLDS = {
    "sleep_duration": (390.0, "lt", "short-sleep nights",
                       "longer-sleep nights", "under 6.5 hours"),
    "bedtime": (180.0, "ge", "after-midnight bedtime nights",
                "before-midnight bedtime nights", "at or after midnight"),
    "awake_min": (30.0, "ge", "high-awake nights", "low-awake nights",
                  "30 minutes or more awake"),
    "midpoint": (390.0, "ge", "late-midpoint nights",
                 "early-midpoint nights", "at or after 3:30 AM"),
    "bedtime_variability": (30.0, "ge", "scattered-bedtime nights",
                            "steady-bedtime nights",
                            "30 minutes or more of scatter"),
}

# Numbers a rationale may state besides fact-pack numbers: the menu's
# own constants (lags, window sizes, cuts and their hour forms).
_MENU_NUMBERS = {1.0, 2.0, 3.0, 6.5, 30.0, 180.0, 390.0}

# Series whose construction uses the outcome night's own sleep
# length (the length itself, the midpoint built from it, and the
# stage shares that divide by it). Tested against that same night's
# length they partly restate a definition, as in the ask layer's
# coupled pairs.
_LENGTH_DERIVED = {"sleep_duration", "midpoint", "rem_share",
                   "light_share", "deep_share"}
_INCLUDES_CURRENT_NIGHT = {"identity", "roll3", "thresh",
                           "weekend_only", "weekday_only"}


def _root(driver):
    """prev_duration is sleep length at a one-night lag; normalize
    so duplicate detection sees through the alias."""
    return "sleep_duration" if driver == "prev_duration" else driver


def _base_transform(driver):
    return "lag1" if driver == "prev_duration" else "identity"


def covered_pairs(ask_pack):
    """The set of (root_driver, transform, outcome) already answered
    by the board (registry) or the ask layer, normalized so a
    proposal cannot re-ask one under a transform alias."""
    covered = set()
    sources = {}
    for (d, o), qid in ask_mod.REGISTRY_PAIRS.items():
        key = (_root(d), _base_transform(d), o)
        covered.add(key)
        sources[key] = f"registry question {qid}"
    for p in ask_pack.get("pairs", []):
        d, o = p["driver"], p["outcome"]
        key = (_root(d), _base_transform(d), o)
        covered.add(key)
        sources.setdefault(key, "an already-evaluated ask pair")
    return covered, sources


def classify(spec, avail_drivers, avail_outcomes, covered, sources):
    """Validate one candidate spec against the menu and the covered
    set. Returns (ok, reject_reason, grain)."""
    d, t, o = spec.get("driver"), spec.get("transform"), spec.get("outcome")
    if d not in ask_mod._DRIVER_DESC or o not in ask_mod._OUTCOME_DESC:
        return False, "driver or outcome is not in the variable menu", None
    if d not in avail_drivers:
        return False, f"this profile's data does not carry {d}", None
    if o not in avail_outcomes:
        return False, f"this profile's data does not carry {o}", None
    nightly = d in {s[0] for s in ask_mod.NIGHTLY_DRIVERS}
    grain = "nightly" if nightly else "weekly"
    if t == "identity":
        key = (_root(d), _base_transform(d), o)
        if key in covered:
            return False, f"already covered by {sources[key]}", None
        return False, ("the untransformed pair is not a proposal; "
                       "every constructible untransformed pair is "
                       "already tested or untestable by construction"), None
    menu = NIGHTLY_TRANSFORMS if nightly else WEEKLY_TRANSFORMS
    if t not in menu:
        return False, (f"transform {t!r} is not in the {grain} menu "
                       f"({', '.join(menu)})"), None
    if d == "weekend":
        return False, ("the weekend indicator takes no transforms; "
                       "weekend splits of another driver are the "
                       "constructible form"), None
    if t == "thresh" and d not in THRESHOLDS:
        return False, f"no fixed cut is defined for {d}", None
    # Same measurement / algebraic coupling, mirroring the ask rules.
    root = _root(d)
    if o == "duration" and root in _LENGTH_DERIVED \
            and t in _INCLUDES_CURRENT_NIGHT:
        return False, ("algebraically coupled: the driver is built "
                       "from that night's sleep length, so the pair "
                       "partly compares a measurement with itself"), None
    # Duplicate under normalization: compose the candidate's own
    # transform with the alias base transform of prev_duration.
    if d == "prev_duration":
        composed = {"lag1": "lag2"}.get(t, t)
        key = ("sleep_duration", composed, o)
    else:
        key = (d, t, o)
    if key in covered:
        return False, f"already covered by {sources[key]}", None
    return True, None, grain


# ------------------------------------------------------- series building

def _transformed_nightly(athlete, dkey, transform):
    """(values, blocked) per wake date for one transformed nightly
    driver. blocked marks pair dates whose construction touches a
    sensor-excluded night (the A13 previous-night rule, generalized
    to every source night in the transform window)."""
    vals = ask_mod._night_driver_values(athlete, dkey)
    by_date = {n.wake_date: n for n in athlete.sleep_nights}

    def excluded(d):
        n = by_date.get(d)
        return bool(n and n.excluded_sensor_unreliable)

    out, blocked = {}, {}
    for n in athlete.sleep_nights:
        d = n.wake_date
        if transform == "lag1":
            src = [d - timedelta(days=1)]
        elif transform == "lag2":
            src = [d - timedelta(days=2)]
        elif transform == "roll3":
            src = [d - timedelta(days=2), d - timedelta(days=1), d]
        else:  # thresh / weekend_only / weekday_only use that night
            src = [d]
        if transform == "roll3":
            vs = [vals.get(s) for s in src]
            v = (sum(vs) / len(vs)) if all(x is not None for x in vs) \
                else None
        elif transform == "thresh":
            cut, side = THRESHOLDS[dkey][0], THRESHOLDS[dkey][1]
            base = vals.get(d)
            if base is None:
                v = None
            elif side == "lt":
                v = 1.0 if base < cut else 0.0
            else:
                v = 1.0 if base >= cut else 0.0
        else:
            v = vals.get(src[0])
        out[d] = v
        blocked[d] = any(excluded(s) for s in src)
    return out, blocked


def _nightly_pairs(athlete, dkey, transform, okey):
    """(usable, collected) for one transformed nightly pair,
    mirroring ask._nightly_pairs' outcome handling."""
    daily = athlete.daily_by_date
    xs, blocked = _transformed_nightly(athlete, dkey, transform)
    usable, collected = [], []
    for n in athlete.sleep_nights:
        d = n.wake_date
        if transform == "weekend_only" and d.weekday() < 5:
            continue
        if transform == "weekday_only" and d.weekday() >= 5:
            continue
        x = xs.get(d)
        if x is None:
            continue
        src_blocked = blocked.get(d, False)
        if okey in ("rhr", "hrv"):
            row = daily.get(d)
            y = (row.resting_hr_bpm if okey == "rhr" else row.hrv_ms) \
                if row else None
            if y is None:
                continue
            collected.append((d, x, y))
            if not n.excluded_sensor_unreliable and not src_blocked:
                usable.append((d, x, y))
        elif okey == "run_quality":
            pace = athlete.day_run_pace(d, include_excluded=True)
            pace_u = athlete.day_run_pace(d)
            if pace is None:
                continue
            collected.append((d, x, pace))
            if (not n.excluded_sensor_unreliable and not src_blocked
                    and pace_u is not None):
                usable.append((d, x, pace_u))
        elif okey == "duration":
            collected.append((d, x, n.duration_min))
            if not n.excluded_sensor_unreliable and not src_blocked:
                usable.append((d, x, n.duration_min))
    return usable, collected


def _week_driver_value(weeks, i, dkey):
    """One weekly driver's value at week i, mirroring
    ask._weekly_pairs (volatility looks one week back)."""
    w = weeks[i]
    if dkey == "mileage":
        return w.mileage_m / 1000.0
    if dkey == "acwr":
        return w.acwr
    if dkey == "monotony":
        return w.monotony
    if dkey == "long_run":
        return w.long_run_m / 1000.0 if w.long_run_m is not None else None
    if dkey == "volatility":
        if i == 0 or weeks[i - 1].mileage_m <= 0:
            return None
        return abs(w.mileage_m - weeks[i - 1].mileage_m) \
            / weeks[i - 1].mileage_m
    raise KeyError(dkey)


def _weekly_pairs(athlete, weeks, dkey, transform, okey):
    """(usable, collected) for one transformed weekly pair. The
    board's pairing is driver week i -> outcome week i+1; lag1 takes
    the driver one week earlier still, roll2 averages the driver
    over weeks i-1 and i. The pair date is week i's Monday."""
    usable = []
    for i in range(1, len(weeks) - 1):
        nxt = weeks[i + 1]
        y = nxt.mean_rhr if okey == "next_rhr" else nxt.mean_pace
        if y is None:
            continue
        if transform == "lag1":
            x = _week_driver_value(weeks, i - 1, dkey)
        else:  # roll2
            a = _week_driver_value(weeks, i - 1, dkey)
            b = _week_driver_value(weeks, i, dkey)
            x = (a + b) / 2.0 if a is not None and b is not None else None
        if x is None:
            continue
        usable.append((weeks[i].monday, x, y))
    return usable, list(usable)


# ------------------------------------------------------------- evaluation

_NIGHT_D = {d[0]: d for d in ask_mod.NIGHTLY_DRIVERS}
_WEEK_D = {d[0]: d for d in ask_mod.WEEKLY_DRIVERS}
_NIGHT_O = {o[0]: o for o in ask_mod.NIGHTLY_OUTCOMES}
_WEEK_O = {o[0]: o for o in ask_mod.WEEKLY_OUTCOMES}


def _phrases(dspec, transform, grain):
    """(question-title driver phrase, dh, dl, scale) for a spec."""
    dkey, dtitle, _dlab, dh, dl, scale = dspec
    unit = "nights" if grain == "nightly" else "weeks"
    if transform == "lag1":
        return dtitle, f"{unit} after {dh}", f"{unit} after {dl}", scale
    if transform == "lag2":
        return dtitle, f"nights two days after {dh}", \
            f"nights two days after {dl}", scale
    if transform == "roll3":
        return dtitle, f"3-night stretches of {dh}", \
            f"3-night stretches of {dl}", scale
    if transform == "roll2":
        return dtitle, f"two-week stretches of {dh}", \
            f"two-week stretches of {dl}", scale
    if transform == "thresh":
        _cut, _side, tdh, tdl, _label = THRESHOLDS[dkey]
        return dtitle, tdh, tdl, 1.0
    if transform in ("weekend_only", "weekday_only"):
        which = "weekend" if transform == "weekend_only" else "weekday"
        return dtitle, f"{which} {dh}", f"{which} {dl}", scale
    return dtitle, dh, dl, scale


def _lower_title(dtitle):
    """Lowercase a driver title for mid-sentence use, keeping
    acronym-led titles ("REM sleep share") intact."""
    first = dtitle.split(" ", 1)[0]
    return dtitle if first.isupper() else dtitle.lower()


def _question_text(dspec, ospec, transform, grain):
    dtitle = dspec[1]
    otitle = ospec[1]
    if transform == "lag1":
        lag = "one-night lag" if grain == "nightly" else "two-week lag"
        return f"{dtitle} ({lag}) and {otitle}"
    if transform == "lag2":
        return f"{dtitle} (two-night lag) and {otitle}"
    if transform == "roll3":
        return f"Three-night average {_lower_title(dtitle)} and {otitle}"
    if transform == "roll2":
        return f"Two-week average {_lower_title(dtitle)} and {otitle}"
    if transform == "thresh":
        return f"{dtitle} ({THRESHOLDS[dspec[0]][4]}) and {otitle}"
    if transform == "weekend_only":
        return f"{dtitle} on weekend nights and {otitle}"
    if transform == "weekday_only":
        return f"{dtitle} on weekday nights and {otitle}"
    return f"{dtitle} and {otitle}"


def evaluate_candidate(athlete, weeks, spec, grain):
    """Construct the pair and run the gate. Returns the evaluated
    candidate record (status "evaluated")."""
    dkey, okey = spec["driver"], spec["outcome"]
    transform = spec["transform"]
    dspec = (_NIGHT_D if grain == "nightly" else _WEEK_D)[dkey]
    ospec = (_NIGHT_O if grain == "nightly" else _WEEK_O)[okey]
    if grain == "nightly":
        usable, collected = _nightly_pairs(athlete, dkey, transform, okey)
        unit_label, window_label = "nights", "180 days"
    else:
        usable, collected = _weekly_pairs(athlete, weeks, dkey,
                                          transform, okey)
        unit_label, window_label = "weeks", "26 weeks"
    _t, dh, dl, scale = _phrases(dspec, transform, grain)
    okind = ospec[4]
    floor_spec = ask_mod._FLOORS[okind]
    if floor_spec[0] == "pace_pct":
        ys = [y for _, _, y in usable]
        floor = 0.01 * statistics.mean(ys) if ys else 0.0
    else:
        floor = floor_spec[1]
    ev = evaluate(usable, scale, floor, grain, athlete.anchor_date,
                  collected_pairs=collected)
    question_text = _question_text(dspec, ospec, transform, grain)
    q = (f"PROP-{dkey}-{transform}-{okey}", "proposal", grain,
         question_text, dh, dl, ospec[3], okind, unit_label,
         window_label, scale, floor_spec)
    card = {"tier": ev["tier"], "n": ev["n"], "effect": ev["effect"],
            "likely_range": ev["likely_range"],
            "binding_rule": ev["binding_rule"], "demotion": None}
    sentence = _sentence(q, card)
    if dkey in ask_mod.STAGE_DRIVERS:
        sentence += " " + A14_CAVEAT
    return {
        "question_text": question_text,
        "driver_spec": {"driver": dkey, "transform": transform},
        "outcome_spec": {"outcome": okey},
        "status": "evaluated",
        "tier": ev["tier"], "effect": ev["effect"],
        "range": ev["likely_range"], "n": ev["n"],
        "binding_rule": ev["binding_rule"],
        "sentence": sentence,
    }


# ---------------------------------------------------------------- prompt

_SYSTEM = """You are the hypothesis proposer for Stridewise, a personal
training-data validation app. A deterministic statistics engine has
already tested a standing set of questions against this athlete's
data. Your job is to propose NEW candidate questions for the board's
"being tested" section. A statistics gate -- not you -- decides
what every candidate shows; a candidate that turns up nothing is a
normal, useful outcome.

Propose 4 to 6 candidates. Each candidate is ONE driver, built from
ONE existing series with ONE transform from this menu:

- lag1: the driver's value one night earlier (weekly grain: from
  one week earlier than the standard pairing)
- lag2: the driver's value two nights earlier (nightly only)
- roll3: the mean of the driver's values over the three nights
  ending that night (nightly only)
- roll2: the mean of the driver's values over that week and the
  week before (weekly only)
- thresh: a yes/no split of nights at a fixed cut. Cuts exist only
  for: sleep_duration (under 6.5 hours), bedtime (at or after
  midnight), awake_min (30 minutes or more), midpoint (at or after
  3:30 AM), bedtime_variability (30 minutes or more of scatter)
- weekend_only / weekday_only: the pair computed on weekend nights
  only, or on weekday nights only

Hard rules:
- Single driver only. No interactions, no combined drivers.
- No steps variables. No raw pace variables (run quality / run pace
  outcomes are allowed; they are on the menu).
- Use only the drivers and outcomes listed for this profile below.
- Do NOT propose the plain (untransformed) form of any pair, and do
  not re-propose anything in the ALREADY TESTED list below -- every
  candidate must use a transform, and duplicates are rejected
  automatically, including under aliases (the "previous night's
  sleep length" variable IS sleep length at a one-night lag).
- The weekend indicator takes no transforms; use weekend_only /
  weekday_only splits of another driver instead.
- Each candidate carries a one-line rationale: why this is worth
  testing. No causal language (never "because", "since", "due to",
  "leads to", "causes", "results in", or claims that one thing
  improves or prevents another). No medical language, no advice, no
  predictions. Any number in a rationale must come from the fact
  pack below or from the menu's own constants.

Respond with ONLY a JSON object:
{"candidates": [{"driver": "...", "transform": "...",
"outcome": "...", "rationale": "..."}, ...]}
No markdown fences, no commentary."""


def build_prompt(pack, ask_pack, failures=None):
    cards = [{
        "question": c["question"], "text": c.get("question_text"),
        "tier": c.get("tier"), "effect": c.get("effect"),
        "range": c.get("likely_range"), "n": c.get("n"),
    } for c in pack.get("cards", [])]
    tested = []
    for (d, o), qid in sorted(ask_mod.REGISTRY_PAIRS.items()):
        tested.append(f"{ask_mod._DRIVER_DESC[d]} -> "
                      f"{ask_mod._OUTCOME_DESC[o]} (board {qid})")
    for p in ask_pack.get("pairs", []):
        tested.append(f"{ask_mod._DRIVER_DESC[p['driver']]} -> "
                      f"{ask_mod._OUTCOME_DESC[p['outcome']]} (asked)")
    drivers = [f"{d['key']}: {ask_mod._DRIVER_DESC[d['key']]}"
               for d in ask_pack.get("drivers", [])]
    outcomes = [f"{o['key']}: {ask_mod._OUTCOME_DESC[o['key']]}"
                for o in ask_pack.get("outcomes", [])]
    user = (
        "AGGREGATES:\n" + json.dumps(pack.get("aggregates", {}))
        + "\n\nSTANDING CARDS (already decided):\n" + json.dumps(cards)
        + "\n\nALREADY TESTED PAIRS (do not re-propose):\n"
        + "\n".join(tested)
        + "\n\nDRIVERS AVAILABLE FOR THIS PROFILE:\n"
        + "\n".join(drivers)
        + "\n\nOUTCOMES AVAILABLE FOR THIS PROFILE:\n"
        + "\n".join(outcomes)
    )
    if failures:
        user += (
            "\n\nYour previous reply was rejected for these reasons:\n"
            + "\n".join(f"- {f}" for f in failures)
            + "\nReply with a corrected JSON object only.")
    return _SYSTEM, user


def validate_proposals(raw, pack):
    """Validate Spark's reply shape and each rationale. Returns
    (candidates, failures): candidates carry a per-candidate
    rationale check; failures are response-level problems worth a
    retry."""
    if not isinstance(raw, dict) or not isinstance(
            raw.get("candidates"), list) or not raw["candidates"]:
        return [], ["output has no candidates list"]
    allowed = _fact_numbers(pack) | _MENU_NUMBERS
    out, failures = [], []
    for i, cand in enumerate(raw["candidates"]):
        if not isinstance(cand, dict):
            out.append({"spec": {"driver": None, "transform": None,
                                 "outcome": None},
                        "rationale": None, "rationale_ok": False,
                        "rationale_failure": "candidate was not an "
                                             "object"})
            continue
        spec = {k: cand.get(k) for k in ("driver", "transform", "outcome")}
        rationale = cand.get("rationale")
        rec = {"spec": spec, "rationale": rationale,
               "rationale_ok": True, "rationale_failure": None}
        if not isinstance(rationale, str) or not rationale.strip():
            rec["rationale_ok"] = False
            rec["rationale_failure"] = "rationale missing"
        else:
            low = rationale.lower()
            for label, pat in _ALL_LINTS:
                if re.search(pat, low):
                    rec["rationale_ok"] = False
                    rec["rationale_failure"] = f"{label} ({pat})"
                    break
            if rec["rationale_ok"]:
                for tok in _text_numbers(rationale):
                    if not _number_traces(tok, allowed):
                        rec["rationale_ok"] = False
                        rec["rationale_failure"] = (
                            f"number {tok:g} does not trace to the "
                            f"fact pack or the menu")
                        break
        if not rec["rationale_ok"]:
            failures.append(
                f"candidate {i} ({spec.get('driver')}/"
                f"{spec.get('transform')}->{spec.get('outcome')}): "
                f"rationale failed validation: "
                f"{rec['rationale_failure']}")
        out.append(rec)
    return out, failures


# ------------------------------------------------------------- fallbacks

_FALLBACK_SPECS = [
    {"driver": "sleep_duration", "transform": "lag2", "outcome": "rhr"},
    {"driver": "sleep_duration", "transform": "lag2",
     "outcome": "duration"},
    {"driver": "bedtime", "transform": "roll3", "outcome": "rhr"},
    {"driver": "sleep_duration", "transform": "thresh",
     "outcome": "rhr"},
    {"driver": "mileage", "transform": "lag1", "outcome": "next_rhr"},
]


def _fallback_candidates():
    return [{"spec": dict(s), "rationale": (
        "Rule-chosen transform of an existing series, proposed for "
        "testing after the model's proposals failed validation."),
        "rationale_ok": True, "rationale_failure": None}
        for s in _FALLBACK_SPECS]


# ----------------------------------------------------------- orchestration

def generate_proposals(pack, athlete, ask_pack, call=None,
                       max_attempts=2):
    """Returns (proposals_dict, usages). usages holds one entry per
    API call actually made."""
    if call is None:
        call = functools.partial(call_spark, max_tokens=4000,
                                 reasoning_effort="low")
    weeks = build_weeks(athlete)
    avail_d = [d["key"] for d in ask_pack.get("drivers", [])]
    avail_o = [o["key"] for o in ask_pack.get("outcomes", [])]
    covered, sources = covered_pairs(ask_pack)
    usages = []
    chosen, failures, source = None, None, "spark"
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        system, user = build_prompt(pack, ask_pack, failures)
        content, usage, finish = call(system, user)
        usages.append(dict(usage, model=MODEL, purpose=(
            f"hypothesis proposals ({pack.get('athlete', {}).get('id')})"
            f" attempt {attempts}")))
        if not isinstance(content, str) or not content.strip():
            failures = [f"empty completion (finish_reason={finish})"]
            continue
        try:
            raw = _parse_content(content)
        except (ValueError, TypeError) as exc:
            failures = [f"output was not usable JSON: {exc}"]
            continue
        got, failures = validate_proposals(raw, pack)
        if got:
            chosen = got
            if not failures:
                break
            # Candidates exist but some rationales failed: the loop
            # retries once with the failures fed back; if this was
            # the last attempt, the set stands and the bad rationales
            # are withheld per-candidate below.
    if not chosen:
        chosen = _fallback_candidates()
        source = "template"
    records = []
    for cand in chosen:
        spec = cand["spec"]
        ok, reason, grain = classify(spec, avail_d, avail_o, covered,
                                     sources)
        if not ok:
            records.append({
                "question_text": None,
                "driver_spec": {"driver": spec.get("driver"),
                                "transform": spec.get("transform")},
                "outcome_spec": {"outcome": spec.get("outcome")},
                "rationale": cand["rationale"] if cand["rationale_ok"]
                else None,
                "rationale_status": ("ok" if cand["rationale_ok"]
                                     else cand["rationale_failure"]),
                "status": "rejected", "reject_reason": reason,
            })
            continue
        rec = evaluate_candidate(athlete, weeks, spec, grain)
        rec["rationale"] = cand["rationale"] if cand["rationale_ok"] \
            else None
        rec["rationale_status"] = ("ok" if cand["rationale_ok"]
                                   else cand["rationale_failure"])
        records.append(rec)
    return {
        "athlete": {"id": athlete.id,
                    "display_label": athlete.display_label,
                    "source": athlete.source},
        "generated_on": pack.get("generated_on"),
        "window_anchor": pack.get("window_anchor"),
        "proposed_by": source,
        "attempts": attempts,
        "candidates": records,
    }, usages


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stridewise hypothesis proposals")
    ap.add_argument("normalized")
    ap.add_argument("factpack")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--anchor")
    ap.add_argument("--spend-log")
    args = ap.parse_args(argv)
    pack = json.loads(Path(args.factpack).read_text())
    athlete = load_normalized(args.normalized)
    anchor = args.anchor or pack.get("window_anchor")
    if anchor:
        athlete = _pinned_view(athlete, parse_date(anchor))
    ask_pack = ask_mod.build_ask(athlete,
                                 generated_on=pack.get("generated_on"))
    proposals, usages = generate_proposals(pack, athlete, ask_pack)
    out = args.out or str(Path(args.normalized).with_name(
        "proposals.json"))
    Path(out).write_text(json.dumps(proposals, indent=2))
    if args.spend_log:
        append_spend_log(args.spend_log, usages)
    print(json.dumps({
        "proposed_by": proposals["proposed_by"],
        "attempts": proposals["attempts"],
        "candidates": [{
            "spec": f"{c['driver_spec']['driver']}/"
                    f"{c['driver_spec']['transform']}->"
                    f"{c['outcome_spec']['outcome']}",
            "status": c["status"],
            "tier": c.get("tier"),
            "reason": c.get("reject_reason"),
        } for c in proposals["candidates"]],
        "calls": len(usages), "out": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
